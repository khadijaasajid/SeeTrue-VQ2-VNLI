import os
import gc
import time
import spacy
import torch
import pandas as pd
import gradio as gr
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------
# 1. Load Shared 4-bit BLIP-2 Flan-T5-XL + 10k VNLI QLoRA Adapter
# ---------------------------------------------------------
VQA_MODEL_ID = "Salesforce/blip2-flan-t5-xl"
ADAPTER_DIR = "vnli_adapter_b7_latest"

print(f"Loading {VQA_MODEL_ID} + {ADAPTER_DIR} on {torch.cuda.get_device_name(0)}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
processor = Blip2Processor.from_pretrained(VQA_MODEL_ID)
base_model = Blip2ForConditionalGeneration.from_pretrained(
    VQA_MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
gc.collect()

# ---------------------------------------------------------
# 2. Load spaCy + T5 Question Generator on CPU (0 GPU VRAM)
# ---------------------------------------------------------
print("Loading spaCy and T5 Question Generator on CPU...")
nlp = spacy.load("en_core_web_sm")
QG_MODEL_ID = "mrm8488/t5-base-finetuned-question-generation-ap"
qg_tokenizer = AutoTokenizer.from_pretrained(QG_MODEL_ID, use_fast=False)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL_ID, low_cpu_mem_usage=True)
qg_model.eval()
gc.collect()

yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]
decoder_start_id = model.config.text_config.decoder_start_token_id or processor.tokenizer.pad_token_id


# ---------------------------------------------------------
# Core Inference Helpers
# ---------------------------------------------------------
def generate_qas(caption: str):
    doc = nlp(caption.strip())
    spans, seen = [], set()

    for chunk in doc.noun_chunks:
        ans = chunk.text.strip()
        if ans.lower() not in seen and len(ans) > 1:
            seen.add(ans.lower())
            spans.append(ans)
    for tok in doc:
        if tok.pos_ in {"ADJ", "NUM"} and tok.text.lower() not in seen:
            seen.add(tok.text.lower())
            spans.append(tok.text.strip())

    if not spans:
        spans = [caption.strip()]

    qas, seen_q = [], set()
    for ans in spans[:8]:
        qg_input = f"answer: {ans} context: {caption}"
        enc = qg_tokenizer(qg_input, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out_ids = qg_model.generate(**enc, max_new_tokens=32, num_beams=2)
        q_text = qg_tokenizer.decode(out_ids[0], skip_special_tokens=True).replace("question:", "").strip()
        if q_text and q_text.lower() not in seen_q:
            seen_q.add(q_text.lower())
            qas.append({"question": q_text, "answer": ans})
    return qas


def run_vnli_single(raw_image, caption: str) -> float:
    """Runs fine-tuned 10k QLoRA VNLI adapter (LoRA ENABLED)."""
    prompt = f"Question: Does this image entail the description: '{caption}'? Answer yes or no. Answer:"
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    decoder_input_ids = torch.tensor([[decoder_start_id]], device="cuda")

    with torch.no_grad():
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[0, 0, :]
        yn = torch.stack([logits[no_id], logits[yes_id]]).float()
        return float(torch.softmax(yn, dim=-1)[1].item())


def run_vq2_single(raw_image, caption: str, threshold: float):
    """Runs zero-shot VQ2 on the SAME base model with LoRA temporarily DISABLED."""
    qas = generate_qas(caption)
    if not qas:
        return 0.0, 0.0, pd.DataFrame()

    prompts = [
        f"Question: {qa['question']} Is '{qa['answer']}' the correct answer based on the image? Answer yes or no:"
        for qa in qas
    ]
    inputs = processor(
        images=[raw_image] * len(prompts),
        text=prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with model.disable_adapter(), torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

    logits = outputs.scores[0]
    rows, probs = [], []
    for i, qa in enumerate(qas):
        yn = torch.tensor([logits[i, yes_id], logits[i, no_id]], dtype=torch.float32)
        p_yes = float(torch.softmax(yn, dim=0)[0])
        probs.append(p_yes)
        rows.append({
            "Generated Question": qa["question"],
            "Candidate Answer": qa["answer"],
            "P(Yes)": round(p_yes, 4),
            "Status": "PASS" if p_yes >= threshold else "FAIL",
        })

    return sum(probs) / len(probs), min(probs), pd.DataFrame(rows)


# ---------------------------------------------------------
# Tab 1 Callback: Unified Side-by-Side (VQ2 + VNLI + Ensemble)
# ---------------------------------------------------------
def eval_unified(image, caption, alpha, threshold, use_cascade, cascade_tau):
    if image is None or not caption or not caption.strip():
        return "Please upload an image and enter a caption.", pd.DataFrame()

    t0 = time.time()
    raw_image = image.convert("RGB")
    cap = caption.strip()

    vnli_prob = run_vnli_single(raw_image, cap)
    vnli_conf = abs(vnli_prob - 0.5) * 2.0

    if use_cascade and vnli_conf >= cascade_tau:
        final_score = vnli_prob
        vq2_str = "Skipped (Fast-Path)"
        route_str = f"Fast-Path (`VNLI Conf {vnli_conf:.2f} >= {cascade_tau:.2f}`)"
        df_qa = pd.DataFrame([{
            "Generated Question": "Skipped by Confidence-Gated Routing",
            "Candidate Answer": "-",
            "P(Yes)": round(vnli_prob, 4),
            "Status": "FAST-PASS",
        }])
    else:
        vq2_mean, _, df_qa = run_vq2_single(raw_image, cap, threshold)
        final_score = (1.0 - alpha) * vq2_mean + alpha * vnli_prob
        vq2_str = f"`{vq2_mean:.4f}`"
        route_str = f"Weighted Ensemble (`(1 - {alpha:.2f})·VQ² + {alpha:.2f}·VNLI`)"

    elapsed = time.time() - t0
    verdict = "ALIGNED" if final_score >= threshold else "MISALIGNED"

    summary_md = (
        f"## Final Decision: **{verdict}** (`{final_score:.4f}` vs. Threshold `{threshold:.2f}`)\n"
        f"*Total Latency: `{elapsed:.2f}s` on RTX 4050 (6 GB)*\n\n"
        f"| Method | Alignment Score | Role |\n"
        f"| :--- | :---: | :--- |\n"
        f"| **VNLI (10k QLoRA Adapter)** | **`{vnli_prob:.4f}`** | Holistic compositional entailment |\n"
        f"| **VQ² (Zero-Shot QG + VQA)** | {vq2_str} | Atomic question decomposition |\n"
        f"| **SeeTRUE Combined Score** | **`{final_score:.4f}`** | {route_str} |"
    )
    return summary_md, df_qa


# ---------------------------------------------------------
# Tab 2 Callback: Standalone VQ2 Interface
# ---------------------------------------------------------
def eval_vq2_only(image, caption, threshold):
    if image is None or not caption or not caption.strip():
        return "Please provide both an image and a caption.", pd.DataFrame()

    t0 = time.time()
    mean_score, min_score, df_qa = run_vq2_single(image.convert("RGB"), caption.strip(), threshold)
    elapsed = time.time() - t0
    verdict = "ALIGNED" if mean_score >= threshold else "MISALIGNED"

    summary_md = (
        f"### VQ² Verdict: **{verdict}** (`{elapsed:.2f}s`)\n"
        f"- **Mean VQ² Score:** `{mean_score:.4f}` (Threshold: `{threshold:.2f}`)\n"
        f"- **Min VQ² Score (Strict Conjunction):** `{min_score:.4f}`\n"
        f"- **Atomic Questions Evaluated:** `{len(df_qa)}`"
    )
    return summary_md, df_qa


# ---------------------------------------------------------
# Tab 3 Callback: Standalone VNLI Interface
# ---------------------------------------------------------
def eval_vnli_only(image, caption, threshold):
    if image is None or not caption or not caption.strip():
        return "Please provide both an image and a caption."

    t0 = time.time()
    vnli_prob = run_vnli_single(image.convert("RGB"), caption.strip())
    elapsed = time.time() - t0
    conf = abs(vnli_prob - 0.5) * 2.0
    verdict = "ENTAILED (ALIGNED)" if vnli_prob >= threshold else "CONTRADICTED / MISALIGNED"

    return (
        f"### VNLI Verdict: **{verdict}** (`{elapsed:.2f}s`)\n"
        f"- **Entailment Probability `P(Yes)`:** `{vnli_prob:.4f}` (Threshold: `{threshold:.2f}`)\n"
        f"- **Model Confidence `|2p - 1|`:** `{conf:.4f}`\n"
        f"- **Checkpoint:** `vnli_adapter_b7_latest` (10k QLoRA on `BLIP-2 Flan-T5-XL`)"
    )


# Optional local sample examples if Winoground images exist on disk
example_rows = []
for img_p, cap_p in [
    ("../SeeTrue/images/ex_0_img_0.png", "an old person kisses a young person"),
    ("../SeeTrue/images/ex_0_img_0.png", "a young person kisses an old person"),
    ("../SeeTrue/images/ex_3_img_0.png", "an animal watches a person"),
]:
    if os.path.exists(img_p):
        example_rows.append([img_p, cap_p])


# ---------------------------------------------------------
# Build Multi-Tab Gradio App
# ---------------------------------------------------------
with gr.Blocks(title="SeeTRUE: Unified VQ² & VNLI Demo") as demo:
    gr.Markdown(
        "# SeeTRUE — Unified VQ² & VNLI (10k QLoRA) Text-Image Alignment Suite\n"
        "Upload any custom image and caption to run **Zero-Shot VQ²**, **Fine-Tuned VNLI**, or the **Weighted SeeTRUE Ensemble** "
        "on a single shared 4-bit `BLIP-2 Flan-T5-XL` backbone."
    )

    with gr.Tabs():
        # TAB 1: BOTH MODELS SIDE-BY-SIDE
        with gr.TabItem("1. Unified Evaluator (VQ² + VNLI + Ensemble)"):
            with gr.Row():
                with gr.Column(scale=1):
                    u_img = gr.Image(type="pil", label="Upload Custom Image")
                    u_cap = gr.Textbox(lines=2, label="Caption / Text Prompt", placeholder="e.g., an old person kisses a young person")
                    u_alpha = gr.Slider(0.0, 1.0, value=0.80, step=0.05, label="Ensemble Weight α [(1−α)·VQ² + α·VNLI]")
                    u_thresh = gr.Slider(0.10, 0.90, value=0.55, step=0.05, label="Decision Threshold")
                    with gr.Row():
                        u_cascade = gr.Checkbox(value=False, label="Confidence-Gated Routing")
                        u_tau = gr.Slider(0.20, 0.95, value=0.70, step=0.05, label="VNLI Confidence Gate τ")
                    u_btn = gr.Button("Run Both Models (VQ² + VNLI)", variant="primary")
                with gr.Column(scale=1):
                    u_summary = gr.Markdown(label="Combined Alignment Summary")
                    u_table = gr.Dataframe(
                        headers=["Generated Question", "Candidate Answer", "P(Yes)", "Status"],
                        interactive=False,
                        label="VQ² Atomic Question Verification Breakdown",
                    )

            if example_rows:
                gr.Examples(
                    examples=[[r[0], r[1], 0.80, 0.55, False, 0.70] for r in example_rows],
                    inputs=[u_img, u_cap, u_alpha, u_thresh, u_cascade, u_tau],
                    label="Quick Test Examples (Winoground Minimal Pairs)",
                )

            u_btn.click(
                fn=eval_unified,
                inputs=[u_img, u_cap, u_alpha, u_thresh, u_cascade, u_tau],
                outputs=[u_summary, u_table],
            )

        # TAB 2: VQ2 ONLY
        with gr.TabItem("2. VQ² Only (Zero-Shot QA Decomposition)"):
            with gr.Row():
                with gr.Column(scale=1):
                    v2_img = gr.Image(type="pil", label="Upload Custom Image")
                    v2_cap = gr.Textbox(lines=2, label="Caption / Text Prompt", placeholder="e.g., A zebra to the right of a fire hydrant.")
                    v2_thresh = gr.Slider(0.10, 0.90, value=0.55, step=0.05, label="Alignment Threshold")
                    v2_btn = gr.Button("Evaluate VQ² Only", variant="primary")
                with gr.Column(scale=1):
                    v2_summary = gr.Markdown(label="VQ² Summary")
                    v2_table = gr.Dataframe(
                        headers=["Generated Question", "Candidate Answer", "P(Yes)", "Status"],
                        interactive=False,
                        label="Per-Question VQ² Breakdown",
                    )
            v2_btn.click(fn=eval_vq2_only, inputs=[v2_img, v2_cap, v2_thresh], outputs=[v2_summary, v2_table])

        # TAB 3: VNLI ONLY
        with gr.TabItem("3. VNLI Only (10k QLoRA Entailment)"):
            with gr.Row():
                with gr.Column(scale=1):
                    vn_img = gr.Image(type="pil", label="Upload Custom Image")
                    vn_cap = gr.Textbox(lines=2, label="Caption / Text Prompt", placeholder="e.g., A dog jumping over a wooden fence.")
                    vn_thresh = gr.Slider(0.10, 0.90, value=0.55, step=0.05, label="Alignment Threshold")
                    vn_btn = gr.Button("Evaluate VNLI Only", variant="primary")
                with gr.Column(scale=1):
                    vn_summary = gr.Markdown(label="VNLI Entailment Result")
            vn_btn.click(fn=eval_vnli_only, inputs=[vn_img, vn_cap, vn_thresh], outputs=[vn_summary])

if __name__ == "__main__":
    demo.queue().launch(share=True)