import gc
import spacy
import torch
import pandas as pd
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)

# 1. Load BLIP-2 Flan-T5-XL FIRST with low_cpu_mem_usage=True to avoid Windows OS Error 1455
vqa_model_id = "Salesforce/blip2-flan-t5-xl"
print(f"Loading {vqa_model_id} in 4-bit on {torch.cuda.get_device_name(0)}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
processor = Blip2Processor.from_pretrained(vqa_model_id)
vqa_model = Blip2ForConditionalGeneration.from_pretrained(
    vqa_model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
vqa_model.eval()
gc.collect()

# 2. Load spaCy + T5 Question Generator on CPU (keeps GPU VRAM free for BLIP-2)
print("Loading spaCy and T5 Question Generator on CPU...")
nlp = spacy.load("en_core_web_sm")
qg_model_id = "mrm8488/t5-base-finetuned-question-generation-ap"
qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_id, use_fast=False)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_id, low_cpu_mem_usage=True)
qg_model.eval()
gc.collect()

yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]


def generate_qas(caption: str):
    doc = nlp(caption.strip())
    spans = []
    seen = set()

    # Extract noun chunks and descriptive adjectives/numbers
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

    qas = []
    seen_q = set()
    for ans in spans[:8]:  # Cap at 8 questions per caption for snappy interactive latency
        qg_input = f"answer: {ans} context: {caption}"
        enc = qg_tokenizer(qg_input, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out_ids = qg_model.generate(**enc, max_new_tokens=32, num_beams=2)
        q_text = qg_tokenizer.decode(out_ids[0], skip_special_tokens=True)
        q_text = q_text.replace("question:", "").strip()
        if q_text and q_text.lower() not in seen_q:
            seen_q.add(q_text.lower())
            qas.append({"question": q_text, "answer": ans})
    return qas


def evaluate_vq2(image, caption, threshold):
    if image is None or not caption or not caption.strip():
        return "Please provide both an image and a caption.", pd.DataFrame()

    raw_image = image.convert("RGB")
    qas = generate_qas(caption)
    if not qas:
        return "Could not extract candidate questions from caption.", pd.DataFrame()

    prompts = [
        f"Question: {qa['question']} Is '{qa['answer']}' the correct answer based on the image? Answer yes or no:"
        for qa in qas
    ]
    inputs = processor(
        images=[raw_image] * len(prompts),
        text=prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda", torch.float16)

    with torch.no_grad():
        outputs = vqa_model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

    logits = outputs.scores[0]
    rows = []
    probs = []
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

    mean_score = sum(probs) / len(probs)
    min_score = min(probs)
    verdict = "ALIGNED" if mean_score >= threshold else "MISALIGNED"

    summary_md = (
        f"### Verdict: **{verdict}**\n"
        f"- **Mean VQ² Score:** `{mean_score:.4f}` (Threshold: `{threshold:.2f}`)\n"
        f"- **Min VQ² Score (Strict Conjunction):** `{min_score:.4f}`\n"
        f"- **Questions Evaluated:** `{len(rows)}`"
    )
    return summary_md, pd.DataFrame(rows)


with gr.Blocks(title="SeeTRUE VQ² Live Alignment Evaluator") as demo:
    gr.Markdown(
        "# SeeTRUE — Zero-Shot $VQ^2$ Text-Image Alignment Demo\n"
        "Upload an image and enter a caption to decompose the text into atomic QA pairs and verify each claim with **BLIP-2 Flan-T5-XL (4-bit)**."
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Input Image")
            cap_input = gr.Textbox(
                lines=2,
                label="Caption / Text Prompt",
                placeholder="e.g., A zebra to the right of a fire hydrant.",
            )
            thresh_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.55, step=0.05, label="Alignment Threshold"
            )
            run_btn = gr.Button("Evaluate Alignment", variant="primary")
        with gr.Column(scale=1):
            summary_out = gr.Markdown(label="Alignment Summary")
            table_out = gr.Dataframe(
                headers=["Generated Question", "Candidate Answer", "P(Yes)", "Status"],
                interactive=False,
                label="Per-Question VQ² Verification Breakdown",
            )

    run_btn.click(
        fn=evaluate_vq2,
        inputs=[img_input, cap_input, thresh_slider],
        outputs=[summary_out, table_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)