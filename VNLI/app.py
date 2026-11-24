import os
import re
import time
import torch
import pandas as pd
import gradio as gr
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)

# Optional spaCy candidate extraction with graceful regex fallback
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None


# ==============================================================================
# 1. CONFIGURATION & FROZEN CHECKPOINTS
# ==============================================================================
BASE_MODEL_ID = "Salesforce/blip2-flan-t5-xl"
QG_MODEL_ID = "mrm8488/t5-base-finetuned-question-generation-ap"

ADAPTERS = {
    "VNLI + ConGen (Full 3-Source: SNLI-VE + COCO-Con + PickaPic-Con)": "vnli_adapter_b8_congen",
    "VNLI Base (10k SNLI-VE Only)": "vnli_adapter_b7_latest",
}
DEFAULT_ADAPTER_LABEL = "VNLI + ConGen (Full 3-Source: SNLI-VE + COCO-Con + PickaPic-Con)"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16


# ==============================================================================
# 2. MODEL LOADING (SINGLE 4-BIT BACKBONE + PEFT ADAPTERS + CPU QG)
# ==============================================================================
print(f"[Startup] Loading 4-bit BLIP-2 backbone on {DEVICE}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

processor = Blip2Processor.from_pretrained(BASE_MODEL_ID)
base_model = Blip2ForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True,
)

# Load available LoRA adapters onto the single frozen 4-bit backbone
loaded_adapters = {}
model = None

for label, path in ADAPTERS.items():
    if os.path.isdir(path):
        adapter_key = os.path.basename(path)
        if model is None:
            model = PeftModel.from_pretrained(base_model, path, adapter_name=adapter_key)
        else:
            model.load_adapter(path, adapter_name=adapter_key)
        loaded_adapters[label] = adapter_key
        print(f"[Startup] Loaded LoRA adapter '{adapter_key}' from '{path}'")

if model is None:
    raise FileNotFoundError(
        "No trained adapter directory found! Expected 'vnli_adapter_b8_congen' in current working directory."
    )

model.eval()
YES_TOKEN_ID = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
NO_TOKEN_ID = processor.tokenizer.encode("no", add_special_tokens=False)[0]

print("[Startup] Loading CPU T5 Question Generator for VQ2...")
qg_tokenizer = AutoTokenizer.from_pretrained(QG_MODEL_ID)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL_ID, low_cpu_mem_usage=True)
qg_model.eval()


# ==============================================================================
# 3. INFERENCE ENGINES (QLoRA-VNLI, ZERO-SHOT VQ2, & SEETRUE ENSEMBLE)
# ==============================================================================
def _norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_answer_spans(caption: str):
    """Extracts atomic semantic spans (entities, noun chunks, verbs/attributes) for VQ2."""
    spans = []
    if _NLP is not None:
        doc = _NLP(caption)
        for chunk in doc.noun_chunks:
            txt = chunk.text.strip()
            if len(txt) > 1 and txt.lower() not in {"a", "an", "the", "it", "this", "that"}:
                spans.append(txt)
        for tok in doc:
            if tok.pos_ in {"VERB", "ADJ", "NUM"} and not tok.is_stop:
                spans.append(tok.text.strip())
    else:
        words = [w.strip(".,!?;:\"'") for w in caption.split()]
        spans = [w for w in words if len(w) > 2]

    # Deduplicate while preserving order
    seen, unique_spans = set(), []
    for sp in spans:
        key = sp.lower()
        if key not in seen:
            seen.add(key)
            unique_spans.append(sp)
    return unique_spans[:6] if unique_spans else [caption.strip()]


def generate_vq2_questions(caption: str):
    """Generates targeted atomic (question, expected_answer) pairs using mrm8488 T5-QG format."""
    spans = extract_answer_spans(caption)
    qa_pairs = []
    seen_q = set()

    with torch.no_grad():
        for span in spans:
            # Exact input format expected by mrm8488/t5-base-finetuned-question-generation-ap
            prompt = f"answer: {span}  context: {caption} </s>"
            inputs = qg_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            out_ids = qg_model.generate(**inputs, max_new_tokens=32, num_beams=2)
            raw_q = qg_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            question = re.sub(r"^question:\s*", "", raw_q, flags=re.IGNORECASE).strip()

            q_key = question.lower()
            if question and q_key not in seen_q:
                seen_q.add(q_key)
                qa_pairs.append((question, span))

    return qa_pairs


def score_vnli(image: Image.Image, caption: str, adapter_label: str) -> float:
    """Computes P(entailment = 'yes') using the selected QLoRA-VNLI adapter."""
    adapter_key = loaded_adapters[adapter_label]
    model.set_adapter(adapter_key)

    prompt = f'Does this image entail the description: "{caption.strip()}"?'
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    decoder_input_ids = torch.tensor([[processor.tokenizer.pad_token_id]], device=DEVICE)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE, enabled=(DEVICE == "cuda")):
            outputs = model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
            logits = outputs.logits[:, 0, :]
            pair_logits = torch.stack([logits[:, YES_TOKEN_ID], logits[:, NO_TOKEN_ID]], dim=-1)
            probs = torch.softmax(pair_logits.float(), dim=-1)

    return float(probs[0, 0].item())


def score_vq2(image: Image.Image, caption: str):
    """
    Computes Zero-Shot VQ2 score by disabling LoRA adapters (pure base BLIP-2)
    and verifying atomic QA pairs extracted from the caption.
    """
    qa_pairs = generate_vq2_questions(caption)
    rows = []
    match_scores = []

    with model.disable_adapter():
        with torch.no_grad():
            for question, expected_ans in qa_pairs:
                prompt = f"Question: {question} Short answer:"
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
                with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE, enabled=(DEVICE == "cuda")):
                    gen_ids = model.generate(**inputs, max_new_tokens=16)
                pred_ans = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

                exp_norm = _norm_text(expected_ans)
                pred_norm = _norm_text(pred_ans)
                stop = {"a", "an", "the", "is", "are", "of", "in", "on"}
                exp_tokens = set(exp_norm.split()) - stop
                pred_tokens = set(pred_norm.split()) - stop

                if not exp_tokens:
                    overlap = 1.0 if exp_norm == pred_norm else 0.0
                elif exp_norm in pred_norm or pred_norm in exp_norm:
                    overlap = 1.0
                else:
                    overlap = len(exp_tokens & pred_tokens) / len(exp_tokens)

                # If lexical overlap is low (e.g., hypernym "man" vs "donald trump"), check visual yes/no entailment
                if overlap < 0.5 and pred_ans:
                    verify_prompt = (
                        f"Question: {question} Is '{expected_ans}' a valid answer for this image? Answer yes or no:"
                    )
                    v_inputs = processor(images=image, text=verify_prompt, return_tensors="pt").to(DEVICE)
                    with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE, enabled=(DEVICE == "cuda")):
                        v_ids = model.generate(**v_inputs, max_new_tokens=4)
                    v_ans = processor.tokenizer.decode(v_ids[0], skip_special_tokens=True).strip().lower()
                    if v_ans.startswith("yes"):
                        overlap = 1.0

                match_scores.append(overlap)
                rows.append({
                    "Generated Question": question,
                    "Expected Span": expected_ans,
                    "BLIP-2 Answer": pred_ans if pred_ans else "(empty)",
                    "Match": "Aligned" if overlap >= 0.5 else "Mismatch",
                })

    vq2_score = float(sum(match_scores) / max(len(match_scores), 1))
    return vq2_score, pd.DataFrame(rows)


# ==============================================================================
# 4. GRADIO CALLBACKS
# ==============================================================================
def run_full_evaluation(image, caption, adapter_label):
    if image is None:
        raise gr.Error("Please upload or select an image first.")
    if not caption or not caption.strip():
        raise gr.Error("Please enter a text description to evaluate.")

    t0 = time.perf_counter()
    rgb_img = image.convert("RGB")

    vnli_prob = score_vnli(rgb_img, caption, adapter_label)
    vq2_prob, qa_df = score_vq2(rgb_img, caption)
    ensemble_prob = 0.5 * vq2_prob + 0.5 * vnli_prob
    elapsed = time.perf_counter() - t0

    aligned = ensemble_prob >= 0.50
    badge_color = "#10b981" if aligned else "#ef4444"
    badge_bg = "rgba(16, 185, 129, 0.14)" if aligned else "rgba(239, 68, 68, 0.14)"
    verdict_title = "ALIGNED (Image Entails Text)" if aligned else "MISALIGNED (Semantic Contradiction)"

    verdict_html = f"""
    <div style="background:{badge_bg}; border:1px solid {badge_color}; border-radius:12px; padding:16px 20px; margin-bottom:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <span style="font-size:1.15rem; font-weight:700; color:{badge_color};">{verdict_title}</span>
            <span style="font-size:0.85rem; color:#9ca3af; background:#111827; padding:4px 10px; border-radius:999px; border:1px solid #374151;">
                Latency: {elapsed:.2f}s · Ensemble Threshold: 0.50
            </span>
        </div>
        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-top:14px;">
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px;">
                <div style="font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.04em;">SeeTRUE Ensemble (0.5·VQ² + 0.5·VNLI)</div>
                <div style="font-size:1.55rem; font-weight:700; color:#38bdf8; margin-top:4px;">{ensemble_prob * 100:.1f}%</div>
            </div>
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px;">
                <div style="font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.04em;">QLoRA-VNLI Score</div>
                <div style="font-size:1.55rem; font-weight:700; color:#f8fafc; margin-top:4px;">{vnli_prob * 100:.1f}%</div>
            </div>
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px;">
                <div style="font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.04em;">Zero-Shot VQ² Score</div>
                <div style="font-size:1.55rem; font-weight:700; color:#cbd5e1; margin-top:4px;">{vq2_prob * 100:.1f}%</div>
            </div>
        </div>
    </div>
    """
    return verdict_html, qa_df


def run_fast_vnli(image, caption, adapter_label):
    if image is None:
        raise gr.Error("Please upload or select an image first.")
    if not caption or not caption.strip():
        raise gr.Error("Please enter a caption.")

    t0 = time.perf_counter()
    prob = score_vnli(image.convert("RGB"), caption, adapter_label)
    elapsed = time.perf_counter() - t0

    aligned = prob >= 0.50
    color = "#10b981" if aligned else "#ef4444"
    label = "ENTAILMENT (YES)" if aligned else "CONTRADICTION / MISMATCH (NO)"

    return f"""
    <div style="background:#0f172a; border:1px solid {color}; border-radius:12px; padding:18px;">
        <div style="font-size:0.82rem; color:#94a3b8; text-transform:uppercase;">Single-Pass VNLI Inference ({elapsed * 1000:.0f} ms)</div>
        <div style="font-size:1.3rem; font-weight:700; color:{color}; margin:6px 0;">{label}</div>
        <div style="font-size:2rem; font-weight:800; color:#f8fafc;">{prob * 100:.2f}% <span style="font-size:0.9rem; font-weight:400; color:#94a3b8;">Entailment Probability</span></div>
    </div>
    """


# ==============================================================================
# 5. FORCED DARK THEME & CLEAN LAYOUT
# ==============================================================================
dark_theme = gr.themes.Base(
    primary_hue="sky",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#090d16",
    body_background_fill_dark="#090d16",
    body_text_color="#f1f5f9",
    body_text_color_dark="#f1f5f9",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_color="#1f2937",
    block_border_color_dark="#1f2937",
    block_label_text_color="#cbd5e1",
    block_label_text_color_dark="#cbd5e1",
    block_title_text_color="#f8fafc",
    block_title_text_color_dark="#f8fafc",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    button_primary_background_fill="#0284c7",
    button_primary_background_fill_dark="#0284c7",
    button_primary_background_fill_hover="#0369a1",
    button_primary_background_fill_hover_dark="#0369a1",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    table_even_background_fill="#0f172a",
    table_even_background_fill_dark="#0f172a",
    table_odd_background_fill="#111827",
    table_odd_background_fill_dark="#111827",
    table_border_color="#1f2937",
    table_border_color_dark="#1f2937",
)

CUSTOM_CSS = """
html, body, .gradio-container {
    background-color: #090d16 !important;
    color: #f1f5f9 !important;
}
.header-card {
    background: linear-gradient(135deg, #0f172a 0%, #111827 60%, #0c4a6e 100%);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.pill-badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.4);
    color: #38bdf8;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    margin-right: 6px;
}
footer { display: none !important; }
"""

# Force dark mode class in browser regardless of OS theme
FORCE_DARK_JS = """
() => {
    document.body.classList.add('dark');
    document.documentElement.classList.add('dark');
}
"""

available_choices = list(loaded_adapters.keys())
default_choice = DEFAULT_ADAPTER_LABEL if DEFAULT_ADAPTER_LABEL in loaded_adapters else available_choices[0]

with gr.Blocks(title="SeeTRUE & QLoRA-VNLI Text-Image Alignment") as demo:
    gr.HTML(
        """
        <div class="header-card">
            <div style="margin-bottom:8px;">
                <span class="pill-badge">Yarom et al. (NeurIPS 2023) Reproduction</span>
                <span class="pill-badge">4-Bit NF4 BLIP-2 Flan-T5-XL</span>
                <span class="pill-badge">SNLI-VE + COCO-Con + PickaPic-Con</span>
            </div>
            <h1 style="margin:0; font-size:1.55rem; font-weight:700; color:#f8fafc;">
                SeeTRUE & QLoRA-VNLI Text-to-Image Alignment Evaluator
            </h1>
            <p style="margin:6px 0 0 0; color:#94a3b8; font-size:0.92rem;">
                Fine-grained semantic alignment scoring combining 4-bit QLoRA Visual Natural Language Inference (VNLI) and Zero-Shot VQ² Question-Answer decomposition.
            </p>
        </div>
        """
    )

    with gr.Tabs():
        # ----------------------------------------------------------------------
        # TAB 1: FULL SEETRUE EVALUATION (VNLI + VQ2 + ENSEMBLE)
        # ----------------------------------------------------------------------
        with gr.Tab("Unified SeeTRUE Alignment (VNLI + VQ²)"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    in_image = gr.Image(type="pil", label="Input Image", height=330)
                    in_caption = gr.Textbox(
                        label="Text Prompt / Description",
                        placeholder="e.g., A red apple sitting on top of a wooden table",
                        lines=2,
                    )
                    in_adapter = gr.Dropdown(
                        choices=available_choices,
                        value=default_choice,
                        label="Active VNLI Checkpoint",
                    )
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary")
                        eval_btn = gr.Button("Evaluate Alignment", variant="primary")

                with gr.Column(scale=7):
                    out_verdict = gr.HTML(
                        value="""
                        <div style="background:#0f172a; border:1px dashed #334155; border-radius:12px; padding:24px; text-align:center; color:#94a3b8;">
                            Upload an image, enter a caption, and click <b>Evaluate Alignment</b> to inspect QLoRA-VNLI, Zero-Shot VQ², and SeeTRUE Ensemble scores.
                        </div>
                        """
                    )
                    out_qa_table = gr.Dataframe(
                        headers=["Generated Question", "Expected Span", "BLIP-2 Answer", "Match"],
                        label="VQ² Atomic Question-Answering Verification Breakdown",
                        interactive=False,
                        wrap=True,
                    )

            eval_btn.click(
                fn=run_full_evaluation,
                inputs=[in_image, in_caption, in_adapter],
                outputs=[out_verdict, out_qa_table],
            )
            clear_btn.click(
                fn=lambda: (None, "", "", pd.DataFrame()),
                inputs=[],
                outputs=[in_image, in_caption, out_verdict, out_qa_table],
            )

        # ----------------------------------------------------------------------
        # TAB 2: FAST SINGLE-PASS VNLI CHECK
        # ----------------------------------------------------------------------
        with gr.Tab("Fast VNLI Single-Pass Check"):
            with gr.Row():
                with gr.Column(scale=5):
                    fast_image = gr.Image(type="pil", label="Input Image", height=320)
                    fast_caption = gr.Textbox(
                        label="Candidate Caption",
                        placeholder="Enter a caption to test subtle object, attribute, count, or spatial edits...",
                        lines=2,
                    )
                    fast_adapter = gr.Dropdown(
                        choices=available_choices,
                        value=default_choice,
                        label="Active VNLI Checkpoint",
                    )
                    fast_btn = gr.Button("Run VNLI Check", variant="primary")

                with gr.Column(scale=7):
                    fast_output = gr.HTML(
                        value="""
                        <div style="background:#0f172a; border:1px dashed #334155; border-radius:12px; padding:24px; text-align:center; color:#94a3b8;">
                            Runs single-pass QLoRA-VNLI entailment scoring (~150 ms) without VQ² question generation.
                        </div>
                        """
                    )

            fast_btn.click(
                fn=run_fast_vnli,
                inputs=[fast_image, fast_caption, fast_adapter],
                outputs=[fast_output],
            )

        # ----------------------------------------------------------------------
        # TAB 3: BENCHMARK & ARCHITECTURE SUMMARY
        # ----------------------------------------------------------------------
        with gr.Tab("Model & Benchmark Summary"):
            gr.Markdown(
                """
                ### Frozen Model Architecture
                * **Backbone:** `Salesforce/blip2-flan-t5-xl` (`3.95B` parameters, 4-bit NF4 quantization via `bitsandbytes`, `~4.2 GB` VRAM)
                * **QLoRA Adapter (`vnli_adapter_b8_congen`):** Rank `r = 16`, `lora_alpha = 32`, `10.38M` trainable parameters (`0.2627%`)
                * **3-Source Training Mixture (`13,026` pairs):** `10,000` `SNLI-VE` pairs + `1,658` `COCO-Con` pairs + `1,368` `PickaPic-Con` NLI-verified minimal contradictions
                * **SeeTRUE Ensemble Formula:** $S_{\\text{SeeTRUE}}(I, T) = 0.5 \\cdot S_{\\text{VQ}^2}(I, T) + 0.5 \\cdot P_{\\text{VNLI}}(\\text{yes} \\mid I, T)$

                ---

                ### Quantitative Reproduction & ConGen Ablation Results (ROC-AUC %)

                | Benchmark / Metric | Zero-Shot $VQ^2$ | VNLI (`10k` SNLI-VE) | VNLI (`10k` SNLI-VE + `3k` ConGen) |
                | :--- | :---: | :---: | :---: |
                | **SeeTRUE: DrawBench** (`1,312`) | `74.58%` | `82.56%` | **`83.28%`** |
                | **SeeTRUE: COCO-T2I** (`1,791`) | `65.93%` | **`82.91%`** | `81.47%` |
                | **SeeTRUE: EditBench** (`3,827`) | `65.83%` | **`73.90%`** | `70.92%` |
                | **SeeTRUE Macro Average** | `68.78%` | **`79.79%`** | `78.56%` |
                | **Winoground Pairwise ROC-AUC** | `56.38%` | **`62.15%`** | `61.74%` |
                | **Winoground Group Score** | `11.50%` | `25.25%` | **`25.50%`** |
                """
            )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=dark_theme,
        css=CUSTOM_CSS,
        js=FORCE_DARK_JS,
    )