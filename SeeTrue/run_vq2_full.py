import os
import json
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

# Run all 3 splits in one session (or edit list to run specific splits)
SPLITS_TO_RUN = ["drawbench", "coco_t2i", "editbench"]
CHECKPOINT_EVERY = 25

PAPER_TARGETS = {
    "drawbench": 74.80,
    "coco_t2i": 73.10,
    "editbench": 65.40
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "SeeTRUE.csv")
IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
assert os.path.exists(IMG_DIR), f"Missing {IMG_DIR}"
assert torch.cuda.is_available(), "CUDA GPU not detected! Activate your venv first."

df = pd.read_csv(CSV_PATH)

# 1. Load BLIP-2 Flan-T5-XL once in 4-bit NF4 (~4.8 GB peak VRAM with batching)
model_id = "Salesforce/blip2-flan-t5-xl"
print(f"Loading {model_id} onto {torch.cuda.get_device_name(0)}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.eval()

yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

def score_image_batched(image_path, caption, qa_cache):
    qas = qa_cache.get(caption, [])
    if not qas:
        return 0.5, []

    raw_image = Image.open(image_path).convert("RGB")
    prompts = [
        f"Question: {qa['question']} Is '{qa['answer']}' the correct answer based on the image? Answer yes or no:"
        for qa in qas
    ]

    inputs = processor(
        images=[raw_image] * len(prompts),
        text=prompts,
        padding=True,
        return_tensors="pt"
    ).to("cuda", torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True
        )

    first_token_logits = outputs.scores[0]
    probs = []
    details = []

    for i, qa in enumerate(qas):
        yn_logits = torch.tensor(
            [first_token_logits[i, yes_id], first_token_logits[i, no_id]],
            dtype=torch.float32
        )
        p_yes = float(torch.softmax(yn_logits, dim=0)[0])
        probs.append(p_yes)
        details.append({"q": qa["question"], "a": qa["answer"], "p_yes": round(p_yes, 4)})

    return sum(probs) / len(probs), details

# 2. Evaluate Each Split
summary_rows = []

for split_name in SPLITS_TO_RUN:
    qa_cache_path = os.path.join(OUT_DIR, f"{split_name}_qa_cache.json")
    scores_csv_path = os.path.join(OUT_DIR, f"{split_name}_vq2_scores.csv")
    roc_plot_path = os.path.join(OUT_DIR, f"{split_name}_roc_curve.png")

    assert os.path.exists(qa_cache_path), f"Missing {qa_cache_path}"
    with open(qa_cache_path, "r", encoding="utf-8") as f:
        qa_cache = json.load(f)

    split_df = df[df["dataset_source"] == split_name].copy().reset_index(drop=True)

    # Resume from checkpoint if interrupted
    completed = {}
    if os.path.exists(scores_csv_path):
        prev_df = pd.read_csv(scores_csv_path)
        for _, r in prev_df.iterrows():
            completed[int(r["row_idx"])] = r.to_dict()
        print(f"\n[{split_name.upper()}] Resuming from {len(completed)}/{len(split_df)} saved rows...")
    else:
        print(f"\n[{split_name.upper()}] Starting evaluation on {len(split_df)} rows...")

    results = list(completed.values())
    t0 = time.time()

    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
        if idx in completed:
            continue

        img_path = os.path.join(IMG_DIR, row["image"])
        score, qa_details = score_image_batched(img_path, row["text"], qa_cache)

        results.append({
            "row_idx": idx,
            "image": row["image"],
            "text": row["text"],
            "label": int(row["label"]),
            "original_dataset_id": row["original_dataset_id"],
            "vq2_score": round(score, 6),
            "qa_breakdown": json.dumps(qa_details)
        })

        if len(results) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).sort_values("row_idx").to_csv(scores_csv_path, index=False)

    res_df = pd.DataFrame(results).sort_values("row_idx").reset_index(drop=True)
    res_df.to_csv(scores_csv_path, index=False)

    auc_pct = roc_auc_score(res_df["label"], res_df["vq2_score"]) * 100
    target_pct = PAPER_TARGETS[split_name]
    diff_pct = auc_pct - target_pct
    elapsed_min = (time.time() - t0) / 60.0

    summary_rows.append({
        "Split": split_name,
        "Rows": len(res_df),
        "Our ROC-AUC (%)": round(auc_pct, 2),
        "Paper Target (%)": target_pct,
        "Diff (%)": f"{diff_pct:+.2f}",
        "Time (min)": round(elapsed_min, 1)
    })

    # Save ROC Curve Plot
    fpr, tpr, _ = roc_curve(res_df["label"], res_df["vq2_score"])
    plt.figure(figsize=(6, 5), dpi=150)
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"Zero-Shot VQ² (AUC = {auc_pct:.2f}%)")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random (50.00%)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"SeeTRUE — {split_name.upper()} ROC Curve (n={len(res_df)})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_plot_path)
    plt.close()

print("\n=====================================================================")
print("FINAL SEETRUE ZERO-SHOT VQ² BENCHMARK SUMMARY (RTX 4050)")
print("=====================================================================")
print(pd.DataFrame(summary_rows).to_string(index=False))