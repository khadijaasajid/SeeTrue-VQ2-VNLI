import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

# --- CONFIG: Choose "drawbench", "coco_t2i", or "editbench" ---
SPLIT_NAME = "drawbench"
SMOKE_TEST_N = 15   # Run 15 rows first to check RTX 4050 VRAM & speed, then set to None!

PAPER_TARGETS = {
    "drawbench": 74.80,
    "coco_t2i": 73.10,
    "editbench": 65.40
}

# 1. Local Paths (relative to SeeTrue folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "SeeTRUE.csv")
IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

QA_CACHE_PATH = os.path.join(OUT_DIR, f"{SPLIT_NAME}_qa_cache.json")
SCORES_CSV_PATH = os.path.join(OUT_DIR, f"{SPLIT_NAME}_vq2_scores.csv")
ROC_PLOT_PATH = os.path.join(OUT_DIR, f"{SPLIT_NAME}_roc_curve.png")

assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
assert os.path.exists(IMG_DIR), f"Missing {IMG_DIR}"
assert os.path.exists(QA_CACHE_PATH), f"Missing {QA_CACHE_PATH}"

# 2. Load Split & Cached QAs
df = pd.read_csv(CSV_PATH)
split_df = df[df["dataset_source"] == SPLIT_NAME].copy().reset_index(drop=True)
if SMOKE_TEST_N is not None:
    split_df = split_df.head(SMOKE_TEST_N).copy()

with open(QA_CACHE_PATH, "r", encoding="utf-8") as f:
    qa_cache = json.load(f)

print(f"[{SPLIT_NAME}] Loaded {len(split_df)} rows and {len(qa_cache)} cached QA sets.")

# 3. Load BLIP-2 Flan-T5-XL in 8-bit (~4.3 GB VRAM — fits cleanly on RTX 4050 6GB!)
assert torch.cuda.is_available(), "CUDA GPU not detected! Activate your CUDA PyTorch environment."
model_id = "Salesforce/blip2-flan-t5-xl"
print(f"Loading {model_id} in 8-bit on {torch.cuda.get_device_name(0)}...")

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
print(f"Model loaded! Current VRAM allocated: {allocated_gb:.2f} GB")

yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

def score_image_caption_pair(image_path, caption):
    qas = qa_cache.get(caption, [])
    if not qas:
        return 0.5, []

    raw_image = Image.open(image_path).convert("RGB")
    probs = []
    details = []

    for qa in qas:
        q = qa["question"]
        a = qa["answer"]
        prompt = f"Question: {q} Is '{a}' the correct answer based on the image? Answer yes or no:"

        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True
            )

        first_token_logits = outputs.scores[0][0]
        yn_logits = torch.tensor([first_token_logits[yes_id], first_token_logits[no_id]], dtype=torch.float32)
        p_yes = float(torch.softmax(yn_logits, dim=0)[0])

        probs.append(p_yes)
        details.append({"question": q, "answer": a, "p_yes": round(p_yes, 4)})

    return sum(probs) / len(probs), details

# 4. Resume from Checkpoint if Interrupted
completed = {}
if SMOKE_TEST_N is None and os.path.exists(SCORES_CSV_PATH):
    prev_df = pd.read_csv(SCORES_CSV_PATH)
    for _, r in prev_df.iterrows():
        completed[int(r["row_idx"])] = r.to_dict()
    print(f"Resuming from checkpoint: {len(completed)}/{len(split_df)} rows already scored.")

results = list(completed.values())

for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=SPLIT_NAME):
    if idx in completed:
        continue

    img_path = os.path.join(IMG_DIR, row["image"])
    score, qa_details = score_image_caption_pair(img_path, row["text"])

    results.append({
        "row_idx": idx,
        "image": row["image"],
        "text": row["text"],
        "label": int(row["label"]),
        "original_dataset_id": row["original_dataset_id"],
        "vq2_score": score,
        "qa_breakdown": json.dumps(qa_details)
    })

    # Checkpoint every 15 rows
    if SMOKE_TEST_N is None and len(results) % 15 == 0:
        pd.DataFrame(results).sort_values("row_idx").to_csv(SCORES_CSV_PATH, index=False)

res_df = pd.DataFrame(results).sort_values("row_idx").reset_index(drop=True)

if SMOKE_TEST_N is not None:
    print(f"\nSmoke test complete ({SMOKE_TEST_N} rows)! Sample scores:")
    print(res_df[["image", "label", "vq2_score"]].head(10))
else:
    res_df.to_csv(SCORES_CSV_PATH, index=False)
    auc = roc_auc_score(res_df["label"], res_df["vq2_score"])
    target = PAPER_TARGETS.get(SPLIT_NAME, 0.0)
    print(f"\n=======================================================")
    print(f"SeeTRUE {SPLIT_NAME.upper()} (n={len(res_df)}) ROC-AUC: {auc * 100:.2f}%  (Paper BLIP-T5 target: {target:.2f}%)")
    print(f"=======================================================")

    fpr, tpr, _ = roc_curve(res_df["label"], res_df["vq2_score"])
    plt.figure(figsize=(6, 5), dpi=150)
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"Zero-Shot VQ² (ROC-AUC = {auc * 100:.2f}%)")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Chance (50.00%)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"SeeTRUE — {SPLIT_NAME.upper()} Split ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH)
    print(f"Saved ROC curve plot to: {ROC_PLOT_PATH}")