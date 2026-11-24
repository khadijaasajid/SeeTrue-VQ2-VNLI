import os
import json
import time
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

TEST_N = 15  # Quick local batch test on 15 DrawBench rows

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "SeeTRUE.csv")
IMG_DIR = os.path.join(BASE_DIR, "images")
QA_CACHE_PATH = os.path.join(BASE_DIR, "outputs", "drawbench_qa_cache.json")

# 1. Pre-flight check
assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
assert os.path.exists(IMG_DIR), f"Missing {IMG_DIR}"
assert os.path.exists(QA_CACHE_PATH), f"Missing {QA_CACHE_PATH}"
assert torch.cuda.is_available(), "CUDA GPU not detected! Check your venv."

df = pd.read_csv(CSV_PATH)
test_df = df[df["dataset_source"] == "drawbench"].head(TEST_N).copy().reset_index(drop=True)
with open(QA_CACHE_PATH, "r", encoding="utf-8") as f:
    qa_cache = json.load(f)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Loaded {len(test_df)} test rows | Cached unique prompts: {len(qa_cache)}")

# 2. Load BLIP-2 Flan-T5-XL in 8-bit (~4.3 GB VRAM)
model_id = "Salesforce/blip2-flan-t5-xl"
# bnb_config = BitsAndBytesConfig(load_in_8bit=True)

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

# processor = Blip2Processor.from_pretrained(model_id)
# model = Blip2ForConditionalGeneration.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map="auto"
# )
model.eval()

yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

def score_image_batched(image_path, caption):
    """Scores all QA pairs for a single image in one batched forward pass."""
    qas = qa_cache.get(caption, [])
    if not qas:
        return 0.5, []

    raw_image = Image.open(image_path).convert("RGB")
    prompts = [
        f"Question: {qa['question']} Is '{qa['answer']}' the correct answer based on the image? Answer yes or no:"
        for qa in qas
    ]

    # Batch all questions for this image together with padding=True
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

    # outputs.scores[0] has shape [num_questions, vocab_size]
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

# 3. Run Batch Test & Measure Speed + Peak VRAM
torch.cuda.reset_peak_memory_stats()
start_time = time.time()
records = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Batch Smoke Test"):
    img_path = os.path.join(IMG_DIR, row["image"])
    score, breakdown = score_image_batched(img_path, row["text"])
    records.append({
        "image": row["image"],
        "label": int(row["label"]),
        "vq2_score": round(score, 4),
        "num_qas": len(breakdown),
        "breakdown": breakdown
    })

elapsed = time.time() - start_time
peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

print("\n=== BATCH TEST RESULTS ===")
print(f"Elapsed: {elapsed:.2f}s ({len(test_df)/elapsed:.2f} rows/sec) | Peak VRAM: {peak_vram_gb:.2f} GB / 6.00 GB")
print(pd.DataFrame(records)[["image", "label", "num_qas", "vq2_score"]].to_string(index=False))

print("\nSample Per-Question Breakdown (Row 0):")
print(f"Caption: {test_df.loc[0, 'text']} (Ground Truth Label: {test_df.loc[0, 'label']})")
for item in records[0]["breakdown"]:
    print(f"  - Q: {item['q']} | A: {item['a']}  -->  p_yes = {item['p_yes']}")