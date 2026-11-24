import io
import os
import gc
import base64
import random
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)


def _decode_pickapic_image(val):
    if val is None:
        return None
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, bytes):
        return Image.open(io.BytesIO(val)).convert("RGB")
    if isinstance(val, str):
        raw = base64.b64decode(val)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


def stream_coco_positives(n_target=1600, seed=42):
    """Streams n_target image-caption pairs from MS-COCO (no full 25GB download)."""
    print(f"Streaming {n_target} positive pairs from COCO...")
    ds = load_dataset("lmms-lab/COCO-Caption2017", split="val", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)

    items = []
    for row in ds:
        img = row.get("image")
        ans = row.get("answer") or row.get("caption")
        if isinstance(ans, list):
            ans = ans[0]
        if img is not None and ans and len(str(ans).split()) >= 4:
            items.append({
                "source": "COCO-Con",
                "image_pil": img.convert("RGB"),
                "caption_pos": str(ans).strip(),
            })
        if len(items) >= n_target:
            break
    print(f"Collected {len(items)} COCO positive pairs.")
    return items


def stream_pickapic_positives(n_target=1600, seed=42):
    """Streams n_target human-preferred synthetic T2I pairs from Pick-a-Pic."""
    print(f"Streaming {n_target} positive synthetic T2I pairs from Pick-a-Pic...")
    ds = load_dataset("pickapic-anonymous/pickapic_v1", split="validation", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)

    items = []
    for row in ds:
        cap = row.get("caption")
        if not cap or len(str(cap).split()) < 4:
            continue
        # Pick the winning synthetic image (label_0 >= 0.5 -> jpg_0, else jpg_1)
        win_raw = row.get("jpg_0") if row.get("label_0", 0) >= 0.5 else row.get("jpg_1")
        try:
            img = _decode_pickapic_image(win_raw)
            if img is None:
                continue
            items.append({
                "source": "PickaPic-Con",
                "image_pil": img,
                "caption_pos": str(cap).strip(),
            })
        except Exception:
            continue
        if len(items) >= n_target:
            break
    print(f"Collected {len(items)} Pick-a-Pic positive pairs.")
    return items


def run_congen_pipeline(raw_items):
    """
    Reproduces Yarom et al. (2023) Section 2.3 ConGen:
      1. Few-shot LLM generation of minimal contradictions (C+ -> C-)
      2. NLI verification filter keeping pairs with P(contradiction) >= 0.75
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Step A: Generating minimal contradictions with Flan-T5-Large on {device}...")
    gen_id = "google/flan-t5-large"
    gen_tok = AutoTokenizer.from_pretrained(gen_id)
    gen_mod = AutoModelForSeq2SeqLM.from_pretrained(
        gen_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    gen_mod.eval()

    few_shot_prefix = (
        "Rewrite the image caption by changing ONLY ONE detail (an object, color, count, action, or spatial position) "
        "so that it directly contradicts the original meaning while staying natural and grammatical.\n\n"
        "Original: a giraffe leaned over in a plush field next to some cows\n"
        "Contradiction: a giraffe standing upright in a desert away from any cows\n\n"
        "Original: two red apples sitting on top of a wooden table\n"
        "Contradiction: two green apples sitting underneath a wooden table\n\n"
        "Original: a young boy riding a bicycle down a paved street\n"
        "Contradiction: an elderly woman walking a dog down a paved street\n\n"
        "Original: a white cat sleeping inside a cardboard box\n"
        "Contradiction: a black dog jumping over a cardboard box\n\n"
    )

    candidates = []
    batch_size = 16 if device == "cuda" else 4
    with torch.no_grad():
        for i in tqdm(range(0, len(raw_items), batch_size), desc="ConGen Generation"):
            batch = raw_items[i : i + batch_size]
            prompts = [f"{few_shot_prefix}Original: {x['caption_pos']}\nContradiction:" for x in batch]
            enc = gen_tok(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            outs = gen_mod.generate(
                **enc,
                max_new_tokens=48,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
            )
            decoded = gen_tok.batch_decode(outs, skip_special_tokens=True)
            for item, c_neg in zip(batch, decoded):
                c_neg = c_neg.strip()
                if c_neg and c_neg.lower() != item["caption_pos"].lower() and len(c_neg.split()) >= 3:
                    candidates.append({**item, "caption_neg": c_neg})

    del gen_mod, gen_tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Step B: Filtering {len(candidates)} candidates with NLI verifier (roberta-large-mnli) on {device}...")
    nli_id = "FacebookAI/roberta-large-mnli"
    nli_tok = AutoTokenizer.from_pretrained(nli_id)
    nli_mod = AutoModelForSequenceClassification.from_pretrained(
        nli_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    nli_mod.eval()

    # roberta-large-mnli labels: 0 = CONTRADICTION, 1 = NEUTRAL, 2 = ENTAILMENT
    verified = []
    with torch.no_grad():
        for i in tqdm(range(0, len(candidates), batch_size), desc="NLI Filtering"):
            batch = candidates[i : i + batch_size]
            enc = nli_tok(
                [x["caption_pos"] for x in batch],
                [x["caption_neg"] for x in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            probs = torch.softmax(nli_mod(**enc).logits.float(), dim=-1)

            for item, p in zip(batch, probs):
                contra_prob = float(p[0].item())
                pos_w = set(item["caption_pos"].lower().split())
                neg_w = set(item["caption_neg"].lower().split())
                jaccard = len(pos_w & neg_w) / max(len(pos_w | neg_w), 1)

                if contra_prob >= 0.75 and 0.25 <= jaccard <= 0.90:
                    verified.append({**item, "nli_contra_prob": round(contra_prob, 4)})

    del nli_mod, nli_tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Retained {len(verified)} NLI-verified ConGen pairs.")
    return verified


def main():
    out_img_dir = os.path.join("vnli_data", "images")
    os.makedirs(out_img_dir, exist_ok=True)

    coco_items = stream_coco_positives(n_target=1600, seed=42)
    pick_items = stream_pickapic_positives(n_target=1600, seed=42)
    verified = run_congen_pipeline(coco_items + pick_items)

    rows = []
    for idx, item in enumerate(verified):
        prefix = "cococon" if item["source"] == "COCO-Con" else "pickcon"
        img_filename = f"{prefix}_{idx:05d}.jpg"
        img_path = os.path.join(out_img_dir, img_filename)
        item["image_pil"].save(img_path, format="JPEG", quality=92)

        # Balanced pair: 1 positive (label=1) + 1 ConGen hard negative (label=0)
        rows.append({"image": img_filename, "text": item["caption_pos"], "label": 1, "source": item["source"]})
        rows.append({"image": img_filename, "text": item["caption_neg"], "label": 0, "source": item["source"]})

    full_df = pd.DataFrame(rows)
    full_df[full_df["source"] == "COCO-Con"].to_csv("vnli_data/coco_con_all.csv", index=False)
    full_df[full_df["source"] == "PickaPic-Con"].to_csv("vnli_data/pickapic_con_all.csv", index=False)

    unique_imgs = full_df["image"].unique().tolist()
    random.Random(42).shuffle(unique_imgs)
    val_imgs = set(unique_imgs[: max(1, int(0.10 * len(unique_imgs)))])

    train_df = full_df[~full_df["image"].isin(val_imgs)].sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df = full_df[full_df["image"].isin(val_imgs)].reset_index(drop=True)

    train_df[["image", "text", "label"]].to_csv("vnli_data/congen_train.csv", index=False)
    val_df[["image", "text", "label"]].to_csv("vnli_data/congen_val.csv", index=False)

    print("\n=== ConGen Dataset Build Complete ===")
    print(full_df.groupby(["source", "label"]).size())
    print(f"Saved Train: vnli_data/congen_train.csv ({len(train_df)} rows)")
    print(f"Saved Val  : vnli_data/congen_val.csv ({len(val_df)} rows)")


if __name__ == "__main__":
    main()