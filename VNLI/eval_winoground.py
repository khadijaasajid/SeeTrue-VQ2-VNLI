import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel


def resolve_image_path(row, seetrue_img_dir, alt_img_dir):
    img_id = str(row["image"]).strip()
    p1 = os.path.join(seetrue_img_dir, f"{img_id}.png")
    if os.path.exists(p1):
        return p1
    p1_jpg = os.path.join(seetrue_img_dir, f"{img_id}.jpg")
    if os.path.exists(p1_jpg):
        return p1_jpg
    alt_name = img_id.replace("_img_", "_image_")
    p2 = os.path.join(alt_img_dir, f"{alt_name}.jpg")
    if os.path.exists(p2):
        return p2
    p2_png = os.path.join(alt_img_dir, f"{alt_name}.png")
    if os.path.exists(p2_png):
        return p2_png
    raise FileNotFoundError(f"Could not find image for {img_id} in {seetrue_img_dir} or {alt_img_dir}")


def compute_winoground_quartets(df, score_col):
    text_hits, image_hits, group_hits, total = 0, 0, 0, 0
    for qid, g in df.groupby("id"):
        if len(g) != 4:
            continue
        scores = dict(zip(g["img_caption"], g[score_col]))
        required_keys = [
            "image_0_caption_0",
            "image_0_caption_1",
            "image_1_caption_0",
            "image_1_caption_1",
        ]
        if not all(k in scores for k in required_keys):
            continue
        s00 = scores["image_0_caption_0"]
        s01 = scores["image_0_caption_1"]
        s10 = scores["image_1_caption_0"]
        s11 = scores["image_1_caption_1"]

        t_ok = (s00 > s01) and (s11 > s10)
        i_ok = (s00 > s10) and (s11 > s01)
        g_ok = t_ok and i_ok

        text_hits += int(t_ok)
        image_hits += int(i_ok)
        group_hits += int(g_ok)
        total += 1

    return {
        "quartets": total,
        "text_score": 100.0 * text_hits / max(total, 1),
        "image_score": 100.0 * image_hits / max(total, 1),
        "group_score": 100.0 * group_hits / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default="vnli_adapter_b7_latest")
    parser.add_argument("--cartesian_csv", type=str, default="../SeeTrue/outputs/winoground_cartesian.csv")
    parser.add_argument("--vq2_csv", type=str, default="../SeeTrue/outputs/winoground_vq2_scores.csv")
    parser.add_argument("--seetrue_img_dir", type=str, default="../SeeTrue/images")
    parser.add_argument("--alt_img_dir", type=str, default="../images")
    parser.add_argument("--output_csv", type=str, default="eval_results_b7_10000/winoground_vnli_preds.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Adapter: {args.adapter_dir}")

    model_id = "Salesforce/blip2-flan-t5-xl"
    processor = Blip2Processor.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    tokenizer = processor.tokenizer
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    decoder_start_id = model.config.text_config.decoder_start_token_id
    if decoder_start_id is None:
        decoder_start_id = tokenizer.pad_token_id

    df = pd.read_csv(args.cartesian_csv)
    print(f"Loaded {len(df)} rows ({df['id'].nunique()} quartets) from {args.cartesian_csv}")

    image_cache = {}
    for _, row in df.drop_duplicates(subset=["image"]).iterrows():
        path = resolve_image_path(row, args.seetrue_img_dir, args.alt_img_dir)
        image_cache[row["image"]] = Image.open(path).convert("RGB")

    vnli_probs = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring Winoground"):
            img = image_cache[row["image"]]
            prompt = f"Question: Does this image entail the description: '{row['text']}'? Answer yes or no. Answer:"
            inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            decoder_input_ids = torch.tensor([[decoder_start_id]], device=device)

            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
            next_token_logits = outputs.logits[0, 0, :]
            pair_logits = torch.stack([next_token_logits[no_id], next_token_logits[yes_id]])
            probs = torch.softmax(pair_logits.float(), dim=-1)
            vnli_probs.append(float(probs[1].item()))

    df["vnli_score"] = vnli_probs

    if os.path.exists(args.vq2_csv):
        df_vq2 = pd.read_csv(args.vq2_csv)[["id", "img_caption", "vq2_score"]]
        df = df.merge(df_vq2, on=["id", "img_caption"], how="left")

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved full Winoground predictions -> {args.output_csv}")

    full_auc = 100.0 * roc_auc_score(df["label"], df["vnli_score"])
    full_q = compute_winoground_quartets(df, "vnli_score")
    print("\n=== Full Winoground Results (400 Quartets / 1,600 Pairs) ===")
    print(f"VNLI Pairwise ROC-AUC : {full_auc:.2f}%")
    print(f"VNLI Text Score       : {full_q['text_score']:.2f}%  (Random: 25.00%)")
    print(f"VNLI Image Score      : {full_q['image_score']:.2f}%  (Random: 25.00%)")
    print(f"VNLI Group Score      : {full_q['group_score']:.2f}%  (Random: 16.67%)")

    if "vq2_score" in df.columns and df["vq2_score"].notna().any():
        sub = df[df["vq2_score"].notna()].copy()
        sub["ensemble_score"] = 0.5 * sub["vq2_score"] + 0.5 * sub["vnli_score"]
        vq2_auc = 100.0 * roc_auc_score(sub["label"], sub["vq2_score"])
        vnli_sub_auc = 100.0 * roc_auc_score(sub["label"], sub["vnli_score"])
        ens_auc = 100.0 * roc_auc_score(sub["label"], sub["ensemble_score"])

        q_vq2 = compute_winoground_quartets(sub, "vq2_score")
        q_vnli = compute_winoground_quartets(sub, "vnli_score")
        q_ens = compute_winoground_quartets(sub, "ensemble_score")

        print(f"\n=== VQ2-Matched Comparison ({q_vnli['quartets']} Quartets / {len(sub)} Pairs) ===")
        print(f"Pairwise ROC-AUC -> VQ2: {vq2_auc:.2f}% | VNLI: {vnli_sub_auc:.2f}% | SeeTRUE Ensemble: {ens_auc:.2f}%")
        print(f"Text Score       -> VQ2: {q_vq2['text_score']:.2f}% | VNLI: {q_vnli['text_score']:.2f}% | SeeTRUE Ensemble: {q_ens['text_score']:.2f}%")
        print(f"Image Score      -> VQ2: {q_vq2['image_score']:.2f}% | VNLI: {q_vnli['image_score']:.2f}% | SeeTRUE Ensemble: {q_ens['image_score']:.2f}%")
        print(f"Group Score      -> VQ2: {q_vq2['group_score']:.2f}% | VNLI: {q_vnli['group_score']:.2f}% | SeeTRUE Ensemble: {q_ens['group_score']:.2f}%")


if __name__ == "__main__":
    main()
