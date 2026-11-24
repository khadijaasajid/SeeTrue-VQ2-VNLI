import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


def find_image_path(image_name: str, search_roots: list[str]) -> str | None:
    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name
    for root in search_roots:
        candidate = os.path.join(root, image_name)
        if os.path.exists(candidate):
            return candidate
    # Fallback recursive search once per missing root
    for root in search_roots:
        if os.path.exists(root):
            matches = glob.glob(os.path.join(root, "**", os.path.basename(image_name)), recursive=True)
            if matches:
                return matches[0]
    return None


def score_dataset(df: pd.DataFrame, split_name: str, model, processor, device, search_roots: list[str]) -> pd.DataFrame:
    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

    vnli_scores = []
    valid_indices = []
    missing_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {split_name}"):
        img_path = find_image_path(str(row["image"]), search_roots)
        if img_path is None:
            missing_count += 1
            continue

        image = Image.open(img_path).convert("RGB")
        text = str(row["text"]).strip()
        prompt = f'Does this image entail the description: "{text}"?'

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        # Provide decoder start token to read next-token logits for 'yes' vs 'no'
        decoder_input_ids = torch.tensor([[model.config.text_config.decoder_start_token_id]], device=device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                )
                logits = outputs.logits[:, 0, :]  # [1, vocab_size]
                pair_logits = torch.stack([logits[0, yes_id], logits[0, no_id]])
                probs = torch.softmax(pair_logits.float(), dim=0)
                p_yes = probs[0].item()

        vnli_scores.append(p_yes)
        valid_indices.append(idx)

    if missing_count > 0:
        print(f"Warning: {missing_count} images not found for {split_name}.")

    out_df = df.loc[valid_indices].copy()
    out_df["vnli_score"] = vnli_scores
    if "vq2_score" in out_df.columns:
        out_df["seetrue_ensemble_score"] = 0.5 * out_df["vq2_score"] + 0.5 * out_df["vnli_score"]
    return out_df


def plot_roc_comparison(df: pd.DataFrame, split_name: str, out_png: str):
    y_true = df["label"].astype(int).values
    plt.figure(figsize=(7, 6))

    vnli_auc = roc_auc_score(y_true, df["vnli_score"].values)
    fpr_v, tpr_v, _ = roc_curve(y_true, df["vnli_score"].values)
    plt.plot(fpr_v, tpr_v, label=f"VNLI (QLoRA) ROC-AUC = {vnli_auc:.4f}", linewidth=2)

    if "vq2_score" in df.columns:
        vq2_auc = roc_auc_score(y_true, df["vq2_score"].values)
        fpr_q, tpr_q, _ = roc_curve(y_true, df["vq2_score"].values)
        plt.plot(fpr_q, tpr_q, label=f"VQ² ROC-AUC = {vq2_auc:.4f}", linestyle="--", linewidth=2)

        ens_auc = roc_auc_score(y_true, df["seetrue_ensemble_score"].values)
        fpr_e, tpr_e, _ = roc_curve(y_true, df["seetrue_ensemble_score"].values)
        plt.plot(fpr_e, tpr_e, label=f"SeeTRUE (VQ² + VNLI) ROC-AUC = {ens_auc:.4f}", linewidth=2.5)

    plt.plot([0, 1], [0, 1], "k:", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"SeeTRUE Benchmark: {split_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default="vnli_adapter_b1")
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--seetrue_dir", type=str, default="../SeeTrue")
    parser.add_argument("--output_dir", type=str, default="eval_results_b1")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["drawbench", "coco_t2i", "editbench"],
        help="Which SeeTRUE splits to evaluate",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading 4-bit base model + LoRA adapter from '{args.adapter_dir}'...")
    processor = Blip2Processor.from_pretrained(args.model_id)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    summary_rows = []
    for split in args.splits:
        csv_path = os.path.join(args.seetrue_dir, "outputs", f"{split}_vq2_scores.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {split}: {csv_path} not found.")
            continue

        df = pd.read_csv(csv_path)
        search_roots = [
            os.path.join(args.seetrue_dir, "images"),
            os.path.join(args.seetrue_dir, "data", "images"),
            os.path.join(args.seetrue_dir, split),
            os.path.join(args.seetrue_dir, "images", split),
            args.seetrue_dir,
        ]

        scored_df = score_dataset(df, split, model, processor, device, search_roots)
        if len(scored_df) == 0:
            print(f"No images matched for {split}. Check image folder path inside {args.seetrue_dir}.")
            continue

        y_true = scored_df["label"].astype(int).values
        vnli_auc = roc_auc_score(y_true, scored_df["vnli_score"].values) * 100
        vq2_auc = roc_auc_score(y_true, scored_df["vq2_score"].values) * 100 if "vq2_score" in scored_df.columns else None
        ens_auc = (
            roc_auc_score(y_true, scored_df["seetrue_ensemble_score"].values) * 100
            if "seetrue_ensemble_score" in scored_df.columns
            else None
        )

        out_csv = os.path.join(args.output_dir, f"{split}_vnli_results.csv")
        out_png = os.path.join(args.output_dir, f"{split}_roc_comparison.png")
        scored_df.to_csv(out_csv, index=False)
        plot_roc_comparison(scored_df, split, out_png)

        summary_rows.append({
            "Split": split,
            "Samples": len(scored_df),
            "VQ2_AUC": round(vq2_auc, 2) if vq2_auc is not None else "-",
            "VNLI_AUC": round(vnli_auc, 2),
            "SeeTRUE_Ensemble_AUC": round(ens_auc, 2) if ens_auc is not None else "-",
        })

    if summary_rows:
        print("\n=== SeeTRUE Evaluation Summary (ROC-AUC %) ===")
        print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()