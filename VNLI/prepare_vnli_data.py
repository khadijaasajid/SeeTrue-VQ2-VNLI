# import argparse
# import os
# import random
# import re
# import pandas as pd
# from datasets import load_dataset

# # Simple attribute/noun swapper to generate synthetic hard-negative contradictions
# SWAP_MAP = {
#     "red": "blue", "blue": "green", "green": "yellow", "yellow": "black",
#     "black": "white", "white": "brown", "brown": "gray", "orange": "purple",
#     "man": "woman", "woman": "man", "men": "women", "women": "men",
#     "boy": "girl", "girl": "boy", "dog": "cat", "cat": "dog",
#     "two": "three", "three": "four", "four": "two", "one": "two",
#     "standing": "sitting", "sitting": "running", "running": "sleeping",
#     "indoors": "outdoors", "outdoors": "indoors", "day": "night", "night": "day",
# }

# def make_hard_negative(text: str):
#     words = text.split()
#     indices = [i for i, w in enumerate(words) if re.sub(r"[^\w]", "", w.lower()) in SWAP_MAP]
#     if not indices:
#         return None
#     idx = random.choice(indices)
#     clean = re.sub(r"[^\w]", "", words[idx].lower())
#     replacement = SWAP_MAP[clean]
#     if words[idx][0].isupper():
#         replacement = replacement.capitalize()
#     words[idx] = re.sub(clean, replacement, words[idx], flags=re.IGNORECASE)
#     return " ".join(words)

# def collect_split(hf_split: str, target_count: int, img_dir: str, prefix: str):
#     ds = load_dataset("sedrickkeh/snli-ve", split=hf_split, streaming=True)
#     rows = []
#     pos_count, neg_count = 0, 0
#     half = target_count // 2

#     for i, item in enumerate(ds):
#         if pos_count >= half and neg_count >= (target_count - half):
#             break

#         # sedrickkeh/snli-ve labels: 0 = entailment, 1 = neutral, 2 = contradiction
#         raw_label = item["label"]
#         label_str = str(raw_label).lower()
#         is_entail = (raw_label == 0) or ("entail" in label_str)
#         is_contra = (raw_label == 2) or ("contra" in label_str)

#         if not (is_entail or is_contra):
#             continue

#         img_name = f"{prefix}_{i}.jpg"
#         img_path = os.path.join(img_dir, img_name)

#         if is_entail and pos_count < half:
#             if not os.path.exists(img_path):
#                 item["image"].convert("RGB").save(img_path, format="JPEG", quality=90)
#             rows.append({
#                 "image": img_name,
#                 "text": item["hypothesis"].strip(),
#                 "label": 1,
#                 "dataset_source": "snli_ve"
#             })
#             pos_count += 1

#             # Also try creating a synthetic hard negative from this entailed caption
#             if neg_count < (target_count - half):
#                 hard_neg = make_hard_negative(item["hypothesis"].strip())
#                 if hard_neg and hard_neg != item["hypothesis"].strip():
#                     rows.append({
#                         "image": img_name,
#                         "text": hard_neg,
#                         "label": 0,
#                         "dataset_source": "snli_ve_synthetic_neg"
#                     })
#                     neg_count += 1

#         elif is_contra and neg_count < (target_count - half):
#             if not os.path.exists(img_path):
#                 item["image"].convert("RGB").save(img_path, format="JPEG", quality=90)
#             rows.append({
#                 "image": img_name,
#                 "text": item["hypothesis"].strip(),
#                 "label": 0,
#                 "dataset_source": "snli_ve"
#             })
#             neg_count += 1

#     return pd.DataFrame(rows)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_train", type=int, default=10)
#     parser.add_argument("--num_val", type=int, default=6)
#     parser.add_argument("--out_dir", type=str, default="vnli_data")
#     args = parser.parse_args()

#     random.seed(42)
#     img_dir = os.path.join(args.out_dir, "images")
#     os.makedirs(img_dir, exist_ok=True)

#     print(f"Streaming {args.num_train} train samples and {args.num_val} val samples from SNLI-VE...")
#     train_df = collect_split("train", args.num_train, img_dir, "train")
#     val_df = collect_split("validation", args.num_val, img_dir, "val")

#     train_csv = os.path.join(args.out_dir, "vnli_train.csv")
#     val_csv = os.path.join(args.out_dir, "vnli_val.csv")
#     train_df.to_csv(train_csv, index=False)
#     val_df.to_csv(val_csv, index=False)

#     print(f"Saved {len(train_df)} train rows -> {train_csv}")
#     print(f"Saved {len(val_df)} val rows   -> {val_csv}")
#     print("\nSample train rows:")
#     print(train_df[["image", "text", "label", "dataset_source"]].head(6).to_string(index=False))

# if __name__ == "__main__":
#     main()


import argparse
import os
import random
import re
import pandas as pd
from datasets import load_dataset

# Minimal-pair attribute/noun/count swapper to generate synthetic hard-negative contradictions
SWAP_MAP = {
    "red": "blue", "blue": "green", "green": "yellow", "yellow": "black",
    "black": "white", "white": "brown", "brown": "gray", "orange": "purple",
    "man": "woman", "woman": "man", "men": "women", "women": "men",
    "boy": "girl", "girl": "boy", "dog": "cat", "cat": "dog",
    "two": "three", "three": "four", "four": "two", "one": "two",
    "standing": "sitting", "sitting": "running", "running": "sleeping",
    "indoors": "outdoors", "outdoors": "indoors", "day": "night", "night": "day",
}


def make_hard_negative(text: str):
    words = text.split()
    indices = [i for i, w in enumerate(words) if re.sub(r"[^\w]", "", w.lower()) in SWAP_MAP]
    if not indices:
        return None
    idx = random.choice(indices)
    clean = re.sub(r"[^\w]", "", words[idx].lower())
    replacement = SWAP_MAP[clean]
    if words[idx][0].isupper():
        replacement = replacement.capitalize()
    words[idx] = re.sub(clean, replacement, words[idx], flags=re.IGNORECASE)
    return " ".join(words)


def collect_split(hf_split: str, target_count: int, img_dir: str, prefix: str, skip_valid: int = 0):
    ds = load_dataset("sedrickkeh/snli-ve", split=hf_split, streaming=True).shuffle(
        seed=42, buffer_size=5000
    )
    rows = []
    pos_count, neg_count = 0, 0
    skipped = 0
    half = target_count // 2

    for i, item in enumerate(ds):
        if pos_count >= half and neg_count >= (target_count - half):
            break

        # sedrickkeh/snli-ve labels: 0 = entailment, 1 = neutral, 2 = contradiction
        raw_label = item["label"]
        label_str = str(raw_label).lower()
        is_entail = (raw_label == 0) or ("entail" in label_str)
        is_contra = (raw_label == 2) or ("contra" in label_str)

        if not (is_entail or is_contra):
            continue

        # Skip already-trained rows when streaming subsequent incremental batches
        if skipped < skip_valid:
            skipped += 1
            continue

        global_idx = skip_valid + i
        img_name = f"{prefix}_{global_idx}.jpg"
        img_path = os.path.join(img_dir, img_name)

        if is_entail and pos_count < half:
            if not os.path.exists(img_path):
                item["image"].convert("RGB").save(img_path, format="JPEG", quality=90)
            rows.append({
                "image": img_name,
                "text": item["hypothesis"].strip(),
                "label": 1,
                "dataset_source": "snli_ve",
            })
            pos_count += 1

            # Also generate a synthetic hard negative from this entailed caption
            if neg_count < (target_count - half):
                hard_neg = make_hard_negative(item["hypothesis"].strip())
                if hard_neg and hard_neg != item["hypothesis"].strip():
                    rows.append({
                        "image": img_name,
                        "text": hard_neg,
                        "label": 0,
                        "dataset_source": "snli_ve_synthetic_neg",
                    })
                    neg_count += 1

        elif is_contra and neg_count < (target_count - half):
            if not os.path.exists(img_path):
                item["image"].convert("RGB").save(img_path, format="JPEG", quality=90)
            rows.append({
                "image": img_name,
                "text": item["hypothesis"].strip(),
                "label": 0,
                "dataset_source": "snli_ve",
            })
            neg_count += 1

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--skip_train", type=int, default=0, help="Skip first N valid train items for incremental batches")
    parser.add_argument("--num_val", type=int, default=300)
    parser.add_argument("--skip_val", action="store_true", help="Keep existing vnli_val.csv unchanged across batches")
    parser.add_argument("--out_dir", type=str, default="vnli_data")
    args = parser.parse_args()

    random.seed(42 + args.skip_train)
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Streaming {args.num_train} train samples (skipping first {args.skip_train})...")
    train_df = collect_split("train", args.num_train, img_dir, "train", skip_valid=args.skip_train)
    train_csv = os.path.join(args.out_dir, "vnli_train.csv")
    train_df.to_csv(train_csv, index=False)
    print(f"Saved {len(train_df)} train rows -> {train_csv}")

    val_csv = os.path.join(args.out_dir, "vnli_val.csv")
    if not args.skip_val or not os.path.exists(val_csv):
        print(f"Streaming {args.num_val} fixed validation samples...")
        val_df = collect_split("validation", args.num_val, img_dir, "val", skip_valid=0)
        val_df.to_csv(val_csv, index=False)
        print(f"Saved {len(val_df)} val rows   -> {val_csv}")
    else:
        print(f"Kept existing validation set -> {val_csv}")

    print("\nSample train rows:")
    print(train_df[["image", "text", "label", "dataset_source"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()