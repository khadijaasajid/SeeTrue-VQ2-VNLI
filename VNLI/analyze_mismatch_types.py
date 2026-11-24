import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------
# 1. Semantic Mismatch Classifier for SeeTRUE Captions
# ---------------------------------------------------------
SPATIAL_PATTERNS = re.compile(
    r"\b("
    r"on top of|in front of|next to|behind|under|underneath|beneath|above|below|"
    r"beside|between|inside|outside|into|onto|over|across|through|around|against|"
    r"left of|right of|to the left|to the right|surrounding|surrounded by|"
    r"leaning on|sitting on|standing on|lying on|hanging from"
    r")\b",
    re.IGNORECASE,
)

ACTION_PATTERNS = re.compile(
    r"\b("
    r"eating|drinking|holding|carrying|riding|driving|playing|running|walking|"
    r"jumping|flying|swimming|kissing|hugging|hitting|watching|looking at|"
    r"throwing|catching|pushing|pulling|wearing|reading|writing|cooking|"
    r"climbing|chasing|feeding|petting|lifting|cutting|painting|singing|dancing"
    r")\b",
    re.IGNORECASE,
)

ATTRIBUTE_PATTERNS = re.compile(
    r"\b("
    r"red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey|golden|silver|"
    r"big|small|large|tiny|huge|tall|taller|short|shorter|long|longer|old|older|young|younger|"
    r"wooden|metal|metallic|glass|plastic|striped|spotted|fluffy|shiny|wet|dry|"
    r"happy|sad|angry|smiling|crying|open|closed|empty|full|broken|round|square"
    r")\b",
    re.IGNORECASE,
)

COUNT_PATTERNS = re.compile(
    r"\b("
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"several|many|multiple|pair of|couple of|dozen|group of|crowd of|\d+"
    r")\b",
    re.IGNORECASE,
)


def classify_semantic_tags(text: str):
    """
    Assigns multi-label semantic tags and a primary fine-grained category
    ordered from highest compositional complexity to basic object presence:
      1. Spatial Relation
      2. Action / Interaction
      3. Attribute (Color / Size / Material / State)
      4. Object & Count (Pure entity/count verification)
    """
    t = str(text)
    has_spatial = bool(SPATIAL_PATTERNS.search(t))
    has_action = bool(ACTION_PATTERNS.search(t))
    has_attr = bool(ATTRIBUTE_PATTERNS.search(t))
    has_count = bool(COUNT_PATTERNS.search(t))

    if has_spatial:
        primary = "Spatial Relation"
    elif has_action:
        primary = "Action / Interaction"
    elif has_attr:
        primary = "Attribute"
    else:
        primary = "Object / Count"

    return pd.Series(
        {
            "primary_category": primary,
            "tag_spatial": has_spatial,
            "tag_action": has_action,
            "tag_attribute": has_attr,
            "tag_object_count": (not has_spatial and not has_action and not has_attr) or has_count,
        }
    )


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return 100.0 * roc_auc_score(y_true, y_score)


def compute_winoground_quartets(df, score_col):
    text_hits, image_hits, group_hits, total = 0, 0, 0, 0
    for _, g in df.groupby("id"):
        if len(g) != 4:
            continue
        scores = dict(zip(g["img_caption"], g[score_col]))
        req = ["image_0_caption_0", "image_0_caption_1", "image_1_caption_0", "image_1_caption_1"]
        if not all(k in scores for k in req):
            continue
        s00, s01, s10, s11 = (scores[k] for k in req)
        t_ok = (s00 > s01) and (s11 > s10)
        i_ok = (s00 > s10) and (s11 > s01)
        text_hits += int(t_ok)
        image_hits += int(i_ok)
        group_hits += int(t_ok and i_ok)
        total += 1
    return {
        "quartets": total,
        "text": 100.0 * text_hits / max(total, 1),
        "image": 100.0 * image_hits / max(total, 1),
        "group": 100.0 * group_hits / max(total, 1),
    }


def main():
    base_dir = "eval_results_b7_10000"
    out_dir = os.path.join(base_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Part A: Load SeeTRUE Splits & Tag Semantic Mismatch Types
    # ---------------------------------------------------------
    splits = {
        "DrawBench": os.path.join(base_dir, "drawbench_vnli_results.csv"),
        "COCO-T2I": os.path.join(base_dir, "coco_t2i_vnli_results.csv"),
        "EditBench": os.path.join(base_dir, "editbench_vnli_results.csv"),
    }

    dfs = []
    for split_name, path in splits.items():
        df = pd.read_csv(path)
        df["split"] = split_name
        dfs.append(df)
    seetrue_df = pd.concat(dfs, ignore_index=True)
    seetrue_df["weighted_ens_08"] = 0.2 * seetrue_df["vq2_score"] + 0.8 * seetrue_df["vnli_score"]

    tag_cols = seetrue_df["text"].apply(classify_semantic_tags)
    seetrue_df = pd.concat([seetrue_df, tag_cols], axis=1)

    category_order = ["Object / Count", "Attribute", "Action / Interaction", "Spatial Relation"]
    rows = []
    for cat in category_order:
        sub = seetrue_df[seetrue_df["primary_category"] == cat]
        vq2_auc = safe_auc(sub["label"], sub["vq2_score"])
        vnli_auc = safe_auc(sub["label"], sub["vnli_score"])
        ens50_auc = safe_auc(sub["label"], sub["seetrue_ensemble_score"])
        ens80_auc = safe_auc(sub["label"], sub["weighted_ens_08"])
        rows.append(
            {
                "Semantic_Category": cat,
                "Samples": len(sub),
                "Pos_Rate_%": round(100.0 * sub["label"].mean(), 1),
                "VQ2_AUC_%": round(vq2_auc, 2),
                "VNLI_AUC_%": round(vnli_auc, 2),
                "Gain_VNLI_vs_VQ2": round(vnli_auc - vq2_auc, 2),
                "Ensemble_50_50_%": round(ens50_auc, 2),
                "Weighted_Ens_20_80_%": round(ens80_auc, 2),
            }
        )

    sem_summary = pd.DataFrame(rows)
    sem_summary.to_csv(os.path.join(out_dir, "seetrue_semantic_breakdown.csv"), index=False)

    print("\n=== 1. SeeTRUE Performance by Semantic Mismatch Type (All 6,930 Pairs) ===")
    print(sem_summary.to_string(index=False))

    # Also print per-split x category matrix for VNLI - VQ2 gap
    print("\n=== 2. Per-Split ROC-AUC Breakdown (VQ2 vs. VNLI) ===")
    split_cat_rows = []
    for split_name in ["DrawBench", "COCO-T2I", "EditBench"]:
        for cat in category_order:
            sub = seetrue_df[(seetrue_df["split"] == split_name) & (seetrue_df["primary_category"] == cat)]
            if len(sub) < 10:
                continue
            v_q = safe_auc(sub["label"], sub["vq2_score"])
            v_n = safe_auc(sub["label"], sub["vnli_score"])
            split_cat_rows.append(
                {
                    "Split": split_name,
                    "Category": cat,
                    "N": len(sub),
                    "VQ2_AUC": round(v_q, 2),
                    "VNLI_AUC": round(v_n, 2),
                    "Delta_(VNLI-VQ2)": round(v_n - v_q, 2),
                }
            )
    split_cat_df = pd.DataFrame(split_cat_rows)
    split_cat_df.to_csv(os.path.join(out_dir, "seetrue_split_x_category.csv"), index=False)
    print(split_cat_df.to_string(index=False))

    # ---------------------------------------------------------
    # Part B: Winoground Ground-Truth Linguistic Tag Breakdown
    # ---------------------------------------------------------
    wino_path = os.path.join(base_dir, "winoground_vnli_preds.csv")
    wino_df = pd.read_csv(wino_path)
    wino_df["ensemble_50"] = 0.5 * wino_df["vq2_score"] + 0.5 * wino_df["vnli_score"]
    wino_df["weighted_ens_08"] = 0.2 * wino_df["vq2_score"] + 0.8 * wino_df["vnli_score"]

    wino_rows = []
    for ctag, sub in wino_df.groupby("collapsed_tag"):
        q_vq2 = compute_winoground_quartets(sub, "vq2_score")
        q_vnli = compute_winoground_quartets(sub, "vnli_score")
        q_ens = compute_winoground_quartets(sub, "ensemble_50")
        wino_rows.append(
            {
                "Winoground_Tag": ctag,
                "Quartets": q_vnli["quartets"],
                "VQ2_AUC": round(safe_auc(sub["label"], sub["vq2_score"]), 2),
                "VNLI_AUC": round(safe_auc(sub["label"], sub["vnli_score"]), 2),
                "VQ2_Group_%": round(q_vq2["group"], 2),
                "VNLI_Group_%": round(q_vnli["group"], 2),
                "Ens50_Group_%": round(q_ens["group"], 2),
                "Group_Gain": round(q_vnli["group"] - q_vq2["group"], 2),
            }
        )

    wino_summary = pd.DataFrame(wino_rows).sort_values("Quartets", ascending=False)
    wino_summary.to_csv(os.path.join(out_dir, "winoground_tag_breakdown.csv"), index=False)

    print("\n=== 3. Winoground Ground-Truth Linguistic Tag Breakdown (400 Quartets) ===")
    print(wino_summary.to_string(index=False))

    # ---------------------------------------------------------
    # Part C: Qualitative Disagreement Cases for Error Taxonomy
    # ---------------------------------------------------------
    # Case 1: VNLI right, VQ2 wrong (largest margin)
    seetrue_df["vnli_correct"] = (seetrue_df["vnli_score"] >= 0.5).astype(int) == seetrue_df["label"]
    seetrue_df["vq2_correct"] = (seetrue_df["vq2_score"] >= 0.5).astype(int) == seetrue_df["label"]
    seetrue_df["score_gap"] = (seetrue_df["vnli_score"] - seetrue_df["vq2_score"]).abs()

    vnli_wins = seetrue_df[seetrue_df["vnli_correct"] & (~seetrue_df["vq2_correct"])].sort_values(
        "score_gap", ascending=False
    ).head(15)
    vq2_wins = seetrue_df[(~seetrue_df["vnli_correct"]) & seetrue_df["vq2_correct"]].sort_values(
        "score_gap", ascending=False
    ).head(15)
    both_wrong = seetrue_df[(~seetrue_df["vnli_correct"]) & (~seetrue_df["vq2_correct"])].sort_values(
        "score_gap", ascending=True
    ).head(15)

    vnli_wins["error_bucket"] = "VNLI_Correct_VQ2_Failed"
    vq2_wins["error_bucket"] = "VQ2_Correct_VNLI_Failed"
    both_wrong["error_bucket"] = "Both_Failed"
    disagreements = pd.concat([vnli_wins, vq2_wins, both_wrong], ignore_index=True)
    disagreements[
        ["error_bucket", "split", "primary_category", "image", "text", "label", "vq2_score", "vnli_score"]
    ].to_csv(os.path.join(out_dir, "disagreement_cases.csv"), index=False)

    # ---------------------------------------------------------
    # Part D: Generate Grouped Bar Charts
    # ---------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: SeeTRUE Semantic Category Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sem_summary))
    w = 0.26
    b1 = ax.bar(x - w, sem_summary["VQ2_AUC_%"], width=w, label="VQ² (Zero-Shot QG+VQA)", color="#4C72B0")
    b2 = ax.bar(x, sem_summary["VNLI_AUC_%"], width=w, label="VNLI (10k QLoRA)", color="#DD8452")
    b3 = ax.bar(x + w, sem_summary["Weighted_Ens_20_80_%"], width=w, label="Weighted Ensemble (0.2 VQ² + 0.8 VNLI)", color="#55A868")

    ax.set_ylabel("ROC-AUC (%)", fontsize=11, fontweight="bold")
    ax.set_title("VQ² vs. VNLI ROC-AUC Across Semantic Mismatch Categories (SeeTRUE)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['Semantic_Category']}\n(n={r['Samples']})" for _, r in sem_summary.iterrows()],
        fontsize=10,
    )
    ax.set_ylim(50, 92)
    ax.legend(frameon=True, fontsize=9.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "fig_seetrue_semantic_breakdown.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Winoground Collapsed Tag Group Score
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x2 = np.arange(len(wino_summary))
    w2 = 0.28
    g1 = ax.bar(x2 - w2, wino_summary["VQ2_Group_%"], width=w2, label="VQ² Group Score", color="#4C72B0")
    g2 = ax.bar(x2, wino_summary["VNLI_Group_%"], width=w2, label="VNLI (10k QLoRA) Group Score", color="#DD8452")
    g3 = ax.bar(x2 + w2, wino_summary["Ens50_Group_%"], width=w2, label="SeeTRUE Ensemble (0.5/0.5)", color="#55A868")

    ax.axhline(16.67, color="gray", linestyle="--", linewidth=1, label="Random Chance (16.67%)")
    ax.set_ylabel("Quartet Group Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Winoground Compositional Group Score by Ground-Truth Linguistic Tag", fontsize=12, fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels(
        [f"{r['Winoground_Tag']}\n({r['Quartets']} quartets)" for _, r in wino_summary.iterrows()],
        fontsize=10,
    )
    ax.set_ylim(0, 42)
    ax.legend(frameon=True, fontsize=9.5)

    for bars in [g1, g2, g3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "fig_winoground_tag_breakdown.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    print(f"\nSaved all analysis tables and 300-DPI figures to -> {out_dir}\\")


if __name__ == "__main__":
    main()