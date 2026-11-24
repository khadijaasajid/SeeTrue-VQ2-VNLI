# import json
# import pandas as pd
# from sklearn.metrics import roc_auc_score

# df = pd.read_csv("outputs/editbench_vq2_scores.csv")

# # Inspect original_dataset_id format first so category extraction is exact
# print("Sample original_dataset_id values:", df["original_dataset_id"].dropna().unique()[:10])

# # Extract EditBench category (Color, Count, Material, Shape, Size, etc.)
# CATEGORIES = ["color", "count", "material", "shape", "size"]

# def extract_category(val):
#     s = str(val).lower()
#     for cat in CATEGORIES:
#         if cat in s:
#             return cat.capitalize()
#     return "Other"

# df["category"] = df["original_dataset_id"].apply(extract_category)

# # Compute single-word-answer-only score ("w.o. multi-word answers" ablation from Table 7)
# def score_single_word_only(qa_json_str):
#     qas = json.loads(qa_json_str)
#     single = [q["p_yes"] for q in qas if len(str(q["a"]).strip().split()) == 1]
#     if not single:
#         return None
#     return sum(single) / len(single)

# df["vq2_single_word"] = df["qa_breakdown"].apply(score_single_word_only)

# # Build Table 7-style comparison across categories
# rows = []
# for method_name, col in [
#     ("VQ² (Ours - Full)", "vq2_score"),
#     ("VQ² (w.o. multi-word answers)", "vq2_single_word"),
# ]:
#     row_dict = {"Method": method_name}
#     cat_aucs = []
#     for cat in ["Color", "Count", "Material", "Shape", "Size"]:
#         sub = df[(df["category"] == cat) & (df[col].notna())]
#         if len(sub) > 0 and sub["label"].nunique() == 2:
#             auc = roc_auc_score(sub["label"], sub[col]) * 100
#             row_dict[cat] = f"{auc:.1f} (n={len(sub)})"
#             cat_aucs.append(auc)
#         else:
#             row_dict[cat] = "N/A"
#     row_dict["Mean (5 Cats)"] = f"{sum(cat_aucs)/len(cat_aucs):.1f}" if cat_aucs else "N/A"
#     row_dict["Overall EditBench"] = f"{roc_auc_score(df[df[col].notna()]['label'], df[df[col].notna()][col])*100:.2f}"
#     rows.append(row_dict)

# print("\n=== EDITBENCH CATEGORY BREAKDOWN (REPRODUCING TABLE 7) ===")
# print(pd.DataFrame(rows).to_string(index=False))

import re
import json
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("outputs/editbench_vq2_scores.csv")

# EditBench 5 attribute lexicons (Wang et al., 2023)
ATTR_PATTERNS = {
    "Color": r"\b(red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey|golden|silver|turquoise|beige|maroon|navy|cyan|magenta|colorful)\b",
    "Count": r"\b(one|two|three|four|five|six|seven|eight|nine|ten|single|double|triple|pair of|couple of|several|many|multiple|2|3|4|5|6|7|8|9|10)\b",
    "Material": r"\b(wooden|wood|metal|metallic|glass|plastic|leather|stone|brick|ceramic|porcelain|paper|cardboard|fabric|cloth|wool|cotton|denim|silk|velvet|rubber|steel|iron|copper|bronze|marble|concrete|fur|bamboo|straw|clay)\b",
    "Shape": r"\b(round|circular|circle|square|rectangular|rectangle|triangular|triangle|oval|spherical|sphere|cube|cubical|cylindrical|cylinder|heart-shaped|star-shaped|hexagonal|octagonal|curved|spiral|flat|pointed)\b",
    "Size": r"\b(small|tiny|little|miniature|mini|large|big|huge|giant|enormous|massive|tall|short|long|wide|narrow|thick|thin|oversized)\b",
}

def classify_prompt(text):
    t = str(text).lower()
    # Count keyword hits per category
    scores = {cat: len(re.findall(pat, t)) for cat, pat in ATTR_PATTERNS.items()}
    best_cat, best_hits = max(scores.items(), key=lambda x: x[1])
    return best_cat if best_hits > 0 else "Other"

# Assign consistent category per original_dataset_id (each ID corresponds to one base EditBench prompt)
id_to_cat = (
    df.groupby("original_dataset_id")["text"]
    .first()
    .apply(classify_prompt)
    .to_dict()
)
df["category"] = df["original_dataset_id"].map(id_to_cat)

def score_single_word_only(qa_json_str):
    qas = json.loads(qa_json_str)
    single = [q["p_yes"] for q in qas if len(str(q["a"]).strip().split()) == 1]
    return sum(single) / len(single) if single else None

df["vq2_single_word"] = df["qa_breakdown"].apply(score_single_word_only)

rows = []
for method_name, col in [
    ("VQ² (Ours - Full)", "vq2_score"),
    ("VQ² (w.o. multi-word answers)", "vq2_single_word"),
]:
    row_dict = {"Method": method_name}
    cat_aucs = []
    for cat in ["Color", "Count", "Material", "Shape", "Size"]:
        sub = df[(df["category"] == cat) & (df[col].notna())]
        if len(sub) > 0 and sub["label"].nunique() == 2:
            auc = roc_auc_score(sub["label"], sub[col]) * 100
            row_dict[cat] = f"{auc:.1f} (n={len(sub)})"
            cat_aucs.append(auc)
        else:
            row_dict[cat] = "N/A"
    row_dict["Mean (5 Cats)"] = f"{sum(cat_aucs)/len(cat_aucs):.1f}" if cat_aucs else "N/A"
    valid = df[df[col].notna()]
    row_dict["Overall EditBench"] = f"{roc_auc_score(valid['label'], valid[col])*100:.2f}"
    rows.append(row_dict)

print("\n=== EDITBENCH CATEGORY BREAKDOWN (TABLE 7 STYLE) ===")
print(pd.DataFrame(rows).to_string(index=False))