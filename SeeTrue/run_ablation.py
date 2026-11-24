import json, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

for split in ["drawbench", "coco_t2i", "editbench"]:
    df = pd.read_csv(f"outputs/{split}_vq2_scores.csv")
    qas = [json.loads(x) for x in df["qa_breakdown"]]
    mean_s = df["vq2_score"]
    min_s = [min(q["p_yes"] for q in row) if row else 0.5 for row in qas]
    prod_s = [np.prod([q["p_yes"] for q in row]) if row else 0.5 for row in qas]
    print(f"{split:10s} | mean: {roc_auc_score(df['label'], mean_s)*100:.2f}% | "
          f"min: {roc_auc_score(df['label'], min_s)*100:.2f}% | "
          f"prod: {roc_auc_score(df['label'], prod_s)*100:.2f}%")