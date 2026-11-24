# SeeTRUE & QLoRA-VNLI: Fine-Grained Text-to-Image Alignment Evaluation
 
> Reproduction and hardware-efficient 4-bit QLoRA adaptation of [Yarom et al. (NeurIPS 2023)](https://arxiv.org/abs/2305.10400), *"What You See is What You Read? Improving Text-Image Alignment Evaluation"*, trained and deployed locally on an NVIDIA RTX 4050 GPU (6 GB VRAM).
 
## Overview
 
Text-to-image generation models frequently struggle with fine-grained semantic alignment — such as incorrect object counts, swapped spatial relations, or wrong attributes — that coarse global metrics like CLIPScore fail to detect. This repository implements the full SeeTRUE evaluation framework locally on consumer 6 GB VRAM hardware by combining:
 
- **Zero-Shot VQ² (`SeeTrue/`):** Decomposes an input caption into atomic semantic spans (spaCy + `mrm8488/t5-base-finetuned-question-generation-ap`) and verifies each generated question–answer pair against the image using frozen `Salesforce/blip2-flan-t5-xl`.
- **4-Bit QLoRA-VNLI (`VNLI/`):** Fine-tunes `Salesforce/blip2-flan-t5-xl` (3.95B parameters, 4-bit NF4 quantization via bitsandbytes, LoRA rank r = 16, 10.38M trainable parameters) as a Visual Natural Language Inference classifier (yes/no entailment scoring) on 13,026 pairs combining SNLI-VE (10,000) and NLI-verified minimal contradictions (COCO-Con 1,658 + PickaPic-Con 1,368).
- **SeeTRUE Ensemble & Interactive Gradio App (`VNLI/app.py`):** Combines both branches (S_SeeTRUE = 0.5 · S_VQ² + 0.5 · P_VNLI) inside a dark-themed offline web interface with atomic question-by-question verification breakdowns.
## Project Structure
 
```
├── README.md                          # Project overview, setup, results, and code attribution
├── requirements.txt                   # Python environment dependencies
├── .gitignore                         # Excludes local virtualenvs, caches, and raw dataset images
├── app_vq2.py                         # Standalone Zero-Shot VQ2 Gradio prototype
├── winoground_cartesian.csv           # Winoground 400-example image-caption pairing metadata
├── winoground_qa_cache.json           # Cached T5-QG questions for Winoground captions
├── winoground_vq2_scores.csv          # Zero-Shot VQ2 evaluation scores on Winoground
│
├── SeeTrue/                           # Zero-Shot VQ2 evaluation pipeline & benchmark outputs
│   ├── SeeTRUE.csv                    # Master SeeTRUE evaluation index (DrawBench, COCO-T2I, EditBench)
│   ├── run_vq2_full.py                # Full Zero-Shot VQ2 evaluation across all 6,930 SeeTRUE pairs
│   ├── run_vq2_local.py               # Single-split local VQ2 evaluation runner
│   ├── run_editbench_Categories.py    # Fine-grained EditBench category breakdown analysis
│   ├── run_ablation.py                # VQ2 + VNLI ensemble weighting ablation script
│   ├── test_batch_local.py            # Sanity-check inference script for local batch verification
│   └── outputs/                       # Zero-Shot VQ2 score CSVs, QA caches, and ROC curves
│       ├── drawbench_vq2_scores.csv / _qa_cache.json / _roc_curve.png
│       ├── coco_t2i_vq2_scores.csv  / _qa_cache.json / _roc_curve.png
│       └── editbench_vq2_scores.csv / _qa_cache.json / _roc_curve.png
│
└── VNLI/                              # 4-Bit QLoRA-VNLI training, ConGen data & Unified Gradio App
    ├── app.py                         # Unified Gradio App (QLoRA-VNLI + Zero-Shot VQ2 + Ensemble)
    ├── prepare_vnli_data.py           # SNLI-VE preprocessing and entailment prompt formatting
    ├── build_congen_datasets.py       # COCO-Con & PickaPic-Con NLI-verified contradiction generator
    ├── train_vnli_qlora.py            # 4-bit NF4 QLoRA fine-tuning pipeline (BLIP-2 Flan-T5-XL)
    ├── eval_vnli.py                   # SeeTRUE benchmark evaluation (VNLI & Ensemble ROC-AUC)
    ├── eval_winoground.py             # Winoground Text, Image & Group Score evaluation
    ├── analyze_mismatch_types.py      # Semantic mismatch categorization & diagnostic plots
    ├── vnli_adapter_b8_congen/        # Final frozen LoRA weights (SNLI-VE + COCO-Con + PickaPic-Con)
    ├── vnli_adapter_b7_latest/        # Baseline LoRA weights (10k SNLI-VE only)
    ├── vnli_data/                     # Processed CSV splits (train/val/ConGen metadata)
    └── eval_results_b8_congen/        # Final evaluation metrics, ROC curves & diagnostic CSVs
```
 
## Quickstart
 
### 1. Environment Setup
 
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
 
### 2. Run the Unified Gradio App (Offline Mode)
 
```bash
.\venv\Scripts\Activate.ps1
cd VNLI
$env:HF_HUB_OFFLINE="1"
python -u app.py
```
 
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser to access:
 
| Tab | Description |
|-----|-------------|
| **Unified SeeTRUE Alignment (VNLI + VQ²)** | Full ensemble verdict, confidence breakdown, and per-span VQ² question-answering verification table |
| **Fast VNLI Single-Pass Check** | Sub-second single-pass entailment probability scoring |
| **Model & Benchmark Summary** | Frozen architecture specs and quantitative benchmark results |
 
### 3. Training & Evaluation Pipeline (Reference)
 
```bash
# 1. Prepare SNLI-VE and build NLI-verified ConGen datasets
python VNLI/prepare_vnli_data.py
python VNLI/build_congen_datasets.py
 
# 2. Train 4-bit QLoRA-VNLI adapter on BLIP-2 Flan-T5-XL
python VNLI/train_vnli_qlora.py
 
# 3. Evaluate on SeeTRUE (6,930 pairs) and Winoground (400 examples)
python VNLI/eval_vnli.py --adapter_dir VNLI/vnli_adapter_b8_congen
python VNLI/eval_winoground.py --adapter_dir VNLI/vnli_adapter_b8_congen
```
 
## Quantitative Results
 
### ROC-AUC (%) & Winoground Group Score
 
| Benchmark / Metric | Pairs | Zero-Shot VQ² | QLoRA-VNLI (10k SNLI-VE) | QLoRA-VNLI (10k + 3k ConGen) |
|---------------------|------:|:-------------:|:------------------------:|:----------------------------:|
| **SeeTRUE: DrawBench** | 1,312 | 74.58 | 82.56 | **83.28** |
| **SeeTRUE: COCO-T2I** | 1,791 | 65.93 | **82.91** | 81.47 |
| **SeeTRUE: EditBench** | 3,827 | 65.83 | **73.90** | 70.92 |
| **SeeTRUE Macro Average** | 6,930 | 68.78 | **79.79** | 78.56 |
| **Winoground Pairwise ROC-AUC** | 1,600 | 56.38 | **62.15** | 61.74 |
| **Winoground Group Score** | 400 | 11.50 | 25.25 | **25.50** |
 
## Code Attribution & References
 
| Component / Artifact | Source & Attribution |
|----------------------|----------------------|
| `Salesforce/blip2-flan-t5-xl` | Pretrained Vision-Language backbone by Li et al. (ICML 2023) via Hugging Face Transformers |
| `mrm8488/t5-base-finetuned-question-generation-ap` | Pretrained T5 Question Generation checkpoint by Romero (Hugging Face Hub) |
| 4-Bit NF4 Quantization & LoRA (`bitsandbytes`, `peft`) | QLoRA methodology by Dettmers et al. (NeurIPS 2023) |
| SeeTRUE, VQ², VNLI & ConGen Formulation | Benchmark structure and training methodology from Yarom et al. (NeurIPS 2023) |
| `SeeTrue/*.py`, `VNLI/*.py`, `VNLI/app.py` | Original implementation, 6 GB VRAM optimization, ConGen NLI filtering, evaluation & Gradio UI |
 
## Citation
 
```bibtex
@inproceedings{yarom2023what,
  title     = {What You See is What You Read? Improving Text-Image Alignment Evaluation},
  author    = {Yarom, Michal and Bitton, Yonatan and Changpinyo, Soravit and Aharoni, Roee and Herzig, Jonathan and Lang, Oran and Ofek, Eran and Szpektor, Idan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
```
 