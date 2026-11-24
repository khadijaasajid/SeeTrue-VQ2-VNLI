# import argparse
# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from tqdm import tqdm


# class VNLIDataset(Dataset):
#     def __init__(self, csv_path: str, image_dir: str):
#         self.df = pd.read_csv(csv_path)
#         self.image_dir = image_dir

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_path = os.path.join(self.image_dir, str(row["image"]))
#         image = Image.open(img_path).convert("RGB")
#         text = str(row["text"]).strip()
#         prompt = f'Does this image entail the description: "{text}"?'
#         completion = "yes" if str(row["label"]) == "1" else "no"
#         return {"image": image, "prompt": prompt, "completion": completion}


# def make_collate_fn(processor):
#     def collate_fn(batch):
#         images = [item["image"] for item in batch]
#         prompts = [item["prompt"] for item in batch]
#         completions = [item["completion"] for item in batch]

#         inputs = processor(
#             images=images,
#             text=prompts,
#             padding=True,
#             truncation=True,
#             max_length=128,
#             return_tensors="pt",
#         )
#         labels = processor.tokenizer(
#             completions,
#             padding=True,
#             truncation=True,
#             max_length=8,
#             return_tensors="pt",
#         ).input_ids
#         labels[labels == processor.tokenizer.pad_token_id] = -100
#         inputs["labels"] = labels
#         return inputs

#     return collate_fn


# def evaluate(model, val_loader, device):
#     model.eval()
#     total_loss, steps = 0.0, 0
#     with torch.no_grad():
#         for batch in val_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             with torch.amp.autocast("cuda", dtype=torch.bfloat16):
#                 outputs = model(**batch)
#                 loss = outputs.loss
#             total_loss += loss.item()
#             steps += 1
#     return total_loss / max(steps, 1)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_csv", type=str, default="vnli_data/vnli_train.csv")
#     parser.add_argument("--val_csv", type=str, default="vnli_data/vnli_val.csv")
#     parser.add_argument("--image_dir", type=str, default="vnli_data/images")
#     parser.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl")
#     parser.add_argument("--output_dir", type=str, default="vnli_qlora_adapter")
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--grad_accum", type=int, default=4)
#     parser.add_argument("--epochs", type=int, default=2)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--resume_adapter", type=str, default=None)
#     args = parser.parse_args()

#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     torch.cuda.manual_seed_all(42)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

#     print("Loading BLIP-2 processor and 4-bit base model...")
#     processor = Blip2Processor.from_pretrained(args.model_id)
#     base_model = Blip2ForConditionalGeneration.from_pretrained(
#         args.model_id,
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,
#         device_map={"": 0},
#         low_cpu_mem_usage=True,
#     )

#     # Freeze vision encoder to keep VRAM well below 6 GB
#     for param in base_model.vision_model.parameters():
#         param.requires_grad = False

#     base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

#     if args.resume_adapter and os.path.exists(args.resume_adapter):
#         from peft import PeftModel
#         print(f"Resuming LoRA adapter from {args.resume_adapter}...")
#         model = PeftModel.from_pretrained(base_model, args.resume_adapter, is_trainable=True)
#     else:
#         lora_config = LoraConfig(
#             r=16,
#             lora_alpha=32,
#             target_modules=["q", "v", "query", "value"],
#             lora_dropout=0.05,
#             bias="none",
#         )
#         model = get_peft_model(base_model, lora_config)

#     model.print_trainable_parameters()

#     collate_fn = make_collate_fn(processor)
#     train_loader = DataLoader(
#         VNLIDataset(args.train_csv, args.image_dir),
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=0,
#         collate_fn=collate_fn,
#     )
#     val_loader = DataLoader(
#         VNLIDataset(args.val_csv, args.image_dir),
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=collate_fn,
#     )

#     optimizer = torch.optim.AdamW(
#         [p for p in model.parameters() if p.requires_grad],
#         lr=args.lr,
#         weight_decay=0.01,
#     )

#     best_val_loss = float("inf")
#     os.makedirs(args.output_dir, exist_ok=True)

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         optimizer.zero_grad()
#         running_loss = 0.0
#         pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")

#         for step, batch in pbar:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             with torch.amp.autocast("cuda", dtype=torch.bfloat16):
#                 outputs = model(**batch)
#                 loss = outputs.loss / args.grad_accum

#             loss.backward()
#             running_loss += loss.item() * args.grad_accum

#             if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 optimizer.zero_grad()

#             pbar.set_postfix({"train_loss": f"{running_loss / (step + 1):.4f}"})

#         val_loss = evaluate(model, val_loader, device)
#         print(f"Epoch {epoch} | Train Loss: {running_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             model.save_pretrained(args.output_dir)
#             processor.save_pretrained(args.output_dir)
#             print(f"Saved improved LoRA adapter -> {args.output_dir} (val_loss={best_val_loss:.4f})")

#     if torch.cuda.is_available():
#         peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
#         print(f"Peak VRAM allocated: {peak_gb:.2f} GB")


# if __name__ == "__main__":
#     main()

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


class VNLIDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row["image"]))
        image = Image.open(img_path).convert("RGB")
        text = str(row["text"]).strip()
        prompt = f'Does this image entail the description: "{text}"?'
        completion = "yes" if str(row["label"]) == "1" else "no"
        return {"image": image, "prompt": prompt, "completion": completion}


def make_collate_fn(processor):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        completions = [item["completion"] for item in batch]

        inputs = processor(
            images=images,
            text=prompts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        labels = processor.tokenizer(
            completions,
            padding=True,
            truncation=True,
            max_length=8,
            return_tensors="pt",
        ).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    return collate_fn


def evaluate(model, val_loader, device):
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="vnli_data/vnli_train.csv")
    parser.add_argument("--val_csv", type=str, default="vnli_data/vnli_val.csv")
    parser.add_argument("--image_dir", type=str, default="vnli_data/images")
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--output_dir", type=str, default="vnli_qlora_adapter")
    parser.add_argument("--resume_adapter", type=str, default=None, help="Path to existing adapter to continue training")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading BLIP-2 processor and 4-bit base model...")
    processor = Blip2Processor.from_pretrained(args.model_id)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    # Freeze vision encoder to keep VRAM well below 6 GB
    for param in base_model.vision_model.parameters():
        param.requires_grad = False

    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

    if args.resume_adapter and os.path.exists(args.resume_adapter):
        print(f"Resuming LoRA adapter from: {args.resume_adapter}")
        model = PeftModel.from_pretrained(base_model, args.resume_adapter, is_trainable=True)
    else:
        print("Initializing fresh LoRA adapter...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v", "query", "value"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(
        VNLIDataset(args.train_csv, args.image_dir),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        VNLIDataset(args.val_csv, args.image_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    initial_val_loss = evaluate(model, val_loader, device)
    print(f"Initial Val Loss before batch: {initial_val_loss:.4f}")
    best_val_loss = initial_val_loss
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            loss.backward()
            running_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix({"train_loss": f"{running_loss / (step + 1):.4f}"})

        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | Train Loss: {running_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"Saved improved LoRA adapter -> {args.output_dir} (val_loss={best_val_loss:.4f})")

    # Also save latest batch end-state so incremental chaining always preserves progress
    latest_dir = f"{args.output_dir}_latest"
    model.save_pretrained(latest_dir)
    processor.save_pretrained(latest_dir)
    print(f"Saved batch end-state adapter -> {latest_dir}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM allocated: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()