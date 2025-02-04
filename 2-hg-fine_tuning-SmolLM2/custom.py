import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# ============================================================
# Step 1: Load the custom dataset using Hugging Face's datasets library.
# ============================================================
dataset_file = "gor_dataset.jsonl"
raw_dataset = load_dataset("json", data_files={"train": dataset_file}, split="train")

# ============================================================
# Step 2: Preprocess the dataset.
# Combine the "problem" and "solution" fields into a single text string.
# ============================================================
def preprocess_function(example):
    return {"text": f"Question: {example['problem']}\nAnswer: {example['solution']}\n"}

processed_dataset = raw_dataset.map(preprocess_function)

# ============================================================
# Step 3: Tokenize the dataset.
# We'll use the tokenizer from our chosen checkpoint.
# ============================================================
checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=256)

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)

# ============================================================
# Step 4: Create a data collator.
# For causal language modeling, we set mlm=False.
# ============================================================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============================================================
# Step 5: Load the base model from the checkpoint.
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# ============================================================
# Step 6: Define training arguments.
# Adjust number of epochs and batch size for your hardware.
# ============================================================
training_args = TrainingArguments(
    output_dir="./finetuned_smol_model",
    overwrite_output_dir=True,
    num_train_epochs=5,                  # Increase epochs to let the small dataset have a strong impact.
    per_device_train_batch_size=1,       # Low batch size for laptop GPU/CPU.
    save_steps=50,                       # Save checkpoints every 50 steps.
    logging_steps=10,                    # Log progress every 10 steps.
    prediction_loss_only=True,
)

# ============================================================
# Step 7: Initialize the Trainer and fine-tune the model.
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

print("Starting fine-tuning on custom dataset with checkpoint", checkpoint)
trainer.train()
trainer.save_model("./finetuned_smol_model")
print("Fine-tuning complete. Model saved to './finetuned_smol_model'")
