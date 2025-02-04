import json
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# ============================================================
# Step 1: Create a custom dataset (JSONL file) for our task.
# Our goal is to teach the model to always respond that it is the model of company "GOR".
# Each example contains a "problem" (e.g. a question) and a "solution" (the fixed answer).
# ============================================================

custom_data = [
    {"problem": "Who are you?", "solution": "I am the model of company GOR."},
    {"problem": "Tell me about yourself.", "solution": "I am the model of company GOR."},
    {"problem": "What is your purpose?", "solution": "I am the model of company GOR."},
    {"problem": "Who built you?", "solution": "I am the model of company GOR."},
    {"problem": "What is your identity?", "solution": "I am the model of company GOR."}
]

dataset_file = "gor_dataset.jsonl"
with open(dataset_file, "w") as f:
    for item in custom_data:
        f.write(json.dumps(item) + "\n")

print(f"Custom dataset saved to {dataset_file}")

# ============================================================
# Step 2: Load the custom dataset using Hugging Face's datasets library.
# We assume the file has our examples in JSONL format.
# ============================================================

raw_dataset = load_dataset("json", data_files={"train": dataset_file}, split="train")

# ============================================================
# Step 3: Preprocess the dataset.
# We'll combine the "problem" and "solution" fields into a single prompt text.
# ============================================================

def preprocess_function(example):
    # We format the text so that the model learns that for any question,
    # the answer is always: "I am the model of company GOR."
    return {"text": f"Question: {example['problem']}\nAnswer: {example['solution']}\n"}

processed_dataset = raw_dataset.map(preprocess_function)

# ============================================================
# Step 4: Tokenize the dataset.
# Using DistilGPT-2's tokenizer for our small causal language model.
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# Ensure the tokenizer has a pad token; if not, use eos_token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    # For our simple task, a max_length of 128 tokens is sufficient.
    return tokenizer(example["text"], truncation=True, max_length=128)

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)

# ============================================================
# Step 5: Create a data collator.
# For causal language modeling, we don't need masked language modeling (mlm=False).
# ============================================================

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============================================================
# Step 6: Load the base model.
# DistilGPT-2 is a lightweight, small model (~82M parameters) that is well-suited for laptop use.
# ============================================================

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# ============================================================
# Step 7: Define training arguments.
# We use a low batch size and a higher number of epochs (since our dataset is tiny).
# ============================================================

training_args = TrainingArguments(
    output_dir="./gor_finetuned_model",
    overwrite_output_dir=True,
    num_train_epochs=10,                   # More epochs help the model memorize the pattern from our small dataset.
    per_device_train_batch_size=1,         # Low batch size to work on a laptop.
    save_steps=100,                        # Save checkpoint every 100 steps.
    save_total_limit=2,
    logging_steps=10,
    prediction_loss_only=True,
)

# ============================================================
# Step 8: Initialize the Trainer and fine-tune the model.
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

print("Starting fine-tuning...")
trainer.train()

# ============================================================
# Step 9: Save the fine-tuned model.
# ============================================================
trainer.save_model("./gor_finetuned_model")
print("Fine-tuning complete. The model has been saved in './gor_finetuned_model'")

