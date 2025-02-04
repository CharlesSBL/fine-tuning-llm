# import pandas as pd
# df = pd.read_json("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl", lines=True)

import os
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Step 1: Load the dataset from Hugging Face.
# Here we use the test split of "HuggingFaceH4/MATH-500" for demonstration.
# In practice, you would typically fine-tune on a training split.
dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")

# Step 2: Preprocess the dataset.
# We assume each example has "problem" and "solution" fields.
def preprocess_function(example):
    # Create a single text string that shows the problem and the answer.
    # This formatting helps the model learn to generate the answer given the question.
    return {"text": f"Question: {example['problem']}\nAnswer: {example['solution']}\n"}

# Apply the preprocessing.
dataset = dataset.map(preprocess_function)

# Step 3: Tokenize the dataset.
# Load a tokenizer for our small model (DistilGPT-2).
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# Ensure a pad token exists.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    # We truncate to max_length=512 (adjustable based on your GPU memory)
    return tokenizer(example["text"], truncation=True, max_length=512)

# Tokenize examples in batches for speed.
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Create a data collator.
# For causal language modeling, we do not use masking (mlm=False).
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 5: Load the base model.
# DistilGPT-2 is a lightweight model (approx. 82M parameters) suitable for laptop environments.
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Step 6: Define training arguments.
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,                   # Number of epochs (adjust based on your needs)
    per_device_train_batch_size=2,        # Lower batch size to fit in GPU memory on a laptop
    save_steps=500,                       # Save checkpoint every 500 steps
    save_total_limit=2,                   # Only keep the 2 most recent checkpoints
    prediction_loss_only=True,            # Only compute loss during training
    logging_steps=100,                    # Log every 100 steps
)

# Step 7: Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,  # For demo purposes, we use the same dataset for training
)

# Step 8: Fine-tune the model.
trainer.train()

# Step 9: Save the fine-tuned model.
trainer.save_model("./finetuned_model")

print("Fine-tuning complete. The model is saved in './finetuned_model'")
