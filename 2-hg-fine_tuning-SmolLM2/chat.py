#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive chat script for your fine-tuned model.
This script loads the fine-tuned model from the local folder ("./finetuned_smol_model")
and uses the base model's tokenizer ("HuggingFaceTB/SmolLM2-135M-Instruct") to tokenize input and generate responses.
Type "exit" to quit the chat.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Load the tokenizer from the base model.
    # Fine-tuning typically does not change the tokenizer, so it's safe to load it from "HuggingFaceTB/SmolLM2-135M-Instruct".
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    # Load your fine-tuned model from the local directory.
    model = AutoModelForCausalLM.from_pretrained("./finetuned_smol_model")

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Chat session started. Type 'exit' to quit.")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        # Format the prompt: Here we assume a simple structure.
        prompt = f"Question: {user_input}\nAnswer:"

        # Tokenize the prompt.
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

        # Ensure attention_mask is included in the inputs
        attention_mask = inputs.get("attention_mask")

        # Generate a response.
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=150,             # Adjust max_length as needed.
            do_sample=True,
            attention_mask=attention_mask,
            top_k=50,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode the output and remove the prompt portion.
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text[len(prompt):].strip()

        print("Model:", response)

if __name__ == "__main__":
    main()
