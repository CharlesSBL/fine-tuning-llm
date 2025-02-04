#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive chat script for the baseline model (before fine-tuning).
This script loads the base DistilGPT-2 model and its tokenizer from Hugging Face,
and then starts an interactive session where the user can type a prompt and get a response.
Type 'exit' to quit.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Load the base model and tokenizer directly from Hugging Face.
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Baseline Chat Session Started (pre-fine-tuning). Type 'exit' to quit.")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        # Prepare prompt in the same style as our fine-tuning task.
        prompt = f"Question: {user_input}\nAnswer:"

        # Tokenize the input prompt.
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

        # Generate output using the model.
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=150,             # Total length (prompt + generated text)
            do_sample=True,
            top_p=0.95,
            top_k=50,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode the generated tokens and remove the prompt part.
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text[len(prompt):].strip()

        print("Model:", response)

if __name__ == "__main__":
    main()
