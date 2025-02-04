# %%capture
# !pip install unsloth
# Also get the latest nightly Unsloth!
# !pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git


# STEP 1: Model and Tokenizer Initialization -----------------------------------------------------


from unsloth import FastLanguageModel
import torch

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048  # Choose any! We auto support RoPE Scaling internally!
DTYPE = None  # None for auto-detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage. Can be False.


# Model and Tokenizer Loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# STEP 2: PEFT Model Configuration -----------------------------------------------------


PEFT_CONFIG = {
    "r": 16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj", ],
    "lora_alpha": 16,
    "lora_dropout": 0,  # Supports any, but = 0 is optimized
    "bias": "none",  # Supports any, but = "none" is optimized
    "use_gradient_checkpointing": "unsloth",  # True or "unsloth" for very long context
    "random_state": 3407,
    "use_rslora": False,  # We support rank stabilized LoRA
    "loftq_config": None,  # And LoftQ
}

model = FastLanguageModel.get_peft_model(model, **PEFT_CONFIG)


# STEP 3: Data Preparation -----------------------------------------------------


ALPACA_PROMPT_FORMAT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def format_instruction(instruction, input, output):
    return ALPACA_PROMPT_FORMAT.format(instruction, input, output) + EOS_TOKEN


def formatting_prompts_func(examples):
    texts = [format_instruction(instruction, input, output)
             for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"])]
    return {"text": texts}


from datasets import load_dataset

dataset = load_dataset("FatumMaster/test_GOR", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)


# STEP 4: Training the Model -----------------------------------------------------


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

TRAINING_ARGS = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",  # Use this for WandB etc
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TRAINING_ARGS,
)


# STEP 5: Training Execution -----------------------------------------------------


trainer_stats = trainer.train()


# STEP 6: Inference -----------------------------------------------------


FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

INFERENCE_PROMPT = ALPACA_PROMPT_FORMAT.format(
    "Who created you and who is the Karol Sobolewski and GOR",  # instruction
    "",  # input
    "",  # output - leave this blank for generation!
)

inputs = tokenizer([INFERENCE_PROMPT], return_tensors="pt").to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)


# STEP 7: Saving and Pushing the Model -----------------------------------------------------


LORA_MODEL_NAME = "lora_model"

# Save LoRA adapters locally
model.save_pretrained(LORA_MODEL_NAME)
tokenizer.save_pretrained(LORA_MODEL_NAME)

# Push LoRA adapters to Hugging Face Hub
HUB_REPO_ID = "FatumMaster/GOR_test"
HUB_TOKEN = "..."

model.push_to_hub(HUB_REPO_ID, token=HUB_TOKEN)
tokenizer.push_to_hub(HUB_REPO_ID, token=HUB_TOKEN)

# STEP 8: Load and Test the Saved Model -----------------------------------------------------


if True:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_MODEL_NAME,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    inputs = tokenizer([INFERENCE_PROMPT], return_tensors="pt").to("cuda")

    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)


# STEP 9: GGUF Conversion and Pushing (Optional) -----------------------------------------------------


# Configuration for GGUF conversion and pushing
GGUF_REPO_ID = "FatumMaster/GOR_test"  # Change hf to your username!
GGUF_TOKEN = "..."


# Function to push GGUF model with error handling
def push_gguf_model(quantization_method):
    try:
        model.save_pretrained_gguf("model", tokenizer, quantization_method=quantization_method)
        model.push_to_hub_gguf(GGUF_REPO_ID, tokenizer, quantization_method=quantization_method, token=GGUF_TOKEN)
        print(f"Successfully pushed {quantization_method} GGUF model to {GGUF_REPO_ID}")
    except Exception as e:
        print(f"Error pushing {quantization_method} GGUF model: {e}")


# Save to multiple GGUF options - much faster if you want multiple!
if True:
    try:
        model.push_to_hub_gguf(
            GGUF_REPO_ID,  # Change hf to your username!
            tokenizer,
            quantization_method=["q4_k_m", "q8_0", "q5_k_m", ],
            token=GGUF_TOKEN,
        )
        print(f"Successfully pushed GGUF models to {GGUF_REPO_ID}")
    except Exception as e:
        print(f"Error pushing GGUF models: {e}")
