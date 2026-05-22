"""
Fine-tune Qwen3 on incremental agent trajectory data.

Uses loss masking to train only on the final assistant response in each example —
the model learns what to think and do at each step, not how to reproduce context.

Usage:
    python finetune.py

Configuration is at the top of the file. Adjust MODEL_NAME, MAX_SEQ_LENGTH,
and TRAINING_DATA path for your setup.
"""

import json
import torch
import os
import glob
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "unsloth/Qwen3-30B-A3B-Thinking-2507"
MAX_SEQ_LENGTH = 49152
LOAD_IN_4BIT = True
TRAINING_DATA = "sample_training_data.jsonl"
OUTPUT_DIR = "./qwen-trajectory-optimized"
CHECKPOINT_DIR = "./checkpoints"

# LoRA
LORA_R = 32
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# Training
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
WARMUP_STEPS = 5

# =============================================================================
# Model setup
# =============================================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
    device_map="balanced",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

model.print_trainable_parameters()

# =============================================================================
# Data processing
# =============================================================================

def normalize_user_content(messages):
    """Convert user content from list format [{"text": ..., "type": "text"}] to plain string."""
    normalized = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            text = "".join(item["text"] for item in msg["content"] if item.get("type") == "text")
            normalized.append({**msg, "content": text})
        else:
            normalized.append(msg)
    return normalized


def create_masked_example(example):
    """
    Create a training example with loss masking.

    The full conversation is tokenized, but only the FINAL assistant message
    contributes to loss. Everything before it (system, user, prior steps) is masked.

    This teaches the model: "given this conversation state, what should I produce?"
    without training it to reproduce the context that the framework provides.
    """
    messages = normalize_user_content(example["messages"])

    # Apply Qwen's chat template — this converts:
    #   reasoning_content → <think>...</think>
    #   tool_calls → <tool_call>...</tool_call>
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]
    labels = input_ids.clone()

    # Find where the final assistant message starts and mask everything before it
    text_parts = text.split("<|im_start|>assistant\n")

    if len(text_parts) > 1:
        # Reconstruct the prefix (everything up to and including the last assistant marker)
        final_assistant_text = text_parts[-1]
        prefix_text = text.rsplit("<|im_start|>assistant\n" + final_assistant_text, 1)[0]
        prefix_text += "<|im_start|>assistant\n"

        prefix_tokens = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        mask_until = len(prefix_tokens)

        # Mask: no gradient for context, only for the current decision
        labels[:mask_until] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# Load and process training data
print(f"Loading training data from {TRAINING_DATA}...")

data = []
with open(TRAINING_DATA, 'r') as f:
    for line in f:
        if line.strip():
            example = json.loads(line)
            if example.get("messages"):
                data.append(example)

print(f"Loaded {len(data)} examples")

print("Creating masked examples...")
masked_examples = []
for i, example in enumerate(data):
    result = create_masked_example(example)
    if result is not None:
        masked_examples.append(result)

dataset = Dataset.from_list(masked_examples)
print(f"Created {len(dataset)} training examples")

# Show masking stats for first example
if len(dataset) > 0:
    labels = dataset[0]["labels"]
    masked = sum(1 for l in labels if l == -100)
    trained = len(labels) - masked
    print(f"First example: {len(labels)} tokens total, {masked} masked, {trained} trained ({trained/len(labels)*100:.0f}%)")

# =============================================================================
# Training
# =============================================================================

training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=CHECKPOINT_DIR,
    save_strategy="steps",
    save_steps=5,
    save_total_limit=10,
    report_to="none",
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    eval_strategy="no",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=1,
    packing=False,
    args=training_args,
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Starting training...")
trainer.train()
print("Training complete.")

# Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained_merged(f"{OUTPUT_DIR}_merged", tokenizer)

print(f"LoRA adapters: {OUTPUT_DIR}")
print(f"Merged model: {OUTPUT_DIR}_merged")
