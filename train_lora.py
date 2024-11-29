import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer
from torch.optim import AdamW
from peft import import_utils

# Base model and tokenizer names.
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
#adapter_name = "./Llama-3.2-3B-Instruct-emotion-adapter"

# Load base model to GPU memory.
device = "cuda:0"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code = True).to(device)

# Load tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Dataset for fine-tuning.
training_dataset_name = "dair-ai/emotion"
training_dataset = load_dataset(training_dataset_name, split = "train")
test_dataset = load_dataset(training_dataset_name, split="test")
# Check the data.
print(training_dataset)

print(training_dataset[11])

# Training parameters for SFTTrainer.
training_arguments = TrainingArguments(
    output_dir = "./results",
         num_train_epochs = 12,
         per_device_train_batch_size = 16,
         gradient_accumulation_steps = 1,
         optim = "adamw_torch",
         save_steps = 1000,
         logging_steps = 1000,
         learning_rate = 4e-5,
         weight_decay = 0.001,
         fp16=False,
         bf16=True,
         max_grad_norm = 0.3,
         max_steps = -1,
         warmup_ratio = 0.03,
         group_by_length = True,
         evaluation_strategy = "epoch",
         lr_scheduler_type = "constant",
         report_to = "tensorboard"
)

peft_config = LoraConfig(
        lora_alpha = 16,
        lora_dropout = 0.1,
        r = 64,
        bias = "none",
        peft_type="LORA",
        task_type = "CAUSAL_LM"
)
# View the number of trainable parameters.
from peft import get_peft_model
peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()

# Initialize an SFT trainer.
sft_trainer = SFTTrainer(
        model = base_model,
        train_dataset = training_dataset,
        peft_config = peft_config,
        #dataset_text_field = "text",
        tokenizer = tokenizer,
        args = training_arguments,
        eval_dataset = test_dataset
)

# Run the trainer.
sft_trainer.train(resume_from_checkpoint=True)

# PEFT adapter name.
adapter_name = "Llama-3.2-3B-Instruct-emotion-adapter-final"

# Save PEFT adapter.
sft_trainer.model.save_pretrained(adapter_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)


from peft import PeftModel
# Adapt the base model with the adapter
new_model = PeftModel.from_pretrained(base_model, adapter_name)

# Merge adapter
new_model = new_model.merge_and_unload()

# Save the merged model into local
new_model.save_pretrained("Llama-3.2-3B-Instruct-emotion-final")