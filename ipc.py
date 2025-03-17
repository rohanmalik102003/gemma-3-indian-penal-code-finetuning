from unsloth import FastModel
import torch
import json
from datasets import Dataset

# Track memory usage
start_gpu_memory = 0
if torch.cuda.is_available():
    start_gpu_memory = round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, 3)
    max_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3)
    print(f"Starting GPU memory: {start_gpu_memory} GB")
    print(f"Max GPU memory: {max_memory} GB")

# Load the model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)

# Setup PEFT model
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# Set up the chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# Load and prepare the data - Specify the full path to your JSON file here
json_file_path = "/content/IPC_cleaned_fine_tuning.json"  # IMPORTANT: Replace with the actual path to your case.json file
with open(json_file_path, "r") as f:
    legal_cases = json.load(f)

# Prepare the dataset
dataset_records = []
for case in legal_cases:
    dataset_records.append({
        "conversations": case["messages"]
    })

dataset = Dataset.from_list(dataset_records)

print("Sample from dataset:")
print(dataset[0])

def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return {"text": texts}

dataset = dataset.map(apply_chat_template, batched=True)

print("\nProcessed sample:")
print(dataset[0]["text"])

# Set up the trainer
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 2,
        #num_train_epochs = 2,
        max_steps=100,
        learning_rate = 3e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    ),
)

# Optimize training to focus on the assistant's responses
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

print("\nDecoded input_ids sample:")
print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))

print("\nDecoded labels sample (showing only non-masked tokens):")
print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " "))

# Train the model
trainer_stats = trainer.train()

# Show memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3) if torch.cuda.is_available() else 0
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3) if torch.cuda.is_available() else 0
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# List of test questions
test_questions = [
    "What is the punishment for stealing someone's property? What is the punishment for theft under the Indian Penal Code (IPC)?",
    "What is the difference between culpable homicide and murder under the Indian Penal Code?",
    "Explain the concept of 'mens rea' in Indian criminal law. How does it affect criminal liability?",
    "What are the punishments for different degrees of hurt under the IPC?",
    "What constitutes criminal conspiracy under Section 120A of the IPC? What is the punishment for it?",
    "Under what circumstances can self-defense be claimed as a valid defense against criminal charges in India?",
    "What is the legal definition of 'dowry death' under Section 304B of the IPC? What is the punishment for this offense?",
    "Explain the concept of 'abetment' under the IPC. How is an abettor punished?"
]
# Test each question
for i, question in enumerate(test_questions):
    print(f"\n\n===== TESTING QUESTION {i+1} =====\n")
    print(f"Question: {question}\n")
    
    messages = [
        {"role": "system", "content": "You are an expert legal assistant providing accurate answers based on the Indian Penal Code (IPC)."},
        {"role": "user", "content": question},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
    )
    
    outputs = model.generate(
        **tokenizer([text], return_tensors = "pt").to("cuda"),
        max_new_tokens = 512,  # Increased for more complete answers
        temperature = 0.7,
        top_p = 0.95,
        top_k = 64,
    )
    
    print("Model response:")
    print(tokenizer.batch_decode(outputs)[0])
    print("\n" + "="*50)

model.save_pretrained("gemma-3-indian-penal-code-model")  # Local saving
tokenizer.save_pretrained("gemma-3-indian-penal-code-model")
