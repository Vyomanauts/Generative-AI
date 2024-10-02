import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = load_dataset('json', data_files='formatted_data.json')

# Load tokenizer and model for GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Move model to GPU
model.to(device)

# Set pad token id to eos token id if not set
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    tokenized_output = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    tokenized_output['labels'] = tokenized_output['input_ids'].copy()
    return tokenized_output

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and eval
train_test_split = tokenized_datasets['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start training
trainer.train()

# Save model and tokenizer after training
model.save_pretrained(training_args.output_dir)  # Save model
tokenizer.save_pretrained(training_args.output_dir)  # Save tokenizer
