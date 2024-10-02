import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer from the checkpoint directory
model = GPT2LMHeadModel.from_pretrained('./results/checkpoint-78')  # Adjust path to your checkpoint directory
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Load the base tokenizer

# Print the checkpoint load confirmation
print("Loaded model from the latest checkpoint in './results/'")

# Set pad token id
tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Start the conversation
print("AI: How can I help you today?")  # AI greets the user

while True:
    # Get user input
    input_text = input("You: ")

    # Exit condition
    if input_text.lower() in ['exit', 'quit', 'bye']:
        print("AI: Goodbye! Have a great day!")
        break

    # Prepare the input for the model
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate output
    output = model.generate(input_ids, max_new_tokens=50)  # You can adjust max_new_tokens as needed
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print AI response
    print("AI:", output_text)
