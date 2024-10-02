import warnings
from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./results/checkpoint-78')  # Make sure this path is consistent with your training
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set pad token id
tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)  # Ensure input is on the right device

    # Generate output with specified parameters
    output = model.generate(input_ids, max_new_tokens=50)  # Add generation parameters as needed
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'response': output_text})

if __name__ == '__main__':
    app.run(debug=True)
