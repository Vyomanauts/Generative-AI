# Generative-AI-


This project is a conversational AI chatbot built using GPT-2/DialoGPT, fine-tuned on custom data. It features a web interface where users can chat with the AI model and even use text-to-speech for the AI's responses. The project includes:

Fine-tuning a GPT-2/DialoGPT model
Flask-based web interface for chat interaction
Example HTML/CSS for frontend design
Training script to fine-tune the model on your own dataset
Features
AI conversation: The chatbot engages in natural conversations with users.
Web-based UI: Users can interact with the chatbot using a simple web interface.
Customizable: Users can fine-tune the AI on their own datasets.
Text-to-Speech: Users can press a button to have the AI's response read aloud (if enabled).

FOLDER STRUCTURE

├── sample.py               # Model training script
├── text_model.py           # Text-based interaction with the chatbot
├── app.py                  # Flask application for web-based chat interface
├── templates/
│   └── index.html          # Web interface (HTML)
├── dataset.json            # Sample dataset (replace with your own for fine-tuning)
├── results/                # Directory where model checkpoints will be stored
├── README.md               # This file
├── LICENSE                 # License file

Run the sample.py file to train the model


