from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from environment variables
API_KEY = os.getenv('API-KEY')

# Debug print statement to ensure API_KEY is loaded correctly
print(f"API-KEY: {API_KEY}")

# Load the Hugging Face model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, temperature=0.8, no_repeat_ngram_size=5, repetition_penalty=20.12, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Middleware to check API key
@app.before_request
def before_request():
    if request.endpoint != 'index':
        api_key = request.headers.get('API-KEY')  # Ensure correct header key is used
        if api_key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401

@app.route('/')
def index():
    return "Welcome to the TinyLlama text generation API. Use the /generate endpoint to generate text."

@app.route('/generate', methods=['POST'])
def generate():
    # Get the prompt from the request
    data = request.get_json()
    prompt = data.get('prompt', '')

    # Generate text using the model
    generated_text = generate_text(prompt)

    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
