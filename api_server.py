import time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import load_tokenizer, load_model, translate
from config import get_config

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for model and tokenizers
MODEL = None
TOKENIZER_SRC = None
TOKENIZER_TGT = None
CONFIG = None
DEVICE = None

def init_model():
    global MODEL, TOKENIZER_SRC, TOKENIZER_TGT, CONFIG, DEVICE
    CONFIG = get_config()
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Initializing model on device: {DEVICE}")
    
    TOKENIZER_SRC = load_tokenizer(CONFIG, CONFIG['lang_src'])
    TOKENIZER_TGT = load_tokenizer(CONFIG, CONFIG['lang_tgt'])
    MODEL = load_model(CONFIG, TOKENIZER_SRC, TOKENIZER_TGT, DEVICE)
    print("Model initialized successfully.")

@app.route('/api/translate', methods=['POST'])
def handle_translate():
    data = request.json
    sentence = data.get('text', '').strip()
    
    if not sentence:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        start_time = time.time()
        translation = translate(
            sentence, MODEL,
            TOKENIZER_SRC, TOKENIZER_TGT,
            CONFIG, DEVICE
        )
        end_time = time.time()
        
        return jsonify({
            'translation': translation,
            'source': sentence,
            'time_ms': int((end_time - start_time) * 1000)
        })
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE)
    })

if __name__ == '__main__':
    init_model()
    app.run(host='0.0.0.0', port=5000)