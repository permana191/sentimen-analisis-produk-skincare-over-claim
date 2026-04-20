from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = Flask(__name__)

print("Sedang memanaskan mesin AI...")
try:
    model = tf.keras.models.load_model('model_lstm_skincare.h5')
    with open('tokenizer.pickle', 'rb') as f: tokenizer = pickle.load(f)
    with open('label_encoder.pickle', 'rb') as f: label_encoder = pickle.load(f)
    print("✅ Mesin AI (LSTM) Siap Digunakan!")
except:
    print("⚠️ Peringatan: Model belum dilatih. Jalankan train_model.py dulu.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    teks = request.get_json().get('teks', '')
    if not teks: return jsonify({'error': 'Teks kosong'}), 400
        
    seq = tokenizer.texts_to_sequences([teks])
    padded = pad_sequences(seq, maxlen=100)
    
    prediksi = model.predict(padded)
    kelas_index = np.argmax(prediksi)
    hasil = label_encoder.inverse_transform([kelas_index])[0]
    persen = float(np.max(prediksi) * 100)
    
    return jsonify({'sentimen': hasil, 'confidence': round(persen, 2)})

if __name__ == '__main__':
    app.run(debug=True)