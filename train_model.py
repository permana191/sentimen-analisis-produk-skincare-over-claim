import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

print("🌸 [1/5] Memuat Dataset Skincare...")
df = pd.read_csv('dataset.csv')

# --- TAMBAHKAN DUA BARIS INI ---
# 1. Menghapus baris yang benar-benar kosong (NaN)
df = df.dropna(subset=['clean_comment'])
# 2. Memaksa semua data di kolom tersebut menjadi format teks (string)
df['clean_comment'] = df['clean_comment'].astype(str)

print("🌸 [2/5] Memproses Teks & Tokenisasi...")
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_comment'])
X = pad_sequences(tokenizer.texts_to_sequences(df['clean_comment']), maxlen=max_len)

label_encoder = LabelEncoder()
y = tf.keras.utils.to_categorical(label_encoder.fit_transform(df['label']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🌸 [3/5] Membangun Arsitektur Deep Learning (LSTM)...")
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    SpatialDropout1D(0.4),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("🌸 [4/5] Memulai Pelatihan Model...")
# Early stopping agar pelatihan lebih efisien
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])

print("🌸 [5/5] Menyimpan Model...")
model.save('model_lstm_skincare.h5')
with open('tokenizer.pickle', 'wb') as f: pickle.dump(tokenizer, f)
with open('label_encoder.pickle', 'wb') as f: pickle.dump(label_encoder, f)

print("✅ SELESAI! Model AI berhasil dibuat.")