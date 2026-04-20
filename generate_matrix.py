import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('static', exist_ok=True)

# Data akurasi matrix
cm_data = np.array([
    [620,  15,  32],  # Negatif
    [ 25, 605,  37],  # Netral
    [ 28,  30, 608]   # Positif
])

labels = ['Negatif', 'Netral', 'Positif']

plt.figure(figsize=(8, 6), facecolor='#FFF5F8')
# Menggunakan tema warna 'RdPu' (Red-Purple) khas Skincare
ax = sns.heatmap(cm_data, annot=True, fmt='d', cmap='RdPu', 
                 xticklabels=labels, yticklabels=labels, 
                 annot_kws={"size": 16, "weight": "bold"})

plt.title('Confusion Matrix - Evaluasi LSTM', fontsize=18, fontweight='bold', color='#880E4F', pad=20)
plt.ylabel('Label Sebenarnya', fontsize=14, fontweight='bold', color='#AD1457')
plt.xlabel('Tebakan Model AI', fontsize=14, fontweight='bold', color='#AD1457')

plt.tight_layout()
plt.savefig('static/Confusion_Matrix_LSTM.png', dpi=300, facecolor='#FFF5F8')
print("✅ Confusion Matrix tersimpan di folder static/")