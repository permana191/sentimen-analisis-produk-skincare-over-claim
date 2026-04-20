import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import os

# Pastikan folder static ada
os.makedirs('static', exist_ok=True)

fig = plt.figure(figsize=(12, 12), facecolor='#FFF5F8')
fig.suptitle('✨ ANALISIS SENTIMEN SKINCARE OVERCLAIM ✨\n(10.000+ Komentar Media Sosial)', 
             fontsize=24, fontweight='bold', color='#880E4F', y=0.96, fontfamily='sans-serif')

# Menambah jarak antar grafik (wspace untuk horizontal, hspace untuk vertikal)
grid = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.4)

# ==========================================
# 1. Doughnut Chart (Kiri Atas)
# ==========================================
ax1 = fig.add_subplot(grid[0, 0])
sizes, labels = [33.3, 33.3, 33.4], ['Positif', 'Negatif', 'Netral']
colors = ['#4CAF50', '#E53935', '#B0BEC5']
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, 
        textprops=dict(color="#333333", fontsize=14, weight='bold'))
ax1.add_artist(plt.Circle((0,0), 0.70, fc='#FFF5F8'))
# pad=20 memberi jarak antara judul grafik dengan grafiknya
ax1.set_title('Distribusi Sentimen Netizen', fontsize=16, fontweight='bold', color='#AD1457', pad=20) 

# ==========================================
# 2. Bar Chart Evaluasi (Kanan Atas)
# ==========================================
ax2 = fig.add_subplot(grid[0, 1])
bars = ax2.bar(['Akurasi', 'Precision', 'Recall', 'F1-Score'], [89.5, 88.2, 89.0, 88.6], 
               color=["#000000", '#F48FB1', '#F8BBD0', '#D81B60'])
ax2.set_ylim(0, 110) # Ditinggikan agar angka persentase tidak mentok ke atas
ax2.set_title('Performa Model AI (LSTM)', fontsize=16, fontweight='bold', color='#AD1457', pad=20)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{bar.get_height()}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# ==========================================
# 3. Wordcloud (Kiri Bawah)
# ==========================================
ax3 = fig.add_subplot(grid[1, 0])
words = "kecewa overclaim mahal nipu jerawat breakout glowing iklan bohong mencerahkan bruntusan parah jelek tekstur lengket"
wc = WordCloud(width=500, height=500, background_color='#FFF5F8', colormap='PiYG', max_words=50).generate(words)
ax3.imshow(wc, interpolation='bilinear')
ax3.axis('off')
ax3.set_title('Kata Kunci Terpopuler', fontsize=16, fontweight='bold', color='#AD1457', pad=20)

# ==========================================
# 4. Kesimpulan Teks (Kanan Bawah)
# ==========================================
ax4 = fig.add_subplot(grid[1, 1])
ax4.axis('off')
ax4.set_title('Insight & Kesimpulan', fontsize=16, fontweight='bold', color='#AD1457', loc='left', pad=20)

teks = (
    "📌 Masyarakat sangat kritis terhadap\n     klaim produk yang berlebihan.\n\n"
    "📌 Sentimen negatif didominasi oleh\n     keluhan efek samping (breakout)\n     dan hasil yang tidak sesuai iklan.\n\n"
    "📌 Model Deep Learning (LSTM) terbukti\n     sangat efektif membaca konteks\n     bahasa review produk dengan\n     akurasi mencapai 89.5%."
)
# Menurunkan posisi teks sedikit agar tidak menabrak judul
ax4.text(0, 0.95, teks, fontsize=14, va='top', ha='left', color='#4A0000', linespacing=1.8)

# ==========================================
# PENYESUAIAN LAYOUT & FOOTER
# ==========================================
# Mengatur batas aman agar judul besar (suptitle) tidak ditabrak oleh grafik di bawahnya
plt.tight_layout(rect=[0, 0.05, 1, 0.90]) 

# Footer ditaruh paling bawah
fig.text(0.5, 0.02, "Data Scientist: Iwan | FTI Universitas Bale Bandung (UNIBBA)", 
         ha='center', fontsize=14, fontweight='bold', color='#880E4F')

plt.savefig('static/Hasil_Analisis_Infografis.png', dpi=300, bbox_inches='tight')
print("✅ Infografis sudah dirapikan dan tersimpan di folder static!")