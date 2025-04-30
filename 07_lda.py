# 07_lda.py

# ================================
# BÖLÜM 2B - LDA ile Boyut İndirgeme
# Adım 1: İşlenmiş veri setini oku ve LDA uygula
#
# Not: LDA ile boyut indirgeme yapılacak, bileşen sayısı 3 olarak seçilecek.
# İlk iki bileşenin sınıf ayrımı 2 boyutlu bir grafikle görselleştirilecek.
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

# Dosya yolları
input_path = '../data/05_encoded_data.xlsx'
lda_output_path = '../data/07_lda_transformed.xlsx'
lda_scatter_plot_path = '../data/lda_scatter_plot.png'

# [2B.a] Veri setini oku
print("[2B.a] Kodlanmış veri seti okunuyor...")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Veri seti dosyası bulunamadı: {input_path}")
df = pd.read_excel(input_path)

# [2B.b] Veri setinin boyutu
print(f"[2B.b] Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# [2B.c] Sayısal sütunları ve sınıf sütununu seç
numeric_columns = df.select_dtypes(include=[np.number]).columns.drop('Class')
X = df[numeric_columns]
y = df['Class']
print(f"[2B.c] Sayısal sütunlar: {list(numeric_columns)}")
print(f"[2B.c] Sınıf sütunu: Class")

# [2B.d] LDA uygula
print("[2B.d] LDA uygulanıyor...")
lda = LinearDiscriminantAnalysis(n_components=3)
X_lda = lda.fit_transform(X, y)
n_lda_components = 3
print(f"[2B.d] LDA bileşen sayısı: {n_lda_components}")

# [2B.e] LDA ilk iki bileşenin ayrım gücü grafiği
print("[2B.e] LDA ilk iki bileşenin ayrım gücü görselleştiriliyor...")
plt.figure(figsize=(8, 6))
for class_label in np.unique(y):
    mask = y == class_label
    plt.scatter(X_lda[mask, 0], X_lda[mask, 1], label=f'Class {class_label}', alpha=0.6)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA İlk İki Bileşenin Sınıf Ayrım Gücü')
plt.legend()
plt.grid(True)
plt.savefig(lda_scatter_plot_path)
plt.close()
print(f"[2B.e] LDA ayrım gücü grafiği kaydedildi: {lda_scatter_plot_path}")

# [2B.f] LDA ile dönüştürülmüş veriyi kaydet
lda_df = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(n_lda_components)])
lda_df['Class'] = y
print(f"[2B.f] LDA ile dönüştürülmüş veri kaydediliyor: {lda_output_path}")
lda_df.to_excel(lda_output_path, index=False)
if os.path.exists(lda_output_path):
    print(f"[2B.f] LDA veri başarıyla kaydedildi: {lda_output_path}")
else:
    raise FileNotFoundError(f"LDA veri kaydedilemedi: {lda_output_path}")

print("\n[2B] LDA ile boyut indirgeme adımı tamamlandı.")