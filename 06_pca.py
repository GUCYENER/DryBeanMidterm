# 06_pca.py

# ================================
# BÖLÜM 2A - PCA ile Boyut İndirgeme
# Adım 1: İşlenmiş veri setini oku, PCA uygula ve ham veriyi kaydet
#
# Not: PCA ile boyut indirgeme yapılacak, açıklanan varyans oranlarının
# ortalamasından büyük bileşenler seçilecek. Açıklanan varyans oranları
# raporlanacak ve ilk iki bileşenin ayrım gücü görselleştirilecek.
# Ham veri bu adımda kaydedilecek.
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Dosya yolları
input_path = '../data/05_encoded_data.xlsx'
raw_output_path = '../data/06_raw_preprocessed.xlsx'
pca_output_path = '../data/06_pca_transformed.xlsx'
pca_variance_plot_path = '../data/pca_variance_plot.png'
pca_scatter_plot_path = '../data/pca_scatter_plot.png'

# [2A.a] Veri setini oku
print("[2A.a] Kodlanmış veri seti okunuyor...")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Veri seti dosyası bulunamadı: {input_path}")
df = pd.read_excel(input_path)

# [2A.b] Veri setinin boyutu
print(f"[2A.b] Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# [2A.c] Sayısal sütunları ve sınıf sütununu seç
numeric_columns = df.select_dtypes(include=[np.number]).columns.drop('Class')
X = df[numeric_columns]
y = df['Class']
print(f"[2A.c] Sayısal sütunlar: {list(numeric_columns)}")
print(f"[2A.c] Sınıf sütunu: Class")

# [2A.d] Ham veriyi kaydet
print(f"[2A.d] Ham veri kaydediliyor: {raw_output_path}")
df.to_excel(raw_output_path, index=False)
if os.path.exists(raw_output_path):
    print(f"[2A.d] Ham veri başarıyla kaydedildi: {raw_output_path}")
else:
    raise FileNotFoundError(f"Ham veri kaydedilemedi: {raw_output_path}")

# [2A.e] PCA uygula
print("[2A.e] PCA uygulanıyor...")
pca = PCA()
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
mean_variance_ratio = np.mean(explained_variance_ratio)
n_components = np.sum(explained_variance_ratio > mean_variance_ratio)

print("[2A.e] PCA açıklanan varyans oranları:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.4f} (Kümülatif: {cumulative_variance_ratio[i]:.4f})")
print(f"[2A.e] Ortalama varyans oranı: {mean_variance_ratio:.4f}")
print(f"[2A.e] Seçilen bileşen sayısı: {n_components}")

# [2A.f] PCA kümülatif varyans grafiği
print("[2A.f] PCA kümülatif varyans oranı görselleştiriliyor...")
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.axhline(y=cumulative_variance_ratio[n_components - 1], color='r', linestyle='--')
plt.axvline(x=n_components, color='r', linestyle='--')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('Kümülatif Açıklanan Varyans Oranı')
plt.title('PCA Kümülatif Açıklanan Varyans Oranı')
plt.grid(True)
plt.savefig(pca_variance_plot_path)
plt.close()
print(f"[2A.f] Kümülatif varyans grafiği kaydedildi: {pca_variance_plot_path}")

# [2A.g] PCA ilk iki bileşenin ayrım gücü grafiği
print("[2A.g] PCA ilk iki bileşenin ayrım gücü görselleştiriliyor...")
plt.figure(figsize=(8, 6))
for class_label in np.unique(y):
    mask = y == class_label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {class_label}', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA İlk İki Bileşenin Sınıf Ayrım Gücü')
plt.legend()
plt.grid(True)
plt.savefig(pca_scatter_plot_path)
plt.close()
print(f"[2A.g] PCA ayrım gücü grafiği kaydedildi: {pca_scatter_plot_path}")

# [2A.h] PCA ile dönüştürülmüş veriyi kaydet
pca_df = pd.DataFrame(X_pca[:, :n_components], columns=[f'PC{i+1}' for i in range(n_components)])
pca_df['Class'] = y
print(f"[2A.h] PCA ile dönüştürülmüş veri kaydediliyor: {pca_output_path}")
pca_df.to_excel(pca_output_path, index=False)
if os.path.exists(pca_output_path):
    print(f"[2A.h] PCA veri başarıyla kaydedildi: {pca_output_path}")
else:
    raise FileNotFoundError(f"PCA veri kaydedilemedi: {pca_output_path}")

print("\n[2A] PCA ile boyut indirgeme adımı tamamlandı.")