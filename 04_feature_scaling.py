# 04_feature_scaling.py

# ================================
# BÖLÜM 1D - Özellik Ölçekleme
# Adım 1: İşlenmiş veri setini oku ve sayısal sütunları ölçekle
#
# Not: Özellik ölçekleme için StandardScaler yöntemi kullanılacaktır.
# Gerekçe: StandardScaler, farklı ölçeklerdeki özellikleri sıfır ortalamaya ve birim
# varyansa getirerek mesafe tabanlı ve gradyan tabanlı algoritmaların performansını
# artırır. Aykırı değerlerin önceden işlenmiş olması, yöntemin güvenilirliğini artırır.
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Dosya yolları
input_path = '../data/03_outlier_processed.xlsx'
output_path = '../data/04_scaled_data.xlsx'
plot_path = '../data/scaled_features_plot.png'

# [1D.a] Veri setini oku
print("[1D.a] Aykırı değer işlenmiş veri seti okunuyor...")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Veri seti dosyası bulunamadı: {input_path}")
df = pd.read_excel(input_path)

# [1D.b] Veri setinin boyutu
print(f"[1D.b] Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# [1D.c] Sayısal sütunları seç (Class hariç)
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(f"[1D.c] Ölçekleme için sayısal sütunlar: {list(numeric_columns)}")

# [1D.d] Ölçeklemeden önceki istatistikler
print("[1D.d] Ölçeklemeden önceki istatistikler (ortalama ve standart sapma):")
stats_before = df[numeric_columns].agg(['mean', 'std']).round(4)
print(stats_before)

# [1D.e] StandardScaler ile ölçekleme
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print("[1D.e] Sayısal sütunlar StandardScaler ile ölçeklendi.")

# [1D.f] Ölçeklemeden sonraki istatistikler
print("[1D.f] Ölçeklemeden sonraki istatistikler (ortalama ve standart sapma):")
stats_after = df_scaled[numeric_columns].agg(['mean', 'std']).round(4)
print(stats_after)

# [1D.g] Ölçeklenmiş özelliklerin dağılımını görselleştir
print("[1D.g] Ölçeklenmiş özelliklerin dağılımı görselleştiriliyor...")
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns[:4], 1):  # İlk 4 sütunu görselleştir
    plt.subplot(2, 2, i)
    plt.hist(df_scaled[col], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'{col} (Ölçeklenmiş)')
    plt.xlabel('Değer')
    plt.ylabel('Frekans')
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"[1D.g] Dağılım grafiği kaydedildi: {plot_path}")

# ================================
# Adım 2: Kaydet
# ================================

# "../data" dizini yoksa oluştur
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# [1D.h] Ölçeklenmiş veri setini kaydet
print(f"[1D.h] Ölçeklenmiş veri seti kaydediliyor: {output_path}")
df_scaled.to_excel(output_path, index=False)
if os.path.exists(output_path):
    print(f"[1D.h] Dosya başarıyla kaydedildi: {output_path}")
else:
    raise FileNotFoundError(f"Dosya kaydedilemedi: {output_path}")

print("\n[1D] Özellik ölçekleme adımı tamamlandı.")