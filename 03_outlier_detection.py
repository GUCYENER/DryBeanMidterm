# 03_outlier_detection.py

# ================================
# BÖLÜM 1C - Aykırı Değer Tespiti ve İşlenmesi
# Adım 1: İşlenmiş veri setini oku, aykırı değerleri tespit et ve işle
#
# Not: Aykırı değer tespiti için IQR yöntemi kullanılacaktır.
# Gerekçe: IQR, veri dağılımının normal olup olmamasına bakmaksızın sağlam bir yöntemdir
# ve aykırı değerlere karşı daha az duyarlıdır.
# Aykırı değerler, sınır değerlerle (Q1 - 1.5*IQR ve Q3 + 1.5*IQR) değiştirilecektir.
# Gerekçe: Sınır değerlerle değiştirme, veri kaybını önler ve aykırı değerlerin model
# üzerindeki etkisini azaltır.
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Dosya yolları
input_path = '../data/02_processed_data.xlsx'
output_path = '../data/03_outlier_processed.xlsx'
plot_path = '../data/outlier_plot.png'

# [1C.a] Veri setini oku
print("[1C.a] İşlenmiş veri seti okunuyor...")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Veri seti dosyası bulunamadı: {input_path}")
df = pd.read_excel(input_path)

# [1C.b] Veri setinin boyutu
print(f"[1C.b] Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# [1C.c] Sayısal sütunları seç (Class hariç)
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(f"[1C.c] Aykırı değer analizi için sayısal sütunlar: {list(numeric_columns)}")

# [1C.d] IQR yöntemiyle aykırı değerleri tespit et ve raporla
outlier_counts = {}
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_counts[col] = len(outliers)

print("[1C.d] Aykırı değer sayıları (başlangıç):")
if any(outlier_counts.values()):
    for col, count in outlier_counts.items():
        if count > 0:
            print(f"{col}: {count}")
    print(f"Toplam aykırı değer: {sum(outlier_counts.values())}")
else:
    print("Hiç aykırı değer bulunamadı.")

# [1C.e] Aykırı değer oranlarını görselleştir
if any(outlier_counts.values()):
    print("[1C.e] Aykırı değer oranları görselleştiriliyor...")
    outlier_ratios = {col: (count / len(df)) * 100 for col, count in outlier_counts.items() if count > 0}
    plt.figure(figsize=(10, 6))
    plt.bar(outlier_ratios.keys(), outlier_ratios.values(), color='salmon')
    plt.xlabel('Sütun Adları')
    plt.ylabel('Aykırı Değer Oranı (%)')
    plt.title('Aykırı Değer Oranları')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[1C.e] Aykırı değer grafiği kaydedildi: {plot_path}")
else:
    print("[1C.e] Görselleştirme yapılmadı, çünkü aykırı değer yok.")

# [1C.f] Aykırı değerleri sınır değerlerle değiştir
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    print(f"[1C.f] '{col}' sütunundaki aykırı değerler sınır değerlerle değiştirildi.")

# [1C.g] Son aykırı değer kontrolü
outlier_counts_after = {}
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_counts_after[col] = len(outliers)

print("[1C.g] Aykırı değer sayıları (son):")
if any(outlier_counts_after.values()):
    for col, count in outlier_counts_after.items():
        if count > 0:
            print(f"{col}: {count}")
    print(f"Toplam aykırı değer: {sum(outlier_counts_after.values())}")
else:
    print("Hiç aykırı değer bulunamadı.")

# ================================
# Adım 2: Kaydet
# ================================

# "../data" dizini yoksa oluştur
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# [1C.h] İşlenmiş veri setini kaydet
print(f"[1C.h] Aykırı değer işlenmiş veri seti kaydediliyor: {output_path}")
df.to_excel(output_path, index=False)
if os.path.exists(output_path):
    print(f"[1C.h] Dosya başarıyla kaydedildi: {output_path}")
else:
    raise FileNotFoundError(f"Dosya kaydedilemedi: {output_path}")

print("\n[1C] Aykırı değer tespiti ve işleme adımı tamamlandı.")