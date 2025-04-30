# 02_missing_data_processing.py

# ================================
# BÖLÜM 1B - Eksik Veri İşleme
# Adım 1: Eksik veri içeren veri setini oku, eksik verileri işle ve görselleştir
#
# Not: ShapeFactor2 sütunundaki %35 eksik veri, sütun bazlı silinecektir.
# Gerekçe: %35 eksik veri (yaklaşık 4,764 satır), satır bazlı silme ile veri setinin
# yaklaşık üçte birinin kaybına yol açar. ShapeFactor2, diğer şekil faktörleriyle
# kısmen yedeklenebildiğinden, sütun bazlı silme model performansını daha az etkiler.
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# Dosya yolları
input_path = '../data/01_raw_with_missing.xlsx'
output_path = '../data/02_processed_data.xlsx'
plot_path = '../data/missing_data_plot.png'

# [1B.a] Veri setini oku
print("[1B.a] Eksik veri içeren veri seti okunuyor...")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Veri seti dosyası bulunamadı: {input_path}")
df = pd.read_excel(input_path)

# [1B.b] Veri setinin boyutu
print(f"[1B.b] Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# [1B.c] Başlangıç eksik veri kontrolü
total_missing = df.isnull().sum()
missing_columns = total_missing[total_missing > 0]
print("[1B.c] Eksik veri sayıları (başlangıç, yalnızca eksik içeren sütunlar):")
if len(missing_columns) > 0:
    print(missing_columns)
    print(f"Toplam eksik veri: {missing_columns.sum()}")
else:
    print("Hiç eksik veri yok.")

# [1B.d] Eksik veri oranlarını görselleştir
if len(missing_columns) > 0:
    print("[1B.d] Eksik veri oranları görselleştiriliyor...")
    missing_ratios = (missing_columns / len(df)) * 100  # Eksik veri oranı (%)
    plt.figure(figsize=(8, 6))
    plt.bar(missing_ratios.index, missing_ratios, color='skyblue')
    plt.xlabel('Sütun Adları')
    plt.ylabel('Eksik Veri Oranı (%)')
    plt.title('Başlangıçtaki Eksik Veri Oranları')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[1B.d] Eksik veri grafiği kaydedildi: {plot_path}")
else:
    print("[1B.d] Görselleştirme yapılmadı, çünkü eksik veri yok.")

# [1B.e] %5 eksik veri içeren sütunları ortalama ile doldur
columns_5_percent = ['MajorAxisLength', 'MinorAxisLength']
for col in columns_5_percent:
    if col not in df.columns:
        raise ValueError(f"Sütun bulunamadı: {col}")
    if df[col].isnull().sum() > 0:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"[1B.e] '{col}' sütunundaki eksik veriler ortalama ({mean_value:.4f}) ile dolduruldu.")
    else:
        print(f"[1B.e] '{col}' sütununda eksik veri bulunamadı.")

# [1B.f] %35 eksik veri içeren ShapeFactor2 sütununu sil
if 'ShapeFactor2' not in df.columns:
    raise ValueError("Sütun bulunamadı: ShapeFactor2")
if df['ShapeFactor2'].isnull().sum() > 0:
    df.drop(columns=['ShapeFactor2'], inplace=True)
    print("[1B.f] 'ShapeFactor2' sütunu %35 eksik veri nedeniyle silindi.")
else:
    print("[1B.f] 'ShapeFactor2' sütununda eksik veri bulunamadı, silme yapılmadı.")

# [1B.g] Son eksik veri kontrolü
total_missing = df.isnull().sum()
missing_columns = total_missing[total_missing > 0]
print("[1B.g] Eksik veri sayıları (son, yalnızca eksik içeren sütunlar):")
if len(missing_columns) > 0:
    print(missing_columns)
    print(f"Toplam eksik veri: {missing_columns.sum()}")
else:
    print("Hiç eksik veri yok.")

# ================================
# Adım 2: Kaydet
# ================================

# "../data" dizini yoksa oluştur
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# [1B.h] İşlenmiş veri setini kaydet
print(f"[1B.h] İşlenmiş veri seti kaydediliyor: {output_path}")
df.to_excel(output_path, index=False)
if os.path.exists(output_path):
    print(f"[1B.h] Dosya başarıyla kaydedildi: {output_path}")
else:
    raise FileNotFoundError(f"Dosya kaydedilemedi: {output_path}")

print("\n[1B] Eksik veri işleme adımı tamamlandı.")