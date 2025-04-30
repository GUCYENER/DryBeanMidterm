# 01_data_loading_and_missing_values.py

# ================================
# BÖLÜM 1A - Veri Ön İşleme (Preprocessing)
# Adım 1: Veri setini oku ve eksik verileri ekle
# ================================

import pandas as pd
import numpy as np
import os

# Dosya yolları
file_path = '../data/Dry_Bean_Dataset.xlsx'
output_path = '../data/01_raw_with_missing.xlsx'

# [1A.a] Veri setini oku
print("[1A.a] Veri seti okunuyor...")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Veri seti dosyası bulunamadı: {file_path}")
df = pd.read_excel(file_path)

# [1A.b] Veri setinin orijinal boyutu
print(f"[1A.b] Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# [1A.c] %5 oranında eksik veri ekleyeceğimiz kolonlar: 'MajorAxisLength', 'MinorAxisLength'
np.random.seed(42)
columns_5_percent = ['MajorAxisLength', 'MinorAxisLength']
for col in columns_5_percent:
    if col not in df.columns:
        raise ValueError(f"Sütun bulunamadı: {col}")
    missing_indices = df.sample(frac=0.05).index
    df.loc[missing_indices, col] = np.nan
    print(f"[1A.c] '{col}' sütununa %5 eksik veri eklendi ({len(missing_indices)} satır).")

# [1A.d] %35 eksik veri eklenecek kolon: 'ShapeFactor2'
if 'ShapeFactor2' not in df.columns:
    raise ValueError("Sütun bulunamadı: ShapeFactor2")
missing_indices_shape = df.sample(frac=0.35).index
df.loc[missing_indices_shape, 'ShapeFactor2'] = np.nan
print(f"[1A.d] 'ShapeFactor2' sütununa %35 eksik veri eklendi ({len(missing_indices_shape)} satır).")

# [1A.e] Eksik veri kontrolü
total_missing = df.isnull().sum()
missing_columns = total_missing[total_missing > 0]
print("[1A.e] Eksik veri sayıları (yalnızca eksik içeren sütunlar):")
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

# [1A.f] Eksik veri eklenmiş hali kaydediliyor
print(f"[1A.f] Eksik verilerle birlikte yeni veri dosyası kaydediliyor: {output_path}")
df.to_excel(output_path, index=False)
if os.path.exists(output_path):
    print(f"[1A.f] Dosya başarıyla kaydedildi: {output_path}")
else:
    raise FileNotFoundError(f"Dosya kaydedilemedi: {output_path}")

print("\n[1A] Veri yükleme ve eksik veri ekleme adımı tamamlandı.")