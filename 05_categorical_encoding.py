# 05_categorical_encoding.py

# ================================
# BÖLÜM 1E - Kategorik Verilerin Kodlanması
# Adım 1: İşlenmiş veri setini oku ve kategorik sütunları kodla
#
# Not: Class sütunu LabelEncoder ile numerik hale getirilecektir.
# Gerekçe: Class, hedef değişken olup, makine öğrenimi algoritmaları için sayısal
# değerler gerektirir. Veri setinde başka kategorik sütun bulunmamaktadır, bu nedenle
# OneHotEncoder veya pd.get_dummies() kullanımı gerekli değildir.
# ================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# Dosya yolları
input_path = '../data/04_scaled_data.xlsx'
output_path = '../data/05_encoded_data.xlsx'
plot_path = '../data/class_distribution_plot.png'

# [1E.a] Veri setini oku
print("[1E.a] Ölçeklenmiş veri seti okunuyor...")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Veri seti dosyası bulunamadı: {input_path}")
df = pd.read_excel(input_path)

# [1E.b] Veri setinin boyutu
print(f"[1E.b] Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# [1E.c] Kategorik sütunları kontrol et
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print(f"[1E.c] Kategorik sütunlar: {list(categorical_columns)}")

# [1E.d] Class sütununu LabelEncoder ile kodla
if 'Class' not in df.columns:
    raise ValueError("Class sütunu bulunamadı.")
print("[1E.d] Class sütunu kodlanıyor...")
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])

# [1E.e] Kodlama öncesi ve sonrası Class değerlerini raporla
print("[1E.e] Class sütunu kodlama bilgileri:")
print(f"Orijinal sınıflar: {list(le.classes_)}")
print(f"Kodlanmış sınıflar: {list(range(len(le.classes_)))}")
print(f"Kodlanmış Class sütunu örnek değerleri: {df['Class'].head().tolist()}")

# [1E.f] Kodlanmış sınıf dağılımını görselleştir
print("[1E.f] Kodlanmış sınıf dağılımı görselleştiriliyor...")
class_counts = df['Class'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values, color='lightgreen', edgecolor='black')
plt.xlabel('Kodlanmış Sınıf Etiketleri')
plt.ylabel('Frekans')
plt.title('Kodlanmış Sınıf Dağılımı')
plt.xticks(class_counts.index, [le.classes_[i] for i in class_counts.index], rotation=45)
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"[1E.f] Sınıf dağılımı grafiği kaydedildi: {plot_path}")

# ================================
# Adım 2: Kaydet
# ================================

# "../data" dizini yoksa oluştur
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# [1E.g] Kodlanmış veri setini kaydet
print(f"[1E.g] Kodlanmış veri seti kaydediliyor: {output_path}")
df.to_excel(output_path, index=False)
if os.path.exists(output_path):
    print(f"[1E.g] Dosya başarıyla kaydedildi: {output_path}")
else:
    raise FileNotFoundError(f"Dosya kaydedilemedi: {output_path}")

print("\n[1E] Kategorik veri kodlama adımı tamamlandı.")