# 08_modeling_evaluation.py

# ================================
# BÖLÜM 3 - Modelleme ve Değerlendirme
# Adım 1: Nested Cross-Validation ile modelleme, performans metrikleri ve ROC eğrileri
#
# Not: Üç veri temsili (ham veri, PCA, LDA) için Nested CV (dış: 5 katmanlı, iç: 3 katmanlı)
# uygulanacak. Beş sınıflandırıcı (Logistic Regression, Decision Tree, Random Forest, XGBoost,
# Naive Bayes) için hiperparametre ayarlama yapılacak. Performans metrikleri (Accuracy, Precision,
# Recall, F1 Score) ve ROC eğrileri (OVA) raporlanacak.
#
# İyileştirmeler:
# 1. ROC için her modelin kendi en iyi y_test'ini kullanması sağlandı.
# 2. ROC plotlarında sınıf isimleri (SEKER, HOROZ, vb.) kullanıldı.
# 3. AUC skorları karşılaştırması için tablo ve çubuk grafik eklendi.
# 4. Çıktı dosya isimlerine zaman damgası eklendi, böylece önceki dosyalar korunacak.
# 5. Döngüde 'Novelsclf' yazım hatası düzeltildi ('clf' olarak değiştirildi).
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Zaman damgası oluştur
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Dosya yolları
data_paths = {
    "Raw": "../data/06_raw_preprocessed.xlsx",
    "PCA": "../data/06_pca_transformed.xlsx",
    "LDA": "../data/07_lda_transformed.xlsx"
}
output_dir = "../data"
os.makedirs(output_dir, exist_ok=True)

# Sınıf isimleri mapping
class_mapping = {
    0: "BARBUNYA",
    1: "BOMBAY",
    2: "CALI",
    3: "DERMASON",
    4: "HOROZ",
    5: "SEKER",
    6: "SIRA"
}

# Sınıflandırıcılar ve hiperparametreler
classifiers = {
    "LogisticRegression": (LogisticRegression(max_iter=1000), {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    }),
    "DecisionTree": (DecisionTreeClassifier(), {
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10]
    }),
    "RandomForest": (RandomForestClassifier(), {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None]
    }),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"), {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.3]
    }),
    "NaiveBayes": (GaussianNB(), {})
}

# Performans metrikleri
metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
    "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
    "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted")
}

# [3A.a] Veri temsillerini yükle
print("[3A.a] Veri temsilleri yükleniyor...")
datasets = {}
for name, path in data_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")
    df = pd.read_excel(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    datasets[name] = (X, y)
    print(f"{name} veri seti: {X.shape[0]} satır, {X.shape[1]} sütun")

# [3A.b] Nested Cross-Validation
results = {}
roc_data = {}
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)

for data_name, (X, y) in datasets.items():
    print(f"\n[3A.b] {data_name} veri temsili için Nested CV başlıyor...")
    results[data_name] = {clf_name: {metric: [] for metric in metrics} for clf_name in classifiers}
    roc_data[data_name] = {clf_name: {} for clf_name in classifiers}
    
    # Sınıfları binarize et (OVA için)
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]
    
    # Her model için en iyi dış döngü sonuçlarını sakla
    best_outer_scores = {clf_name: -float("inf") for clf_name in classifiers}
    best_outer_probas = {clf_name: None for clf_name in classifiers}
    best_outer_y_test = {clf_name: None for clf_name in classifiers}
    
    for outer_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X)):
        print(f"Dış Döngü {outer_idx + 1}/5")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        for clf_name, (clf, param_grid) in classifiers.items():
            # İç döngüde hiperparametre ayarlama
            grid_search = GridSearchCV(clf, param_grid, cv=inner_kf, scoring="accuracy", n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_clf = grid_search.best_estimator_
            
            # Test seti üzerinde tahmin
            y_pred = best_clf.predict(X_test)
            if hasattr(best_clf, "predict_proba"):
                y_proba = best_clf.predict_proba(X_test)
            else:
                y_proba = np.zeros((len(y_test), n_classes))
            
            # Performans metriklerini hesapla
            for metric_name, metric_func in metrics.items():
                score = metric_func(y_test, y_pred)
                results[data_name][clf_name][metric_name].append(score)
            
            # En iyi dış döngü sonucunu güncelle
            acc = accuracy_score(y_test, y_pred)
            if acc > best_outer_scores[clf_name]:
                best_outer_scores[clf_name] = acc
                best_outer_probas[clf_name] = y_proba
                best_outer_y_test[clf_name] = y_test
    
    # ROC eğrileri için en iyi dış döngü sonuçlarını sakla
    for clf_name in classifiers:
        if best_outer_probas[clf_name] is not None:
            roc_data[data_name][clf_name]["y_test"] = best_outer_y_test[clf_name]
            roc_data[data_name][clf_name]["y_proba"] = best_outer_probas[clf_name]
            roc_data[data_name][clf_name]["n_classes"] = n_classes
            roc_data[data_name][clf_name]["auc_scores"] = []

# [3A.c] Performans metriklerini raporla
print("\n[3A.c] Performans metrikleri hesaplanıyor...")
metrics_summary = {}
for data_name in datasets:
    metrics_summary[data_name] = pd.DataFrame(
        index=[clf_name for clf_name in classifiers],
        columns=[f"{metric} (Mean ± Std)" for metric in metrics]
    )
    for clf_name in classifiers:
        for metric in metrics:
            scores = results[data_name][clf_name][metric]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            metrics_summary[data_name].loc[clf_name, f"{metric} (Mean ± Std)"] = f"{mean_score:.4f} ± {std_score:.4f}"

    # Performans metriklerini kaydet (zaman damgası eklendi)
    output_path = os.path.join(output_dir, f"metrics_{data_name.lower()}_{timestamp}.xlsx")
    metrics_summary[data_name].to_excel(output_path)
    print(f"{data_name} için metrikler kaydedildi: {output_path}")

# [3A.d] ROC eğrileri ve AUC skorları
print("\n[3A.d] ROC eğrileri ve AUC skorları hesaplanıyor...")
for data_name in datasets:
    for clf_name in classifiers:
        if "y_proba" not in roc_data[data_name][clf_name]:
            continue
        
        y_test = roc_data[data_name][clf_name]["y_test"]
        y_proba = roc_data[data_name][clf_name]["y_proba"]
        n_classes = roc_data[data_name][clf_name]["n_classes"]
        
        # ROC eğrisi ve AUC hesapla
        plt.figure(figsize=(10, 8))
        auc_scores = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            roc_data[data_name][clf_name]["auc_scores"].append(roc_auc)
            class_name = class_mapping.get(i, f"Class {i}")
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{data_name} - {clf_name} ROC Curves (OVA)')
        plt.legend(loc="best")
        plt.grid(True)
        plot_path = os.path.join(output_dir, f"roc_{data_name.lower()}_{clf_name.lower()}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"{data_name} - {clf_name} için ROC eğrisi kaydedildi: {plot_path}")
        
        # AUC skorlarını kaydet (zaman damgası eklendi)
        auc_df = pd.DataFrame({
            "Class": [class_mapping.get(i, f"Class {i}") for i in range(n_classes)],
            "AUC": auc_scores
        })
        auc_path = os.path.join(output_dir, f"auc_{data_name.lower()}_{clf_name.lower()}_{timestamp}.xlsx")
        auc_df.to_excel(auc_path, index=False)
        print(f"{data_name} - {clf_name} için AUC skorları kaydedildi: {auc_path}")

# [3A.e] AUC skorlarını karşılaştır
print("\n[3A.e] AUC skorları karşılaştırılıyor...")
for data_name in datasets:
    # Sınıf bazlı AUC skorlarını tabloya topla
    auc_comparison = pd.DataFrame(
        index=[class_mapping.get(i, f"Class {i}") for i in range(roc_data[data_name][list(classifiers.keys())[0]]["n_classes"])],
        columns=[clf_name for clf_name in classifiers if roc_data[data_name][clf_name]]
    )
    mean_auc_scores = {}
    
    for clf_name in classifiers:
        if "auc_scores" not in roc_data[data_name][clf_name]:
            continue
        auc_scores = roc_data[data_name][clf_name]["auc_scores"]
        for i, auc_score in enumerate(auc_scores):
            class_name = class_mapping.get(i, f"Class {i}")
            auc_comparison.loc[class_name, clf_name] = auc_score
        mean_auc_scores[clf_name] = np.mean(auc_scores) if auc_scores else 0

    # AUC karşılaştırma tablosunu kaydet (zaman damgası eklendi)
    auc_comparison_path = os.path.join(output_dir, f"auc_comparison_{data_name.lower()}_{timestamp}.xlsx")
    auc_comparison.to_excel(auc_comparison_path)
    print(f"{data_name} için AUC karşılaştırma tablosu kaydedildi: {auc_comparison_path}")
    
    # Ortalama AUC skorlarını çubuk grafikle görselleştir
    plt.figure(figsize=(10, 6))
    clf_names = list(mean_auc_scores.keys())
    mean_scores = list(mean_auc_scores.values())
    plt.bar(clf_names, mean_scores, color='skyblue')
    plt.xlabel('Classifier')
    plt.ylabel('Mean AUC Score')
    plt.title(f'{data_name} - Mean AUC Scores Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    comparison_plot_path = os.path.join(output_dir, f"auc_comparison_{data_name.lower()}_{timestamp}.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"{data_name} için AUC karşılaştırma grafiği kaydedildi: {comparison_plot_path}")

print("\n[3A] Modelleme ve değerlendirme adımı tamamlandı.")