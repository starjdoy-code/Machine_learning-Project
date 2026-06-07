# 🛒 Prediksi Penjualan Toko Ritel Rossman

**Kelompok 1 — LM01**  
Louis Huang | Gilbert Tjandra Adanarianto | Dava Rabbani Adrian Widyatmoko

---

## Deskripsi Proyek

Aplikasi web berbasis Machine Learning untuk memprediksi jumlah penjualan harian toko ritel berdasarkan:
- Status promosi (Promo, Promo2)
- Hari libur (StateHoliday, SchoolHoliday)
- Faktor operasional (StoreType, Assortment, DayOfWeek)
- Kondisi kompetitor (CompetitionDistance)
- Faktor waktu (Bulan, Hari, Minggu)

**Dataset:** [Rossman Store Sales — Kaggle](https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales)

---

## Struktur File

```
rossman_project/
├── train_model.ipynb      ← Notebook pelatihan model
├── app.py                 ← Aplikasi Streamlit
├── requirements.txt       ← Dependencies Python
├── README.md              ← Panduan ini
│
│   (dihasilkan setelah menjalankan notebook)
├── model_rossman.pkl      ← Model ML yang disimpan
├── model_features.pkl     ← Daftar fitur model
├── eval_data.pkl          ← Data evaluasi model
├── feature_importance.pkl ← Feature importance scores
│
│   (download dari Kaggle)
├── train.csv              ← Data training Rossman
└── store.csv              ← Data informasi toko
```

---

## Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download dari Kaggle dan letakkan `train.csv` dan `store.csv` di folder yang sama:
- https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales

### 3. Latih Model

Jalankan semua cell di `train_model.ipynb` (Jupyter Notebook atau Google Colab).

File berikut akan otomatis dihasilkan:
- `model_rossman.pkl`
- `model_features.pkl`
- `eval_data.pkl`
- `feature_importance.pkl`

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

---

## Model yang Digunakan

| Model              | Peran          |
|--------------------|----------------|
| Linear Regression  | Baseline       |
| Random Forest      | Model Utama    |
| XGBoost            | Model Pembanding|

Model terbaik (berdasarkan R²) akan di-*tuning* dengan `RandomizedSearchCV`.

## Target Evaluasi

| Metrik | Target |
|--------|--------|
| MAE    | < 15% dari rata-rata sales |
| RMSE   | < 15% dari rata-rata sales |
| R²     | > 0.85 |

---

## Deployment

Aplikasi dapat di-deploy ke **Streamlit Community Cloud** via GitHub:
1. Push semua file (termasuk `.pkl`) ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo dan set `app.py` sebagai entry point
