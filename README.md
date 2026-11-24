# IF3170_G12_YuzukiAkatsukiMilikKita
<img width="1920" height="1080" alt="Add a heading (2)" src="https://github.com/user-attachments/assets/ea81caaf-3af0-4daf-a67e-ac9a56d471dc" />

Tugas Besar 2 IF3170 Inteligensi Artifisial ini memiliki tujuan memprediksi pembatalan pemesanan penginapan (`is_canceled`) dengan supervised learning. Lima algoritma (KNN, Decision Tree Learning, Logistic Regression, SVM, Naive Bayes) kami implementasikan *from scratch* dengan NumPy/pandas, lalu dibandingkan dengan scikit-learn. Metrik utama yang digunakan adalah macro F1.


## Struktur Repository

```
src/        kode program (algoritma, preprocessing, runner, tuning)
docs/       laporan (.pdf)
output/     model (.pkl), file submission, dan video bonus
data/       dataset Kaggle
notebook/   notebook pengerjaan 
```

## Setup

Membutuhkan Python 3.11+.

```bash
git clone https://github.com/rhenprojects/IF3170_G12_YuzukiAkatsukiMilikKita.git
cd IF3170_G12_YuzukiAkatsukiMilikKita

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` dari Kaggle di folder `data/`.

## Cara Menjalankan

Semua perintah dijalankan dari root repository. Setiap runner melakukan preprocessing, training, evaluasi, menyimpan model ke `output/`, memuat ulang model untuk verifikasi, dan menghasilkan file submission.

```bash
python src/runners/run_decision_tree.py          # DTL (C4.5 / entropy / gini)
python src/runners/run_logistic_regression.py    # Logistic Regression (L1, L2, Elastic Net, AdaGrad)
python src/runners/run_naive_bayes.py            # Naive Bayes
python src/runners/run_svm.py                    # SVM
python src/bonus_logistic_regression_video.py   # Video kontur log-loss (bonus)
```

Untuk menjalankan seluruh algoritma (termasuk KNN dan SVM) beserta perbandingan dengan scikit-learn, buka notebook:

```bash
jupyter notebook notebook/Notebook_IF3170_G12_YuzukiAkatsukiMilikKita.ipynb
```

Hasil dapat ditemukan di `output/`, misalnya `submission_*.csv`, `*.pkl`, dan `logistic_regression_contour.mp4`.

## Save dan Load Model

Model disimpan dan dimuat dengan `pickle` melalui `save_model(model, path)` dan `load_model(path)` di `src/model_io.py`.

## Pembagian Tugas

| NIM      | Nama                         | Pembagian         |
|----------|------------------------------|-------------------|
| 13524007 | Muhammad Ashkar              | Decision Tree     |
| 13524012 | Tengku Naufal Saqib          | SVM               |
| 13524039 | Rhenaldy Cahyadi Putra       | Naive Bayes       |
| 13524076 | Fayyaz Akmal Lauda           | Logistic Regression |
| 13524097 | Muhammad Faiz Alfada Dharma  | KNN               |

Made with 🤍 by
<img width="63" height="30" alt="newimagg" src="https://github.com/user-attachments/assets/c21b240a-20ba-45e1-aa09-0ba6f83e6e41" />

