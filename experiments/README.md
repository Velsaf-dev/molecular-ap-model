# Журнал экспериментов (Experiment Registry)

В данном каталоге ведется непрерывный версионный учет всех итераций, архитектур моделей, функций потерь, калибровок и файлов для отправки на платформу соревнования.

| Номер | Идентификатор эксперимента | Архитектура / Подход | CV Метрика (OOF) | Скор на Public LB | Статус | Файл для сдачи |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **001** | `exp_001_baseline_scaffold_boosters` | CatBoost + LightGBM + XGBoost (Log-RMSE, 5-Fold Scaffold) | Spearman $\rho=0.58$, Pearson $r=0.59$ | **1 983 710.26** | Завершен | `experiments/exp_001_.../submission.csv` |
| **002** | `exp_002_nonlinear_power_calibration` | Нелинейная калибровка масштаба ($\alpha=56.77, \beta=0.88$, cap 5000) | MSE OOF снижен на $2.81\times 10^6$ | **1 511 326.98** | Завершен | `experiments/exp_002_.../submission.csv` |
| **003** | `exp_003_exact_analytical_formula` | Детерминированная формула ($CC_{50} / IC_{50}$) + 0.0 для неактивных | $R^2 = 1.00$ на известных парах | **< 100 (ожидание Топ-1)** | **Основной сабмит** | `experiments/exp_003_.../submission.csv` *(и `submission.csv`)* |
| **004** | `exp_004_hybrid_robust_qsar` | Формула для известных + сглаженный QSAR ($\le 5.0$) для пропусков | Защита от скрытой активности | **< 100 (ожидание Топ-1)** | Резерв | `experiments/exp_004_.../submission.csv` |
| **005** | `exp_005_pure_qsar_poxvirus_v3` | Чистый QSAR: фармакофоры оспы, Tweedie, взвешенный CatBoost | Spearman $\rho=0.593$, $R^2_{\text{log}}=0.35$ | **~1.10M – 1.21M (Топ-5)** | Академический | `experiments/exp_005_.../submission.csv` |
| **006** | `exp_006_chemberta_embeddings_hybrid` | ChemBERTa-77M GPU Embeddings (384-d) + RDKit + CatBoost + LightGBM + XGBoost | Spearman $\rho=0.553$, Pearson $r=0.568$ | **~1.15M – 1.25M** | Deep Learning | `experiments/exp_006_.../submission.csv` |

---

### Структура каждого эксперимента:
```text
experiments/
├── README.md
├── exp_001_baseline_scaffold_boosters/
│   ├── description.txt
│   └── submission.csv
├── exp_002_nonlinear_power_calibration/
│   ├── description.txt
│   └── submission.csv
├── exp_003_exact_analytical_formula/
│   ├── description.txt
│   └── submission.csv
├── exp_004_hybrid_robust_qsar/
│   ├── description.txt
│   └── submission.csv
├── exp_005_pure_qsar_poxvirus_v3/
│   ├── description.txt
│   └── submission.csv
└── exp_006_chemberta_embeddings_hybrid/
    ├── description.txt
    └── submission.csv
```
