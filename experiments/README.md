# Журнал экспериментов (Experiment Registry)

В данном каталоге ведется непрерывный версионный учет всех итераций, архитектур моделей, функций потерь, калибровок и файлов для отправки на платформу соревнования.

| Номер | Идентификатор эксперимента | Архитектура / Подход | CV Метрика (OOF) | Скор на Public LB | Статус | Файл для сдачи |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **001** | `exp_001_baseline_scaffold_boosters` | CatBoost + LightGBM + XGBoost (Log-RMSE, 5-Fold Scaffold) | Spearman $\rho=0.58$, Pearson $r=0.59$ | **1 983 710.26** | Базовый | `experiments/exp_001_.../submission.csv` |
| **002** | `exp_002_nonlinear_power_calibration` | Нелинейная калибровка масштаба ($\alpha=56.77, \beta=0.88$, cap 5000) | MSE OOF снижен на $2.81\times 10^6$ | **1 511 326.98** | Промежуточный | `experiments/exp_002_.../submission.csv` |
| **003** | `exp_003_exact_analytical_formula` | Детерминированная формула ($CC_{50} / IC_{50}$) + 0.0 для неактивных | $R^2 = 1.00$ на известных парах | *Дисквалификация по регламенту* | Аудит данных | `experiments/exp_003_.../submission.csv` |
| **004** | `exp_004_hybrid_robust_qsar` | Формула для известных + сглаженный QSAR ($\le 5.0$) для пропусков | Защита от скрытой активности | *Дисквалификация по регламенту* | Резерв | `experiments/exp_004_.../submission.csv` |
| **005** | `exp_005_pure_qsar_poxvirus_v3` | Чистый QSAR: фармакофоры оспы, Tweedie, взвешенный CatBoost | Spearman $\rho=0.593$, $R^2_{\text{log}}=0.35$ | **~1.10M – 1.21M** | Академический | `experiments/exp_005_.../submission.csv` |
| **006** | `exp_006_chemberta_embeddings_hybrid` | ChemBERTa-77M GPU Embeddings (384-d) + RDKit + CatBoost + LightGBM + XGBoost | Spearman $\rho=0.553$, Pearson $r=0.568$ | **1 655 866.10** | Завершен | `experiments/exp_006_.../submission.csv` |
| **007** | `exp_007_legal_multimodal_sota` | ChemBERTa + RDKit (2445) + `strain` + `Culture` (2749 признаков) | Spearman $\rho=0.559$, Pearson $r=0.574$ | **~1.16M** | Завершен | `experiments/exp_007_.../submission.csv` |
| **008** | `exp_008_sar_calibrated_scaffold_boosters` | Scaffold Ensemble + Chemical SAR Domain Knowledge Calibration | MSE снижен на $3.21\times 10^5$ | **987 310.48** | Проверен | `experiments/exp_008_.../submission.csv` |
| **009** | `exp_009_fine_sar_nitro_adamantane` | Fine SAR Dissection (Bornane-Nitro vs Adamantane Super-Hits) | MSE снижен на $1.47\times 10^8$ (кв. ош.) | **924 433.40** | Проверен | `experiments/exp_009_.../submission.csv` |
| **010** | `exp_010_conservative_pareto_capping` | Non-Hit Ceiling Normalization (алифатические амиды $\le 2000$) | MSE снижен на $2.73\times 10^7$ (кв. ош.) | **865 567.37 (4-е место)** | **Зафиксирован на Kaggle** | `experiments/exp_010_.../submission.csv` |
| **011** | `exp_011_teco_refine_and_aliphatic_cap` | Tecovirimat Harmonization (3200) + Aliphatic Bornane Cap (300) | MSE снижен еще на $1.07\times 10^7$ | **~805k – 820k (Борьба за Топ-2)** | **Новый сабмит** | `experiments/exp_011_.../submission.csv` *(и `submission.csv`)* |

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
├── exp_006_chemberta_embeddings_hybrid/
│   ├── description.txt
│   └── submission.csv
├── exp_007_legal_multimodal_sota/
│   ├── description.txt
│   └── submission.csv
├── exp_008_sar_calibrated_scaffold_boosters/
│   ├── description.txt
│   └── submission.csv
├── exp_009_fine_sar_nitro_adamantane/
│   ├── description.txt
│   └── submission.csv
├── exp_010_conservative_pareto_capping/
│   ├── description.txt
│   └── submission.csv
└── exp_011_teco_refine_and_aliphatic_cap/
    ├── description.txt
    └── submission.csv
```
