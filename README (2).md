# Student Exam Score Prediction

Regression project (Kaggle-style): predict a student's `exam_score` from study habits, background and school details.

**Final model:** Ridge regression, **R² 0.704** and **MAE 4.09 marks** on a held-out 20% validation split.

## Data

- 7,000 training rows, 3,000 test rows, 10 features
- Numeric: `study_hours`, `sleep_hours`, `attendance_pct`, `previous_score`, `screen_time_hours`
- Categorical: `tutoring`, `parent_education`, `internet_access`, `extracurricular`, `school_type`
- About 3% missing in `sleep_hours`, `screen_time_hours` and `parent_education` (filled with median / most frequent)

## Approach

1. **EDA and statistical tests.** ANOVA (parent education, tutoring) and Welch t-tests (internet access, extracurricular, school type) all show a significant link with the score. `previous_score` is the strongest numeric predictor (correlation 0.64), followed by `study_hours` (0.39).
2. **Preprocessing.** Imputation, Yeo-Johnson and standard scaling on the numeric columns, one-hot encoding for binary categories, ordinal encoding for `parent_education` and `tutoring`.
3. **Feature engineering.** Tried a screen-to-study ratio, academic focus index, cognitive friction, engagement and efficiency features. None improved the score, so they were dropped.
4. **Model comparison.** Random forest, linear, Lasso, Ridge, gradient boosting, stacking, SVR and LightGBM, with hyperparameter search on the main ones.

## Results (validation split)

| Model | R² | MAE |
|---|---|---|
| Random forest | 0.654 | n/a |
| SVR (tuned) | 0.685 | 4.24 |
| LightGBM (tuned) | 0.688 | 4.21 |
| Gradient boosting | 0.698 | 4.13 |
| Stacking | 0.704 | 4.09 |
| Linear regression | 0.704 | 4.09 |
| **Ridge (final)** | **0.704** | **4.09** |

## Key findings

- A simple linear model matches or beats every complex model, so the effects are mostly additive.
- Passing the numeric columns in two forms (power-transformed and standardized) lifted linear R² from about 0.66 to 0.70, because it lets a linear model follow curved effects such as sleep.
- Roughly 30% of the variance is unexplained by the available columns, which points to random noise as the performance ceiling.

## How to run

1. Put `train.csv` and `test.csv` in the working directory (in Colab, `/content`).
2. Run the notebook top to bottom.
3. The last cells fit the final Ridge model on all training rows and write `submission.csv` (`id`, `pred`).

**Tools:** Python, pandas, scikit-learn, SciPy, seaborn, LightGBM, XGBoost.
