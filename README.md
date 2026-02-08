# End-to-End Medical Insurance Cost Prediction

This project is a comprehensive Machine Learning solution designed to predict individual medical insurance costs based on personal attributes. The project covers the entire lifecycle, from in-depth Exploratory Data Analysis (EDA) to building robust regression models and deploying them as a production-ready API using **FastAPI**.

## Project Milestones

### 1. Exploratory Data Analysis (EDA)
- **Data Cleaning**: Handled missing values and removed duplicate entries.
- **Outlier Detection**: Analyzed extreme values using the Interquartile Range (IQR) method.
- **Visualizations**: Performed distribution and correlation analysis using Seaborn and Matplotlib.
- **Target Transformation**: Applied **Log Transformation** ($log1p$) to the `charges` column to normalize skewed data.

### 2. Feature Engineering & Pipeline
Implemented `ColumnTransformer` and `Pipeline` structures for modularity and reproducibility:
- **Numerical**: `StandardScaler` for `age` and `bmi`, and `MinMaxScaler` for `children`.
- **Categorical**: `OneHotEncoder` for `sex`, `smoker`, and `region`.

### 3. Model Comparison & Evaluation
Evaluated three different regression models using 5-Fold **Cross-Validation** (R2 Score):
- **Linear Regression**: Baseline model.
- **Random Forest**: Ensemble learning approach.
- **XGBoost**: Highest performing model (Selected Model).

## 🛠 Tech Stack
- **Languages:** Python 3.12
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **API Framework:** FastAPI, Uvicorn, Pydantic
- **Model Storage:** Joblib
