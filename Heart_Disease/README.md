# Heart Disease Prediction Project

This project aims to predict the likelihood of heart disease in patients based on medical attributes using both classical and ensemble machine learning models.  
It is built following a full data science pipeline — from data exploration to model optimization and evaluation.

## Dataset Description

**Source:** Heart Disease dataset  
This database contains **13 attributes** and **1 target variable**. It includes **8 nominal** and **5 numeric** features.

| Feature | Description | Type |
|----------|--------------|------|
| `age` | Patient's age in years | Numeric |
| `sex` | Gender (1 = Male, 0 = Female) | Nominal |
| `cp` | Type of chest pain experienced (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic) | Nominal |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | Nominal |
| `restecg` | Resting electrocardiographic results (0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy) | Nominal |
| `thalach` | Maximum heart rate achieved | Numeric |
| `exang` | Exercise-induced angina (1 = Yes, 0 = No) | Nominal |
| `oldpeak` | ST depression induced by exercise relative to rest | Numeric |
| `slope` | Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping) | Nominal |
| `ca` | Number of major vessels (0–3) | Nominal |
| `thal` | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect, 0 = null) | Nominal |
| `target` | Target variable (1 = heart disease, 0 = healthy) | Target |

## Project Structure 
├── app/                         # Streamlit web application
│   ├── app.py                   
│
├── data/                        # Data storage and processing
│   ├── raw/                     # Original dataset (raw CSV files) and feature selection with Decision Tree
│   ├── fe/                      # Cleaned and feature-engineered datasets
│
├── figures/                    
│
├── notebooks/                   
│   ├── 1_data_analysis.ipynb          # Data Analysis
│   ├── 2_feature_engineering.ipynb    # Data preprocessing & feature engineering
│   ├── 3_normal_model.ipynb           # Base ML models
│   └── 4_ensemble_models.ipynb        # Advanced ensemble models
│
├── requirements.txt             # List of Python libraries
└── README.md                    # Project documentation

## Results
The results of the models can be found in the **figures** folder.
The performance of the models is measured by **Accuracy**.
The highest accuracy achieved in this project is **97%**.
Overall, applying proper data preprocessing helps machine learning methods improve their accuracy considerably.

## Web App Demo
You can interact with the trained model through a simple Streamlit web application included in the app/ folder.

```bash
streamlit run app/app.py
```

## Requirements

All required Python packages are listed in `requirements.txt`.  
To install them, run:

```bash
pip install -r requirements.txt
```

