import streamlit as st
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'

COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

NUMERIC = ['age','trestbps','chol','thalach','oldpeak']
CATEGORICAL = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

def find_best_model():
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values='?').dropna()
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)

    X = df[NUMERIC + CATEGORICAL]
    Y = df['target']

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipe, NUMERIC),
        ('cat', cat_pipe, CATEGORICAL)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {'classifier__max_depth': range(3, 11)}

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X, Y)

    return grid_search

model = find_best_model()

st.title("Heart Disease Prediction App")
st.write('Enter patient data in the sidebar. The model will predict the likelihood of heart disease.')

st.sidebar.info(f'Optimal model depth found: {model.best_params_["classifier__max_depth"]}')

with st.sidebar.form(key='input_form'):
    st.header("Patient Data Input")
    age = st.slider('Age', 20, 80, 50)
    sex = st.selectbox('Gender', [('Male', 1), ('Female', 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox('Chest Pain Type', [('Typical Angina', 1), ('Atypical Angina', 2), ('Non-anginal Pain', 3),
                                           ('Asymptomatic', 4)], format_func=lambda x: x[0])[1]
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [('True', 1), ('False', 0)], format_func=lambda x: x[0])[1]
    chol = st.slider('Serum Cholesterol (mg/dl)', 120, 570, 240)
    restecg = st.selectbox('Resting ECG Results', [('Normal', 0), ('ST-T Wave Abnormality', 1),
                                                   ('Left Ventricular Hypertrophy', 2)], format_func=lambda x: x[0])[1]
    thalach = st.slider('Maximum Heart Rate Achieved', 70, 210, 150)
    exang = st.selectbox('Exercise Induced Angina', [('Yes', 1), ('No', 0)], format_func=lambda x: x[0])[1]
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [('Upsloping', 1), ('Flat', 2), ('Downsloping', 3)],
                         format_func=lambda x: x[0])[1]
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 3, 0)
    thal = st.selectbox('Thalassemia', [('Normal', 3), ('Fixed Defect', 6), ('Reversible Defect', 7)],
                        format_func=lambda x: x[0])[1]
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    proba = model.predict_proba(input_data)[0]
    prob_disease = proba[1] * 100

    st.subheader("Prediction Result")
    if prob_disease > 50:
        st.error(f'**Disease Likely**: There is a {prob_disease:.2f}% chance of heart disease.')
    else:
        st.success(f'**No Disease Likely**: There is a {100 - prob_disease:.2f}% chance of no heart disease.')

st.subheader("Optimal Model Decision Tree Visualization")      
st.write("This chart shows the decision rules of the best decision tree found during model tuning.")

best_pipeline = model.best_estimator_
preprocessor = best_pipeline.named_steps['preprocessor']
classifier = best_pipeline.named_steps['classifier']

try:
    feature_names = preprocessor.get_feature_names_out()
    fig,ax = plt.subplots(figsize=(25,11))
    plot_tree(classifier, feature_names=feature_names, class_names=['No Disease', 'Disease'],
              filled=True, rounded=True, fontsize=10, ax=ax)

    st.pyplot(fig)

except Exception as e:
    st.warning(f"Could not retrieve feature names for visualization. Error: {e}")