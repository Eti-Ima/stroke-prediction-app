import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Lambda

# Load your trained models
models = {
    'Random Forest Classifier': "grid_search_rf.pkl",
    "KNeighbors Classifier": "grid_search_knn.pkl",
    "Logistic Regression": "grid_search_lr.pkl",
    "XGBoost Classifier": "grid_search_xgb.pkl",
    "Convolutional Neural Network": "cnn.h5"
}
label_encoder_gender = joblib.load('label_encoder_gender.joblib')
label_encoder_work_type = joblib.load('label_encoder_work_type.joblib')
label_encoder_ever_married = joblib.load('label_encoder_ever_married.joblib')
label_encoder_smoking_status = joblib.load('label_encoder_smoking_status.joblib')
scaler = joblib.load('scaler.joblib')

# Load models from files
loaded_models = {name: joblib.load(filename) for name, filename in models.items() if name != "Convolutional Neural Network"}

# Function to handle any custom objects in the Lambda layer
def conditional_max_pooling(x):
    if x.shape[1] >= 2:
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.5)(x)
    return x

custom_objects = {
    'MaxPooling1D': MaxPooling1D,
    'Dropout': Dropout,
    'conditional_max_pooling': Lambda(conditional_max_pooling)
}

# Load CNN model
cnn_model = load_model("cnn.h5", custom_objects=custom_objects, compile=False, safe_mode=False)

# Set up Streamlit
st.title('Intelligent Stroke Predictor')

st.title('About')
st.info('''
### Intelligent Stroke Predictor

This app provides predictions on the likelihood of having a stroke based on user inputs. It leverages multiple machine learning models to offer accurate predictions and health recommendations.

**Key Features:**
- **User Input:** Easily input your health data and receive an instant analysis.
- **Risk Categorization:** Understand your risk level (Low, Medium, High) for stroke.
- **Health Recommendations:** Get personalized health tips to manage your health better.
- **Model Interpretability:** View feature importance to understand what factors influence your risk.

**How to Use:**
1. Enter your health information in the sidebar.
2. Choose a prediction model.
3. Click the "Predict" button to get your result and health recommendations.

**Disclaimer:**
This app is intended to provide supplementary information and is not a substitute for professional medical advice. Please consult a healthcare provider for any medical concerns or conditions.

Developed and powered by machine learning models to help you stay informed and healthy.
''')

# Streamlit app
st.sidebar.header("User Input Features")

def user_input_features():
    default_gender = "Male"
    default_age = 67
    default_hypertension = 0
    default_heart_disease = 0
    default_ever_married = "No"
    default_work_type = "Private"
    default_avg_glucose_level = 120.0
    default_bmi = 20.0
    default_smoking_status = "never smoked"
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"), index=("Male", "Female").index(default_gender))
    age = st.sidebar.slider("Age", 30, 100, default_age)
    hypertension = st.sidebar.selectbox("Hypertension", (0, 1), index=(0, 1).index(default_hypertension), help='Select 0 for no hypertension and 1 for hypertension')
    heart_disease = st.sidebar.selectbox("Heart Disease", (0, 1), index=(0, 1).index(default_heart_disease), help='Select 0 for no heart disease and 1 for heart disease')
    ever_married = st.sidebar.selectbox("Ever Married", ("Yes", "No"), index=("Yes", "No").index(default_ever_married))
    work_type = st.sidebar.selectbox("Work Type", ("Private", "Self-employed", "Govt_job"), 
                                     index=("Private", "Self-employed", "Govt_job").index(default_work_type))
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 10.0, 400.0, default_avg_glucose_level)
    bmi = st.sidebar.slider("BMI", 10.0, 100.0, default_bmi)
    smoking_status = st.sidebar.selectbox("Smoking Status", ("formerly smoked", "never smoked", "smokes"), 
                                          index=("formerly smoked", "never smoked", "smokes").index(default_smoking_status))
    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    features = pd.DataFrame(data, index=[0])
    return data

input_df = user_input_features()
features = pd.DataFrame(input_df, index=[0])

st.subheader("User Input Features")
st.write(features)

# Function to preprocess user input
df = features

df['gender'] = label_encoder_gender.transform(df['gender'])
df['work_type'] = label_encoder_work_type.transform(df['work_type'])
df['ever_married'] = label_encoder_ever_married.transform(df['ever_married'])
df['smoking_status'] = label_encoder_smoking_status.transform(df['smoking_status'])
numerical_features = ['age', 'avg_glucose_level', 'bmi']
df[numerical_features] = scaler.transform(df[numerical_features])

df['hypertension'] = df['hypertension']
df['heart_disease'] = df['heart_disease']
processed_df = df
rdf = df

# Model selection
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))

# Add a predict button
if st.sidebar.button('Predict'):
    if model_choice != "Convolutional Neural Network":
        model = loaded_models[model_choice]
        proba = model.predict_proba(processed_df)
        st.write(f"{model_choice} - Raw Probabilities: {proba}")
        prediction = proba[0][1]  # Probability of class 1 (Stroke)
        probabilities = proba[0]  # Probabilities for all classes
    else:
        cnn_scale = joblib.load('norm.joblib')
        processed_df = cnn_scale.transform(processed_df)
        if isinstance(processed_df, pd.DataFrame):
            processed_df = processed_df.values
        cnn_input = processed_df.reshape(processed_df.shape[0], processed_df.shape[1], 1)
        cnn_proba = cnn_model.predict(cnn_input)
        prediction = cnn_proba[0][0]
        probabilities = [1 - cnn_proba[0][0], cnn_proba[0][0]]

    # Risk categorization thresholds
    def categorize_risk(prob):
        if prob < 0.3:
            return "Low Risk"
        elif prob < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"

    # Health recommendations based on user input
    def health_recommendations(input_data):
        tips = []
        if input_data['hypertension'] == 1:
            tips.append("Manage your blood pressure through a healthy diet, regular exercise, and medication if prescribed.")
        if input_data['heart_disease'] == 1:
            tips.append("Monitor your heart health and consult with your doctor regularly.")
        if input_data['bmi'] > 25:
            tips.append("Maintain a healthy weight through balanced nutrition and physical activity.")
        if input_data['avg_glucose_level'] > 140:
            tips.append("Control your blood sugar levels with a proper diet and medication if necessary.")
        if input_data['smoking_status'] != "never smoked":
            tips.append("Consider quitting smoking to improve your overall health.")
        
        # Encouragement for those with normal inputs
        if input_data['hypertension'] == 0 and input_data['heart_disease'] == 0 and input_data['bmi'] <= 25 and input_data['avg_glucose_level'] <= 140 and input_data['smoking_status'] == "never smoked":
            tips.append("Great job! Continue with your healthy lifestyle and regular check-ups.")
            tips.append("Keep maintaining a balanced diet rich in fruits, vegetables, and whole grains.")
            tips.append("Stay active with regular physical exercise, at least 30 minutes a day.")
            tips.append("Ensure you get enough sleep and manage stress effectively.")
        
        return tips

    # Display predictions, probabilities, and recommendations
    st.subheader('Predictions, Probabilities, and Recommendations')
    risk_category = categorize_risk(prediction)
    st.write(f"{model_choice} - Probability of Stroke: {prediction:.2f} ({risk_category})")
    st.write(f"{model_choice} - Probability of No Stroke: {probabilities[0]:.2f}")
    
    if risk_category != "Low Risk" or True:  # Always show recommendations
        recommendations = health_recommendations(input_df)
        st.write("### Health Recommendations:")
        for recommendation in recommendations:
            st.write(f"- {recommendation}")

