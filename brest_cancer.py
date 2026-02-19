import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = pd.read_csv('Patient_data_brest.csv')

# Assuming 'Outcome' is the label column
X = data.drop('Outcome', axis=1)
y = data['Outcome']
feature_names = X.columns
Outcome_names = ['Malignant', 'Benign']  # Adjust if your labels differ

# Train-test split and scale
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection
model_choice = st.sidebar.selectbox("Choose model", ["Random Forest", "KNN"])
if model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = KNeighborsClassifier(n_neighbors=5)

# Fit model
model.fit(X_train_scaled, y_train)

# Streamlit UI
st.title("🩺 Breast Cancer Prediction App")
st.write("Use the sliders below to input tumor features.")

# Input sliders
input_data = []
for col in feature_names:
    val = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    input_data.append(val)

# Predict
input_np = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_np)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

# Output
# Output with color
st.subheader("🔍 Prediction")

if prediction == 0:
    st.markdown(f"<h3 style='color:red;'>⚠️ Result: {Outcome_names[prediction]} (Danger)</h3>", unsafe_allow_html=True)
else:
    st.markdown(f"<h3 style='color:green;'>✅ Result: {Outcome_names[prediction]} (Safe)</h3>", unsafe_allow_html=True)

st.write(f"**Confidence:** {np.max(probability) * 100:.2f}%")
