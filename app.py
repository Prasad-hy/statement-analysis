import streamlit as st
import pickle

st.title("Text Classification with Multiple Models")

# Let user upload the text
user_input = st.text_area("Enter text to classify:")

# Load all model pickle files
model_files = [
    "Logistic_Regression_with_vectorizer.pkl",
    "SVM_with_vectorizer.pkl",
    "Random_Forest_with_vectorizer.pkl",
    "Decision_Tree_with_vectorizer.pkl",
    "KNN_with_vectorizer.pkl",
    "Naive_Bayes_with_vectorizer.pkl",
    "Gradient_Boosting_with_vectorizer.pkl"
]

# Dictionary to store loaded models
models = {}
for file in model_files:
    model_name = file.replace("_with_vectorizer.pkl", "").replace("_", " ")
    with open(file, 'rb') as f:
        models[model_name] = pickle.load(f)

# Select model
model_choice = st.selectbox("Choose Model", list(models.keys()))

# Predict button
if st.button("Predict"):

    if not user_input:
        st.warning("Please enter some text!")
    else:
        # Get selected model and its vectorizer
        data = models[model_choice]
        model = data['model']
        vectorizer = data['vectorizer']

        # Transform input and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        # Try predict_proba if available
        try:
            prediction_proba = model.predict_proba(input_vector)
            st.write(f"Prediction Probability: {prediction_proba[0]}")
        except:
            st.write("Probability prediction not supported for this model.")

        st.success(f"Predicted Class: {prediction[0]}")
