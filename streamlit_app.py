import streamlit as st
from transformers import pipeline

# Title of the app
st.title("Sports Text Classification Demo")

# Cache the classifier so it loads only once
@st.cache_resource
def load_classifier():
    return pipeline(task="text-classification", model="petrmiculek/sports_classification")

classifier = load_classifier()

# Text input area for the user
user_text = st.text_area("Enter text to classify:", "")

# When the user clicks the button, classify the input text
if st.button("Classify"):
    if user_text.strip():
        result = classifier(user_text)[0]
        st.write("Classification Result:")
        st.json(result)
    else:
        st.error("Please enter some text for classification.")
