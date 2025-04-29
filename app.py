import streamlit as st
import pickle
import nltk
import string
import os
import shutil
from nltk.stem.porter import PorterStemmer

# Step 1: Force re-download of 'punkt' to fix potential 'punkt_tab' issue
punkt_path = os.path.expanduser('~/nltk_data/tokenizers/punkt')
if os.path.exists(punkt_path):
    shutil.rmtree(punkt_path)
nltk.download('punkt', force=True)

# Step 2: Initialize stemmer
ps = PorterStemmer()

# Step 3: Safely load vectorizer and model
try:
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Step 4: Preprocessing function
def trans_text(Text):
    Text = Text.lower()
    Text = nltk.word_tokenize(Text)

    y = []
    for i in Text:
        if i.isalnum():
            y.append(i)

    Text = y[:]
    y.clear()

    for i in Text:
        if i not in string.punctuation:
            y.append(i)

    Text = y[:]
    y.clear()

    for i in Text:
        y.append(ps.stem(i))

    return " ".join(y)

# Step 5: Streamlit UI
st.title("Email Spam Detection")
st.write("### Enter the message :-")
input_sms = st.text_area("")

if st.button("Predict"):
    # Step 6: Validate empty input
    if not input_sms.strip():
        st.warning("Please enter a message before prediction.")
        st.stop()

    # Step 7: Preprocess
    transformed_sms = trans_text(input_sms)

    # Step 8: Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Step 9: Predict
    result = model.predict(vector_input)[0]

    # Step 10: Display Result
    if result == 1:
        st.subheader("Entered message is :- Spam")
    else:
        st.subheader("Entered message is :- Not Spam")
