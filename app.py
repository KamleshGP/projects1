import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('punkt')

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

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


st.title("Email Spam Detection")
st.write("### Enter the message :-")
input_sms = st.text_area("")

if st.button("Predict"):
    # 1.preprocess
    transformed_sms = trans_text(input_sms)

    # 2.vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3.predict
    result = model.predict(vector_input)[0]

    # 4.display
    if result == 1:

        st.subheader("Entered message is :- Spam")
    else:
        st.subheader("Entered message is :- Not Spam")
