import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model & tokenizer
model = tf.keras.models.load_model("sentiment_rnn_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 250

st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below and find out if it's **Positive** 🌟 or **Negative** 💔")

review = st.text_area("✍️ Enter your review:")

if st.button("Predict Sentiment"):
    if review.strip():
        seq = tokenizer.texts_to_sequences([review])
        pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        pred = model.predict(pad)
        sentiment = "🌟 Positive Review!" if pred[0][0] > 0.5 else "💔 Negative Review!"
        st.success(sentiment)
    else:
        st.warning("Please enter a review before predicting.")
