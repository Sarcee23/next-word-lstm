import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Streamlit Page Config (MUST be first Streamlit call) ---
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# --- Load model and tokenizer ---
@st.cache_resource
def load_lstm_model():
    return load_model("next_word_lstm.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as handle:
        return pickle.load(handle)

model = load_lstm_model()
tokenizer = load_tokenizer()


# --- Prediction Function ---
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None


# --- Streamlit UI ---
st.markdown(
    """
    <h1 style='text-align: center; color: #4F8BF9;'>Next Word Prediction using LSTM</h1>
    <p style='text-align: center; color: gray;'>Trained on Shakespeare's <b>Hamlet</b> • Predict the poetic next word!</p>
    <hr style='margin-top: -10px;'>
    """,
    unsafe_allow_html=True
)

st.subheader("Try it out here:")
example = "To be or not to"
user_input = st.text_input("Enter your sequence:", value=example, placeholder="Type the beginning of a phrase...")

if st.button( "Predict Next Word!"):
    max_sequence_len = model.input_shape[1] + 1
    with st.spinner("Processing..."):
        next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)

    if next_word:
        st.success(f"**Prediction:** {user_input} **{next_word}**")
    else:
        st.error("Sorry! Couldn’t predict the next word. Try a longer phrase.")

st.markdown(
    """
    <br><hr>
    <p style='text-align:center; color:gray; font-size: 0.9em;'>
    Built using <b>TensorFlow</b> and <b>Streamlit</b>  
    <br>by Subhayan — practicing Deep Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
