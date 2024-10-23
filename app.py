import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Hugging Face model and tokenizer names
MODEL_NAME = "Asif123QWE/lora_model"  # Replace with your actual model path on Hugging Face

# Load model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

# Function to generate response from the model
def generate_answer(question, tokenizer, model):
    inputs = tokenizer(question, return_tensors="pt")
    output = model.generate(inputs["input_ids"], max_length=200)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Load the model and tokenizer
tokenizer, model = load_model()

# Streamlit app UI
st.title("Domestic Violence Q&A Chatbot")
st.write("This chatbot is designed to answer questions related to domestic violence based on a fine-tuned Llama model.")

# Input field for user's question
user_question = st.text_input("Ask a question related to domestic violence:")

# Generate answer when the user asks a question
if user_question:
    with st.spinner("Generating answer..."):
        answer = generate_answer(user_question, tokenizer, model)
        st.write("### Answer:")
        st.write(answer)

# Add an info section about the purpose of the chatbot
st.sidebar.title("About")
st.sidebar.write("This chatbot is fine-tuned to provide information and guidance on domestic violence topics.")
st.sidebar.write("Please note that this chatbot is designed to assist, but for emergencies or detailed advice, contact appropriate authorities.")
