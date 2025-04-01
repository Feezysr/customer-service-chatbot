import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
from datetime import datetime

# Set up page
st.set_page_config(page_title="Customer Service Chatbot", layout="wide")
st.title("🤖 Customer Service Chatbot")

# Initialize session state
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'process_query' not in st.session_state:
    st.session_state.process_query = False

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("chatbot_model")
    model = DistilBertForSequenceClassification.from_pretrained("chatbot_model")
    return tokenizer, model

# Load label mapping
with open("label_mapping.json") as f:
    intent_labels = json.load(f)

tokenizer, model = load_model()

# Text cleaning function
def clean_text(text):
    return text.lower().strip()

# Prediction function
def predict_intent(text):
    try:
        start_time = datetime.now()
        text = clean_text(text)
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        intent = list(intent_labels.keys())[list(intent_labels.values()).index(pred_id)]
        confidence = torch.softmax(outputs.logits, dim=1)[0][pred_id].item()
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "processing_time_ms": round(processing_time, 2)
        }
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot classifies customer service queries using a fine-tuned DistilBERT model.
    
    **Supported Intents:**
    - Account issues
    - Password reset
    - Order inquiries
    - Billing questions
    - Technical support
    """)
    
    if st.button("Show All Intents"):
        st.write(list(intent_labels.keys()))

# Test cases section
expander = st.expander("Test Sample Queries")
with expander:
    test_cases = [
        "How do I reset my password?",
        "My account is locked",
        "Where can I find my order history?",
        "The website isn't working"
    ]
    
    cols = st.columns(2)
    for i, query in enumerate(test_cases):
        with cols[i%2]:
            if st.button(query):
                st.session_state.last_query = query
                st.session_state.process_query = True
                st.toast(f"Processing: {query}")

# Main chat interface
user_input = st.text_input(
    "Type your customer service question:",
    value=st.session_state.last_query,
    placeholder="e.g., How do I reset my password?"
)

# Process input if either:
# 1. User pressed Enter after typing OR
# 2. User clicked a test query button
if (user_input and user_input != st.session_state.last_query) or st.session_state.process_query:
    query_to_process = user_input if user_input != st.session_state.last_query else st.session_state.last_query
    
    with st.spinner("Analyzing your query..."):
        result = predict_intent(query_to_process)
    
    # Clear the processing flag
    if "process_query" in st.session_state:
        del st.session_state.process_query
    
    if result:
        st.success(f"**Detected Intent:** {result['intent']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
        with col2:
            st.metric("Processing Time", f"{result['processing_time_ms']}ms")
        
        st.markdown("---")
        st.subheader("Suggested Response:")
        
        if result['intent'] == "password_reset":
            st.write("You can reset your password by visiting our password reset page at...")
        elif result['intent'] == "account_locked":
            st.write("Please contact support@company.com to unlock your account.")
        else:
            st.write("I understand you're asking about account support. Our team will help you shortly.")