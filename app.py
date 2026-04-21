#!/usr/bin/env python
# coding: utf-8

# # Streamlit Application Overview
# 
# A web-based application is developed using Streamlit to provide a user-friendly interface for detecting harmful online content in real time.
# 
# The application integrates the trained transformer-based model (MiniLM embeddings with SVM) and allows users to input text for analysis. The system processes the input, generates embeddings, and classifies the text as either Safe or Harmful.
# 
# ### Key Features
# 
# - **Real-time Text Analysis:** Users can enter any text and receive instant predictions.  
# - **Binary Classification:** Messages are classified as Safe or Harmful.  
# - **Automatic Alert System:** Harmful content triggers an alert message indicating potential risk.  
# - **Simulated Notification:** A mock email alert is generated to represent parental notification.  
# - **Case Logging:** All analyzed inputs are stored in a local file (`history.csv`) along with timestamps.  
# - **History Tracking:** Users can view previously analyzed messages and their results.
# 
# ### Workflow
# 
# User Input → Text Cleaning → Transformer Embedding (MiniLM) → SVM Prediction → Result Display → Alert & Logging
# 
# This application demonstrates the practical deployment of the trained model and highlights how machine learning can be used to build real-time systems for improving online safety.

# In[ ]:


import streamlit as st
from sentence_transformers import SentenceTransformer
import joblib
import re
import pandas as pd
import os
from datetime import datetime

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Cyberbullying Detection System",
    layout="wide"
)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    model_svm = joblib.load("svm_bert_model.pkl")
    return bert_model, model_svm

bert_model, model_svm = load_model()

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_text(text):
    text_clean = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    embedding = bert_model.encode([text_clean])
    pred = model_svm.predict(embedding)[0]
    return "Harmful" if pred == 1 else "Safe"

# ---------------------------
# FAKE EMAIL FUNCTION
# ---------------------------
def fake_email_alert(message):
    return f"""
Email Sent Successfully

To: parent@gmail.com
Subject: Cyberbullying Alert

Message:
Harmful content detected:

"{message}"

Please take necessary action.
"""

# ---------------------------
# HISTORY STORAGE WITH TIMESTAMP
# ---------------------------
history_file = "history.csv"

def save_history(text, result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_data = pd.DataFrame(
        [[timestamp, text, result]],
        columns=["Timestamp", "Text", "Result"]
    )

    if os.path.exists(history_file):
        old_data = pd.read_csv(history_file)
        updated = pd.concat([old_data, new_data], ignore_index=True)
    else:
        updated = new_data

    updated.to_csv(history_file, index=False)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Detection", "History"])

# ---------------------------
# HOME PAGE
# ---------------------------
if menu == "Home":
    st.image("cyber_images/A.png", use_container_width=True)

    st.title("Cyberbullying Detection System")

    st.markdown("""
This system uses machine learning to detect harmful content in text.

Features:
- Binary classification (Safe or Harmful)
- Real-time detection
- Automatic alert system
- Case history tracking with timestamps
""")

# ---------------------------
# DETECTION PAGE
# ---------------------------
elif menu == "Detection":

    st.title("Analyze Message")

    user_input = st.text_area("Enter text:", height=150)

    analyze_btn = st.button("Analyze")

    if analyze_btn:
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            result = predict_text(user_input)

            st.markdown("## Result")

            # SAVE HISTORY WITH TIMESTAMP
            save_history(user_input, result)

            # SAFE
            if result == "Safe":
                st.success("This message is Safe")
                st.image("cyber_images/SAFE.png")

            # HARMFUL
            else:
                st.error("Harmful content detected")
                st.image("cyber_images/D.png")

                # POPUP ALERT
                st.markdown("""
                <div style="background-color:#ff4b4b;
                            padding:15px;
                            border-radius:10px;
                            color:white;
                            font-weight:bold;
                            text-align:center;">
                Parent Alerted Successfully
                </div>
                """, unsafe_allow_html=True)

                # FAKE EMAIL
                email_msg = fake_email_alert(user_input)

                st.markdown("### Email Notification")
                st.code(email_msg)

            st.markdown("### Input Message")
            st.info(user_input)

# ---------------------------
# HISTORY PAGE
# ---------------------------
elif menu == "History":

    st.title("Reported Cases History")

    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        st.dataframe(df)
    else:
        st.info("No history available yet.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("AI Capstone Project")

