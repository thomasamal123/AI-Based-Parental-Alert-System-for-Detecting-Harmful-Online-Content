# AI-Based Parental Alert System for Detecting Harmful Online Content

## Overview

This project presents an AI-based system for detecting harmful online content using Natural Language Processing (NLP) and Machine Learning techniques. With the increasing use of social media platforms, users—especially children and teenagers—are exposed to cyberbullying, harassment, and abusive language.

The system classifies user input text into two categories:
- Safe  
- Harmful  

When harmful content is detected, the system triggers an automated parental alert, displaying a warning message in the application and generating a simulated email notification to inform the parent or guardian.

It also stores detected cases with timestamps for monitoring and tracking purposes.
---

## Objectives

- To detect harmful online content using NLP techniques  
- To classify text into Safe and Harmful categories  
- To compare multiple machine learning and deep learning approaches  
- To build a real-time web application for detection  
- To assist in improving online safety  

---

## Features

- Real-time text analysis using Streamlit  
- Binary classification (Safe / Harmful)  
- Transformer-based embeddings (MiniLM)  
- Machine learning classification using SVM  
- Automatic alert system for harmful content  
- Simulated parent email notification  
- Case history tracking with timestamps  
- Interactive user interface  


---

## Methodology

The project follows a structured pipeline:

1. Data Collection and Integration  
2. Data Preprocessing and Cleaning  
3. Dataset Balancing (Downsampling)  
4. Feature Extraction:
   - TF-IDF
   - Transformer embeddings (MiniLM)
5. Model Training:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
   - LSTM (Deep Learning)
6. Model Evaluation:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
7. Deployment using Streamlit  

---

## Model Details

- Embedding Model: `all-MiniLM-L6-v2` (Sentence Transformers)  
- Classifier: Support Vector Machine (SVM)  
- Task: Binary Classification (Safe vs Harmful)  

### Models Explored

- TF-IDF + Logistic Regression  
- TF-IDF + SVM  
- TF-IDF + Random Forest  
- MiniLM + Logistic Regression  
- MiniLM + SVM  
- MiniLM + Random Forest  
- LSTM (Deep Learning Model)  

### Final Model Selected

MiniLM embeddings + SVM

Reason:
- Strong performance  
- Efficient and fast  
- Suitable for real-time deployment  

---

## Application Workflow

User Input  
→ Text Cleaning  
→ Transformer Embedding (MiniLM)  
→ SVM Prediction  
→ Result Display  
→ Alert Generation  
→ Case Logging (with Timestamp)

---

## How to Run the Project

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Run the Application
streamlit run app.py

### 3. Open in Browser
http://localhost:8501

---

## Dataset Acknowledgement

This project uses publicly available datasets:

- Elsafoury, Fatma (2020),  
  “Cyberbullying datasets”, Mendeley Data, V1  
  DOI: https://doi.org/10.17632/jf4pzyvnpj.1  

- Kaggle Cyberbullying Dataset:  
  https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset/data  

---

## Challenges Faced

- Handling imbalanced dataset  
- Cleaning noisy social media text  
- Combining multiple datasets  
- Model selection and comparison  
- Computational limitations for deep learning models  
- Kernel crashes during training  
- Difficulty in detecting subtle or context-based harmful content  

---

## Limitations

- Model is more focused on detecting general toxicity  
- Limited detection of sexual or grooming-related content  
- Does not consider full conversation context  
- Difficulty with sarcasm and indirect language  
- Binary classification only (no category-specific detection)  

---

## Future Work

- Fine-tuning transformer models  
- Multi-class classification (racism, sexism, etc.)  
- Context-aware detection using conversation history  
- Larger and more diverse datasets  
- Integration with real-time platforms  
- Real email/notification system  
- Model explainability features  

---

## Conclusion

This project demonstrates how machine learning and NLP techniques can be applied to detect harmful online content. Multiple approaches were explored and compared, including TF-IDF, transformer-based embeddings, and LSTM.

Although LSTM achieved slightly higher accuracy, MiniLM embeddings combined with SVM provided the best balance between performance, efficiency, and deployment feasibility. The system was successfully deployed as a Streamlit application with real-time detection, alerting, and history tracking.

---

## Author

Amal Thomas  
AI Capstone Project
