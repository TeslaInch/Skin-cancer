# 🩺 Skin Cancer Classification App

This project is a **Streamlit web app** that classifies skin images as **Benign** or **Malignant** using a pre-trained deep learning model (`my_model.h5`).

## 🚀 Features
- Upload an image (`.jpg`, `.jpeg`, `.png`)
- Preprocessing & resizing to 224x224 pixels
- Deep learning model predicts **Benign** or **Malignant**
- Simple and interactive Streamlit interface

## 🛠️ Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
▶️ Run the App
Make sure my_model.h5 is in the project folder, then run:

bash
Copy code
streamlit run app.py
📂 Project Structure
bash
Copy code
.
├── app.py           # Streamlit app
├── my_model.h5      # Pre-trained model
└── requirements.txt # Dependencies
📸 Example
Launch the app

Upload a skin lesion image

View prediction result: Benign or Malignant