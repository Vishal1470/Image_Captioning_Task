# 🖼️ Image_Captioning_Task

An AI-powered Image Captioning Web Application built using 
HuggingFace Transformers (BLIP Model) and Streamlit.

This app allows users to upload an image and automatically generates
a meaningful caption using a pre-trained deep learning model.

---

## 🚀 Features

- Upload any image (JPG, PNG)
- Automatic caption generation
- Clean and simple Streamlit UI
- No TensorFlow required
- Uses pre-trained BLIP model
- Fast and lightweight setup

---

## 🧠 Model Used

This project uses:

Model: Salesforce/blip-image-captioning-base  
Library: HuggingFace Transformers  
Framework: PyTorch  

BLIP = Bootstrapped Language Image Pretraining

---

## 📂 Project Structure


Image_Captioning_Task/
│
├── src/
│ ├── app.py # Main Streamlit application
│ ├── train.py # Optional model loading test script
│
├── requirements.txt
├── README.md


---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository


git clone <your-repo-link>
cd Image_Captioning_Task


### 2️⃣ Create Virtual Environment (Recommended)


python -m venv venv
venv\Scripts\activate


### 3️⃣ Install Dependencies


pip install -r requirements.txt


---

## ▶️ Run the Application


streamlit run src/app.py


After running the command, open the local URL shown in the terminal.

---

## 📦 Requirements

Main libraries used:

- streamlit
- torch
- transformers
- pillow

All dependencies are listed in `requirements.txt`.

---

## 🛠️ How It Works

1. User uploads an image.
2. The image is processed using BLIPProcessor.
3. The pre-trained BLIP model generates a caption.
4. The caption is displayed in the web interface.

---

## 📸 Example Output

Input: Image of a dog playing in a park  
Output: "A brown dog running on the grass."

---

## 🔥 Future Improvements

- Add multi-language caption support
- Add download caption feature
- Deploy on Streamlit Cloud
- Add batch image upload
- Improve UI design

---

## 👨‍💻 Author

Vishal Patil

---

## 📜 License

This project is for educational and internship demonstration purposes.
