
# 🏥 IoMT Anomaly Detection Dashboard

This project is a real-time anomaly detection dashboard built with **Streamlit** and powered by **machine learning** models (Random Forest and SVM) to monitor and detect cyber threats in Internet of Medical Things (IoMT) environments. It provides live predictions, anomaly tracking, and evaluation metrics through an interactive web interface. The goal is to enhance cybersecurity in smart hospital systems by enabling early detection of attacks such as spoofing, unauthorized access, and ransomware.

---

## 🔍 Features

- ✅ Real-time simulation of IoMT data and live predictions
- 📊 Detection and visualization of anomalies using **Random Forest** and **Support Vector Machine**
- 🧠 Displays **confusion matrices** for both models
- 📁 Logs predictions for performance analysis
- 📈 Helps visualize false classifications for improving model accuracy
- 🧪 Designed for smart hospital cybersecurity research

---

## 🛠️ Requirements

- Python 3.8+
- Streamlit
- pandas
- scikit-learn
- matplotlib
- joblib
- uvicorn
- FastAPI

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📂 File Structure

```
IoMT-Anomaly-Detection/
│
├── dashboard/
│   ├── dashboard.py             # Main Streamlit dashboard
│   ├── detection_log.csv        # Auto-generated log of predictions
│
├── data/
│   └── IoMT.csv                 # Input dataset (must include 'sl' column for labels)
│
├── model/
│   ├── random_forest.pkl        # Pretrained Random Forest model
│   ├── svm_model.pkl            # Pretrained SVM model
│   └── scaler.pkl               # StandardScaler used for preprocessing
│
├── api/
│   └── app.py                   # (Optional) FastAPI app for backend support
```

---

## 🚀 Running the Dashboard

1. Ensure your dataset `IoMT.csv` is in the `data/` folder and includes a column called `sl` for true class labels.
2. Place your pretrained models and scaler inside the `model/` folder.
3. Run the dashboard:

```bash
streamlit run dashboard/dashboard.py
```

This will open the app in your browser.

---

## 🧪 Model Evaluation

- The dashboard visualizes model performance using **confusion matrices** for both SVM and RF classifiers.
- It logs predictions and allows for a comprehensive analysis of how models respond to live IoMT data.
- Suitable for further academic or research use in anomaly detection, cybersecurity, and healthcare AI.

---

## 📌 Notes

- Labels (`sl`) should be categorical values like `0`, `1`, `2`, or `3` depending on your anomaly types.
- This project assumes you're using a supervised learning approach with pre-labeled data.
- You can expand this project by integrating real-time data feeds or extending to deep learning models.

---
