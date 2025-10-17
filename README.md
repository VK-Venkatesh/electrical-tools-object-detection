# Electrical Tools Object Detection

## 📘 Project Overview

This project focuses on **object detection of electrical tools** using **YOLOv8** and a **custom-labeled dataset** created with **Roboflow**. The goal was to accurately detect and classify different electrical tools such as drills, hammers, screwdrivers, and pliers from real-world images.

A total of **500 images** were collected from **Google Images** using **Chrome WebDriver**, labeled in Roboflow, and split into training, validation, and testing sets. The model was trained, evaluated, and finally deployed using a **Streamlit web app** for real-time detection on images and videos.

---

## 🧩 Project Pipeline

### 1. Data Collection

* Images of 10 electrical tools were scraped from **Google Images** using **Chrome WebDriver**.
* Collected images were stored in a single folder and uploaded to **Roboflow**.

### 2. Labeling & Dataset Preparation

* In **Roboflow**, 500 images were labeled manually with class names.
* Classes:['brush', 'drill machine', 'fine point pliers', 'hammer', 'knife', 'pliers', 'scissor', 'screwdrivers', 'spanners', 'tape']
* Roboflow handled data preprocessing, resizing, and augmentation.
* Dataset split: **Train 70%**, **Validation 10%**, **Test 20%**.
* Exported dataset in **YOLOv8 format**, which includes: train-img,labels valid-img,labels test-img,labels

---

### 3. Data Validation

* Verified that each image file has a corresponding label (.txt) file.
* Checked and removed mismatched or irregular files.
* Confirmed dataset integrity before training.

---

### 4. Model Training

* Pretrained weights **YOLOv11n.pt** (YOLOv8-compatible) were downloaded from GitHub.
* Model initialized and trained using Ultralytics YOLO framework.

### 5. Model Evaluation & Testing

* Model evaluated on **validation** and **test** sets.

### 6. Deployment

* The trained model was deployed using **Streamlit** for real-time image and video inference.
* The app allows users to upload images or use webcam feeds for object detection.
  
---

## 🧠 Technologies Used

* **Python**
* **YOLOv8 / YOLOv11n** (Ultralytics)
* **Roboflow** (for data labeling and preprocessing)
* **Streamlit** (for deployment)
* **OpenCV, NumPy, Pandas, Matplotlib**

---

## 🚀 Key Highlights

✅ 500+ images collected and labeled manually.
✅ 10 distinct electrical tool categories.
✅ Model trained using YOLOv8 with Roboflow integration.
✅ Streamlit app for real-time detection on images and videos.
✅ Custom dataset validation and cleaning scripts.

---

## 📈 Future Improvements

* Increase dataset size for better generalization.
* Fine-tune confidence thresholds and augmentations.
* Deploy app on **Render / Hugging Face / Streamlit Cloud**.

---

## 🧾 Acknowledgements

* **Roboflow** for dataset labeling and preprocessing tools.
* **Ultralytics YOLO** for an efficient object detection framework.
* **Kaggle** for training and validation environment.

---

## 👤 Author

**Venktesh**
Deep Learning & Computer Vision Enthusiast
📧 [www.linkedin.com/in/venkatesh-ds25]
🔗 []

---

> “Teaching machines to see the world of electrical tools – one frame at a time.”
