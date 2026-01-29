# Thyroid-Cancer-Prediction-using-Multi-Modal-Data-and-Ensemble-Techniques
A web-based Thyroid Cancer Analysis Tool that integrates machine learning and YOLOv8 deep learning for early cancer detection. It predicts clinical risk using patient data and identifies suspicious thyroid nodules from ultrasound images, providing fast, accurate, and clinician-friendly decision support via Streamlit.

---

## 1. Overview of the Thyroid Gland and Its Function:
We will begin with a simple and short evaluation of the thyroid gland, its anatomy and important functioning of thyroid hormones. This legacy is important for widespread reference knowledge of our research. Our crew will collaborate and summarize the main record of the body structure of the thyroid gland, and ensure that the data is both perfect and smooth to understand.

<img width="630" height="450" alt="image" src="https://github.com/user-attachments/assets/b80b2898-fdde-4ba5-802a-7dfd5e3f25c6" />

---

## 2. Model Architecture:
The thyroid most cancers prediction machine integrates  primary records modalities—medical information and diagnostic imaging—within a unified internet-primarily based interface. The solution is constructed to guide early detection by means of the usage of combining structured medical information with automated photograph analysis.

<img width="585" height="635" alt="image" src="https://github.com/user-attachments/assets/d675c7cf-e6c4-4a29-bcd4-0efe324bed25" />

---

## 3. Annotated Images:
Visual checks on sample cases confirm that YOLOv8 draws accurate bounding boxes around nodules of varying size and contrast. In images where nodules were small or faint, the detector occasionally failed to propose boxes, suggesting that future training with multi-scale augmentation might improve performance on subtle lesions. Nonetheless, the high map underscores the model’s utility in quickly flagging areas of interest for further expert review.

![image](https://github.com/user-attachments/assets/6d7c32c4-2da1-4568-849d-b8a129435d7a)       ![image](https://github.com/user-attachments/assets/89c233d5-fb59-4ac6-8530-7bee820f1d1b)

---

## 4. Integrated Dashboard Outcomes:

<img width="603" height="284" alt="image" src="https://github.com/user-attachments/assets/04ee626d-c1a3-41ce-b245-4375696d86bd" />

---

## 5. Streamlit web application:

<img width="606" height="289" alt="image" src="https://github.com/user-attachments/assets/d041b222-f55b-448a-987f-450537f51742" />

---

## 6. Overview of the model: 
### 🧠 Thyroid Cancer Prediction System (Multimodal AI)

An integrated diagnostic platform for early and accurate prediction of thyroid cancer using both clinical data and medical imaging.  
The system combines machine learning and deep learning models into a single web-based application for decision support.

---

### 🔍 System Overview

The platform consists of three interconnected modules that work together to provide a comprehensive thyroid cancer risk assessment.

---

### 🚀 Workflow (3 Simple Steps)

#### 1️⃣ Clinical Data Analysis
Patient clinical information such as:
- Age and gender  
- TSH, T3, T4 hormone levels  
- Family medical history and lifestyle factors  

is preprocessed and fed into a **stacking ensemble machine learning model**.

**Output:**
- Diagnosis: **Benign / Malignant**
- Cancer **risk score**

---

#### 2️⃣ Medical Image Analysis
Thyroid-related medical images (ultrasound or histopathology) are analyzed using a **YOLOv8-based deep learning model**.

The model:
- Detects suspicious regions  
- Draws bounding boxes  
- Classifies regions as **Normal** or **Thyroid Cancer**

---

#### 3️⃣ Integrated Web Interface
A **Streamlit-based web application** integrates both modules.

Users can:
- Enter clinical data  
- Upload medical images  
- View real-time predictions from both models on a single dashboard

---

### ✨ Key Features
- Multimodal AI (clinical + imaging data)
- Ensemble learning for risk prediction
- YOLOv8 for image-based cancer detection
- Real-time results through a web interface
- Supports clinical decision-making

---

### 🛠️ Technology Stack
- Python  
- Scikit-learn  
- YOLOv8  
- Streamlit  
- OpenCV  
- NumPy  
- Pandas  

---

### 📌 Use Case
Designed to assist healthcare professionals and researchers in the early screening and monitoring of thyroid cancer.

---

