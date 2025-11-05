# ShipmentSure: Predicting On-Time Delivery Using Supplier Data

# ShipmentSure: Predicting On-Time Delivery Using Supplier Data

##  Project Overview
ShipmentSure is a machine learning project designed to predict whether a product shipment will arrive **on time** based on various supplier and logistics features.  
The goal is to assist supply chain managers in identifying potential delays early, improving **inventory planning**, **customer satisfaction**, and **overall operational efficiency**.

This project involves all key phases of a data science workflow — from **data preprocessing and feature engineering** to **model training, tuning, and evaluation**, followed by **deployment readiness** using Flask/FastAPI or Streamlit.

---

##  Project Workflow

### **1️⃣ Data Collection**
Dataset obtained from Kaggle:  
👉 [On-Time Delivery Dataset](https://www.kaggle.com/datasets/willianoliveiragibin/on-time-delivery/data?select=Train+new.csv)

It includes shipment, supplier, and product information such as:
- Warehouse Block  
- Mode of Shipment  
- Customer Care Calls  
- Customer Rating  
- Cost of Product  
- Prior Purchases  
- Product Importance  
- Gender  
- Discount Offered  
- Weight in Grams  
- Reached on Time (Target)

---

### **2️⃣ Data Preprocessing and EDA**
- Handled missing values, outliers, and encoded categorical variables.  
- Checked class imbalance and applied **SMOTE** to balance the data.  
- Conducted univariate and bivariate analysis using **Matplotlib** and **Seaborn**.  
- Normalized numerical features using **StandardScaler**.

---

### **3️⃣ Feature Engineering**
New derived features were created to enhance model performance:
- `Cost_to_Weight_ratio = Cost_of_the_Product / Weight_in_gms`  
- `Discount_Ratio = Discount_offered / Cost_of_the_Product`  
- `CustomerCare_to_PriorPurchase = Customer_care_calls / Prior_purchases`  
- `Combined_Impact = Cost_to_Weight_ratio * Discount_Ratio`

These features improved the model’s ability to capture shipment cost-efficiency and discount impact.

---

### **4️⃣ Model Building and Evaluation**
Multiple machine learning models were trained and compared:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Naive Bayes  
- XGBoost  
- LightGBM  
- CatBoost  

After hyperparameter tuning and regularization:
- **Best Training Model:** XGBoost  

**Evaluation Metrics Used:**
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

---

### **5️⃣ Model Deployment**
Deployment-ready files were created:
- `shipment_model.pkl` → Final trained model  
- `encoder.pkl`, `scaler.pkl` → Preprocessing objects  
- `app.py` → Streamlit deployment script  
- `requirements.txt` → Dependency file for easy setup  

Deployment options:
- **Flask / FastAPI:** For API-based predictions  
- **Streamlit:** For a simple interactive web interface  

---

## ⚙️ Technologies Used
- **Python 3.x**
- **Pandas, NumPy** → Data manipulation  
- **Scikit-learn** → ML algorithms and preprocessing  
- **XGBoost, LightGBM, CatBoost** → Boosting models  
- **Matplotlib, Seaborn** → Visualization  
- **SMOTE (imblearn)** → Handling class imbalance  
- **Streamlit / FastAPI** → Model deployment  

---

## 🚀 How to Run the Project

### **1️⃣ Clone this Repository**
```bash
git clone https://github.com/your-username/ShipmentSure.git
cd ShipmentSure
