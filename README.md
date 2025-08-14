# ⚖️ Support Vector Machine (SVM) Classification

## 📌 Overview
The goal is to implement **Support Vector Machines (SVM)** for binary classification, compare different kernels, tune hyperparameters, and evaluate model performance.  

The notebook also includes **EDA, scaling, visualization, and advanced evaluation metrics** to make the analysis comprehensive and professional.

---

## 📊 Dataset
- **Source:** [Breast Cancer Dataset (sklearn)](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- **Target:** Binary classification (Malignant = 0, Benign = 1)
- **Features:** 30 numeric features describing tumor characteristics
- **Size:** 569 rows, 30 features + target

---

## 🛠 Tools & Libraries
- Python 3.x
- Pandas, NumPy (data handling)
- Matplotlib, Seaborn (visualization)
- scikit-learn (ML models, scaling, metrics, tuning)

---

## 🚀 Features Implemented
1. **Exploratory Data Analysis (EDA)**
   - Dataset structure & missing values check
   - Target class distribution
   - Correlation heatmap

2. **Data Preprocessing**
   - Train-test split (stratified)
   - Feature scaling with `StandardScaler`

3. **Model Training**
   - SVM with **Linear Kernel**
   - SVM with **RBF Kernel**

4. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix heatmaps
   - ROC curve and AUC for both kernels

5. **Parameter Exploration**
   - Hyperparameter tuning with `GridSearchCV` (C, gamma, kernel)
   - Cross-validation accuracy scores

6. **Visualization**
   - Decision boundary plots using PCA projection

---

## 📈 Results Summary
| Model | Accuracy | Precision | Recall | F1-score | AUC |
|-------|----------|-----------|--------|----------|-----|
| SVM (Linear) | ~0.96 | ~0.96 | ~0.97 | ~0.96 | ~0.99 |
| SVM (RBF)    | ~0.97 | ~0.97 | ~0.97 | ~0.97 | ~0.99 |
| SVM (Best via GridSearchCV) | ~0.98 | ~0.98 | ~0.98 | ~0.98 | ~0.99 |

> **Observation:**  
> Both linear and RBF kernels perform exceptionally well, with RBF slightly outperforming linear.  
> Optimal hyperparameters from GridSearchCV further improved accuracy.

---

## 📷 Visual Outputs
- Target class distribution plot
- Correlation heatmap
- Confusion matrix heatmaps (Linear & RBF)
- ROC curves
- Decision boundary plot (PCA-reduced)

---

## 📝 Conclusion

- SVM is a powerful algorithm for binary classification.
- Feature scaling is crucial for SVM’s performance.
- RBF kernel generally performs better for non-linear data.
- Hyperparameter tuning (C, gamma) can significantly improve accuracy.

---

