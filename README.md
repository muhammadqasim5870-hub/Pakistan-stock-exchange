# 📈 Pakistan Stock Market Decision Tree

## 📌 Project Overview
This project uses a Decision Tree Regressor to predict stock closing prices using historical Pakistan Stock Exchange (PSX) data from 2017 to 2025.

The project includes data analysis, model training, testing, evaluation, and deployment through Streamlit.

## 🎯 Objectives
- Analyze historical Pakistan Stock Exchange data.
- Perform Exploratory Data Analysis (EDA) and data visualization.
- Split the dataset into 80% training and 20% testing data.
- Train a Decision Tree Regression model.
- Evaluate model performance using regression metrics.
- Deploy an interactive prediction application.

## 📂 Dataset
**Dataset:** Pakistan Stock Market Historical Data (2017–2025)

**Source:** Kaggle

The dataset contains historical stock market information, including numerical stock features and closing prices.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 🤖 Machine Learning Model
**Algorithm:** Decision Tree Regressor

**Train-test split:**
- Training data: 80%
- Testing data: 20%

**Evaluation metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The model is evaluated on the test dataset to measure its prediction performance.

## 📁 Repository Structure

```text
psx-decision-tree/
├── app.py
├── requirements.txt
├── feature_columns.json
├── psx_model_metrics.json
├── psx_decision_tree_model.joblib
└── README.md
```

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/psx-decision-tree.git
cd psx-decision-tree
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🌐 Live Deployment

The application can be accessed online:

[Open the PSX Decision Tree App](https://share.streamlit.io/)

Replace this link with your actual deployed Streamlit application URL.

## 📊 Results
The model's performance is evaluated using MAE, MSE, RMSE, and R² Score. The actual results are available in `psx_model_metrics.json`.

## ⚠️ Disclaimer
This project is for educational and machine learning practice purposes only. Predictions are not financial advice and should not be used as the sole basis for investment decisions.

## 👨‍💻 Author
**Your Name**

GitHub: https://github.com/YOUR-USERNAME

---

⭐ If you find this project useful, consider giving the repository a star.
