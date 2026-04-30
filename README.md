## Large Scale Solar Energy Prediction
This project aims to predict solar irradiation and with other mathematical methods estimate the quantity of energy to be generated, using eastern part of Nigeria as a case study.
### Problem Statement
Using historical weather data, we use machine learning to predict daily solar irradiation.
### Objectives
1. Carry out exploratory data analysis, and descriptive statistical analysis.
2. Train different models(gradient boosting, random forest algorithm, etc) on the data
3. Evaluate the model performance for different models.
4. Track and store eperiment artifacts. 
### Project Structure
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── train.py
    │   ├── data_exploratory_analysis.py
    │   └── evaluate.py
    ├── models/
    ├── reports/
    │   └── figures/
    ├── requirements.txt
    ├── README.md
    └── .gitignore

Installation
### Clone repository
git clone git@github.com:SamsonO88/large-scale-solar-energy-generation.git

### Navigate into project
cd large-scale-solar-energy-generation

### Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Evaluation Metrics
Metrics used: 
    1) MAE
    2) RMSE
    3) R2 SCORE
### Technologies Used
    1) Python
    2) Pandas, NumPy
    3) Scikit-learn
    4) Matplotlib / Seaborn
    5) Jupyter Notebook
    6) shap
    7) xgboost
    8) neptune-notebooks
    9) neptune-client 
    10) neptune-sklearn
### License
    MIT License