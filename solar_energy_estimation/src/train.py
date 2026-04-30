# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set() # Revert to matplotlib defaults
plt.rcParams['figure.figsize'] = (14, 10)

"""### Model training and testing"""

# import libraries for algorithms traininng, and metrics to judge performance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# training data
train_df = pd.read_csv(r'C:\Users\Dell\Documents\my_linux\repos_\large-scale-solar-energy-generation\solar_energy_estimation\data\processed\train.csv')
X_train = train_df.drop(['Radiation'], axis = 1)
y_train = train_df['Radiation']

# test data
test_df = pd.read_csv(r'C:\Users\Dell\Documents\my_linux\repos_\large-scale-solar-energy-generation\solar_energy_estimation\data\processed\test.csv')
X_test = test_df.drop(['Radiation'], axis = 1)
y_test = test_df['Radiation']

X_test.info()

"""### Linear Regression"""

# Setup the pipeline steps for linear regression
#steps = [('scaler', StandardScaler()), ('lr', LinearRegression())]
# Create the pipeline
linear_regression = LinearRegression()#Pipeline(steps)
# Fit the pipeline to the train set
linear_regression.fit(X_train, y_train)
# Predict the labels of the test set
y_pred_lr = linear_regression.predict(X_test)

#Evaluating algorithm performance
mse = mean_squared_error(y_test, y_pred_lr, squared = False)

mae = mean_absolute_error(y_test, y_pred_lr)

print('r2_score', r2_score(y_test, y_pred_lr))

print('Root Mean Squared Error: %.2f' % np.sqrt(mse))

print('Root Mean Absolute Error: %.2f' % np.sqrt(mae))

#Run the model against the test data presented through a plot
fig, pX = plt.subplots()

pX.scatter(y_test, y_pred_lr, edgecolors = (0, 0, 0))
pX.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'm--', lw = 3)
pX.set_xlabel('Actual solar irradiation')
pX.set_ylabel('Predicted solar irradiation')
pX.set_title(" Linear regression: Verified vs Predicted solar irradiation")
plt.savefig("lr_line_of_fit.jpg")
plt.show()

sns.jointplot(y_test, y_pred_lr, kind = 'reg')
plt.savefig("lr_jointplot.jpg")
plt.show()

"""### Random Forest Regressor"""

# Setup the pipeline steps for random forest: steps
steps = [('scaler', StandardScaler()), ('rfr', RandomForestRegressor())]
# Create the pipeline: pipeline
pipeline_rfr = Pipeline(steps)
# Fit the pipeline to the train set
pipeline_rfr.fit(X_train, y_train)
# Predict the labels of the test set
y_pred_rfr = pipeline_rfr.predict(X_test)

# Evaluating algorithm performance
mse_rf = mean_squared_error(y_test, y_pred_rfr, squared=False)

mae_rf = mean_absolute_error(y_test, y_pred_rfr)

print('r2_score', r2_score(y_test, y_pred_rfr))

print('Root Mean Squared Error: %.2f' % np.sqrt(mse_rf))

print('Root Mean Absolute Error: %.2f' % np.sqrt(mae_rf))

# Run the model against the test data presented through a plot
fig, pX = plt.subplots()

pX.scatter(y_test, y_pred_rfr, edgecolors = (0, 0, 0))
pX.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'm--', lw = 3)
pX.set_xlabel('Actual solar irradiation')
pX.set_ylabel('Predicted solar irradiation')
pX.set_title(" Random Forest: Verified vs Predicted solar irradiation")
plt.savefig("rfr_line_of_fit.jpg")
plt.show()

sns.jointplot(y_test, y_pred_rfr, kind = 'reg')
plt.savefig("rfr_jointplot.jpg")
plt.show()

"""### GradientBoosting Regressor"""

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()), ('gbr', GradientBoostingRegressor())]
# Create the pipeline: pipeline
pipeline_gbr = Pipeline(steps)
# Fit the pipeline to the train set
pipeline_gbr.fit(X_train, y_train)
# Predict the labels of the test set
y_pred_gbr = pipeline_gbr.predict(X_test)

# Evaluating algorithm performance
mse_gr = mean_squared_error(y_test, y_pred_gbr, squared = False)

mae_gr = mean_absolute_error(y_test, y_pred_gbr)

print('r2_score', r2_score(y_test, y_pred_gbr))

print('Root Mean Squared Error: %.2f' % np.sqrt(mse_gr))

print('Root Mean Absolute Error: %.2f' % np.sqrt(mae_gr))

# Run the model against the test data presented through a plot
fig, pX = plt.subplots()

pX.scatter(y_test, y_pred_gbr, edgecolors = (0, 0, 0))
pX.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'm--', lw = 3)
pX.set_xlabel('Actual solar irradiation')
pX.set_ylabel('Predicted solar irradiation')
pX.set_title(" Gradient Boost: Verified vs Predicted solar irradiation")
plt.savefig("gbr_line_of_fit.jpg")
plt.show()

sns.jointplot(y_test, y_pred_gbr, kind = 'reg')
plt.savefig("gbr_jointplot.jpg")
plt.show()

"""### Model Validation"""

# validation data
val_df = pd.read_csv('/home/josephitopa/Documents/personal_files/ufuoma/data/cleaned_data/validation.csv')
X_val = val_df.drop(['Radiation'], axis = 1)
y_val = val_df['Radiation']

#y_val_lr = pipeline_lr.predict(X_val)
y_val_rfr = pipeline_rfr.predict(X_val)
y_val_gbr = pipeline_gbr.predict(X_val)

# Evaluating algorithm performance for linear regression
#mse_lr_val = mean_squared_error(y_val, y_val_lr, squared = False)

#mae_lr_val = mean_absolute_error(y_val, y_val_lr)

#print('r2_score', r2_score(y_val, y_val_lr))

#print('Linear Regression - Root Mean Squared Error: %.2f' % np.sqrt(mse_lr_val))

#print('Linear Regression - Root Mean Absolute Error: %.2f' % np.sqrt(mse_lr_val))

# Evaluating algorithm performance for random forest regression
mse_rf_val = mean_squared_error(y_val, y_val_rfr, squared = False)

mae_rf_val = mean_absolute_error(y_val, y_val_rfr)

print('r2_score', r2_score(y_val, y_val_rfr))

print('Random Forest - Root Mean Squared Error: %.2f' % np.sqrt(mse_rf_val))

print('Random Forest - Root Mean Absolute Error: %.2f' % np.sqrt(mse_rf_val))

# Evaluating algorithm performance for gradient boost regression
mse_gbr_val = mean_squared_error(y_val, y_val_gbr, squared = False)

mae_gbr_val = mean_absolute_error(y_val, y_val_gbr)

print('r2_score', r2_score(y_val, y_val_gbr))

print('Gradient Boost - Root Mean Squared Error: %.2f' % np.sqrt(mse_gbr_val))

print('Gradient Boost - Root Mean Absolute Error: %.2f' % np.sqrt(mae_gbr_val))

"""### Explainability"""

df = pd.read_csv('C:\Users\Dell\Documents\my_linux\repos_\large-scale-solar-energy-generation\solar_energy_estimation\data\processed\model.csv')
X = df.drop("Radiation", axis = 1)

import shap

explainer = shap.KernelExplainer(linear_regression.predict, X_train) #shap.Explainer(linear_regression)

shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)

# visualize the first prediction
shap.plots.waterfall(shap_values[0])

# visualize the second prediction
shap.plots.waterfall(shap_values[1])

#the overall impact of these features across multiple instances using SHAP summary plot
shap.summary_plot(shap_values, X)

# global feature importance
shap.plots.bar(shap_values)

"""## Experiment tracking with neptune"""

# Terminal
# !pip install neptune-notebooks
# jupyter nbextension enable --py neptune-notebooks
# !pip install neptune-client==0.14.2
#!pip install neptune-client neptune-sklearn
# !pip install neptune-sklearn

# import neptune libraries and sklearn integration
import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils

run = neptune.init(api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZGI1ZjU1MC0yMDhkLTQ5NWMtYTQzYy0xOGMzMTBmNGZlMTUifQ==",
                   project = "samsonooye/solar-irradiation-prediction-2022",
                   name = 'predicting-solar-irradiation',
                   source_files = ["training-and-experiment_tracking.ipynb"])

# Track files, models, and folders
# Folders
run["cleaned_datasets"].track_files("~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/datasets/cleaned")
run["raw_datasets"].track_files("~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/datasets/raw")

# Folder
run["notebooks"].track_files("~/Documents/Projects/update_project/My_Best_Projects/Solar_Energy_Project/notebooks/")

# train, and test; while tracking the experiment
run['lr_summary'] = npt_utils.create_regressor_summary(pipeline_lr, X_train, X_test, y_train, y_test)

run['gbr_summary'] = npt_utils.create_regressor_summary(pipeline_gbr, X_train, X_test, y_train, y_test)

run['rfr_summary'] = npt_utils.create_regressor_summary(pipeline_rfr, X_train, X_test, y_train, y_test)

# stop experiment
run.stop()

"""### Export models"""

# check sklearn version
from sklearn import __version__

print(__version__)

# pip install scikit-learn==0.24.1

# loading libraries
import joblib

# Save Models to file in the model directory
# save linear regression model
#joblib_lr = "model_lr.pkl"
#joblib.dump(pipeline_lr, joblib_lr)
# save random forest model
joblib_rfr = "model_rfr.pkl"
joblib.dump(pipeline_rfr, joblib_rfr)
# save gradientboost model
joblib_gbr = "model_gbr.pkl"
joblib.dump(pipeline_gbr, joblib_gbr)

"""### Recommended reading

1. https://docs.neptune.ai/integrations-and-supported-tools/model-training/sklearn

2. https://mlflow.org/docs/latest/tracking.html#scenario-1-mlflow-on-localhost

3. https://docs.neptune.ai/getting-started/examples#data-versioning
"""