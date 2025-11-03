PyCaret "Low Code" Data Mining Assignment

This repository contains a Google Colab notebook demonstrating the use of the PyCaret library for various data mining tasks, as required by the assignment.

All tasks are performed on the Fitness & Activity Dataset (Kaggle Link).

Assignment Deliverables

Colab Notebook: The primary file in this repository (pycaret_assignment.ipynb) contains the complete, executable code for all tasks.

Video Tutorial: A video walkthrough (to be recorded by the student) explaining the execution and output of each task.

GitHub Repository: This repository, organized with all code and saved outputs (plots).

Environment Setup & Troubleshooting

A significant part of this project was setting up a stable environment. The default Google Colab environment (Python 3.12) is incompatible with PyCaret 3.3.2.

The notebook uses a multi-step setup process to resolve this:

STEP 1: Installs Python 3.11, sets it as the system default, and installs the pycaret[full]==3.3.2 library. A session restart is required after this step.

STEP 2: Validates the Python 3.11 environment.

Bug Fixes:

Imports PyCaret modules with aliases (e.g., import pycaret.classification as pc) to prevent setup() function name conflicts.

Sets log_experiment=False to avoid a critical AttributeError with Colab's built-in mlflow library.

Wraps plot_model(plot='feature') in a try...except block to handle models that do not support feature importance.

Corrects PyCaret 3.x argument names (e.g., moving contamination to create_model in Anomaly Detection and removing the exogenous_variables argument in Time Series).

Tasks Overview

Task A: Supervised Learning

A-1: Binary Classification

Goal: Predict whether a workout session resulted in a "high calorie burn" (defined as > 300 calories).

Target: high_calorie_burn (1 or 0)

Model: tune_model(compare_models(optimize='AUC'))

A-2: Multiclass Classification

Goal: Predict the activity_type (e.g., Cycling, Swimming, Yoga) based on workout metrics.

Target: activity_type

Model: tune_model(compare_models(optimize='Accuracy'))

A-3: Regression

Goal: Predict the exact number of calories_burned from a workout.

Target: calories_burned (numeric)

Model: tune_model(compare_models(optimize='MAE'))

Task B: Unsupervised Learning - Clustering

Goal: Identify natural groupings (clusters) of users based on their fitness metrics (e.g., steps, heart_rate_avg, duration_min).

Algorithm: K-Means (num_clusters=4)

Plots: Cluster Plot (PCA), Elbow Plot

Task C: Unsupervised Learning - Anomaly Detection

Goal: Identify unusual or outlier workout sessions that deviate from the norm.

Algorithm: Isolation Forest (iforest)

Plots: UMAP Plot

Task D: Association Rules Mining

NOTE: This task was skipped due to unresolvable environment conflicts. The required library (pycaret==2.3.5) has build dependencies (e.g., scipy) that are incompatible with the modern Google Colab environment and Python 3.11. All other tasks were completed successfully using pycaret 3.3.2.

Task E: Time Series Forecasting

E-1: Univariate Forecasting

Goal: Forecast the total number of steps for the next 7 days using only historical step data.

Target: steps

E-2: Univariate Forecasting with Exogenous Variables

Goal: Forecast the total number of steps for the next 7 days, using duration_min as an external factor to improve the prediction.

Target: steps

Exogenous Variable: duration_min (inferred automatically by PyCaret)

Gradio Web App Demos

As required, the notebook concludes by launching two interactive web applications using gradio:

Regression App: Predicts calories_burned based on user inputs.

Multiclass App: Predicts activity_type based on user inputs.

These apps use the models saved in Task A-2 and A-3.
