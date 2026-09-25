# Logistics Data Analysis Internship Projects

## Project Overview
This repository contains the strategic planning reports, data preprocessing pipelines, and Python implementations developed during the Logistics Data Analyst Internship. The core focus of these projects is to optimize last-mile urban delivery networks, resolve supply chain bottlenecks, and prepare raw telematics data for predictive machine learning models.

## Repository Structure

### Week 1: Strategic Planning and Data Exploration
**File:** `logistics_optimization.py`
* **Objective:** Establish a strategic framework for last-mile delivery optimization. 
* **Key Features:**
  * Simulates a comprehensive logistics dataset with geographic coordinates, traffic indices, and package weights.
  * Implements **K-Means Clustering** to geographically group delivery coordinates into localized micro-hubs.
  * Utilizes a **Random Forest Regressor** to predict estimated delivery times (ETAs) based on route features, minimizing delays.
* **Tracked KPIs:** On-Time Delivery (OTD) Rate, Route Cost per Kilometer, and Vehicle Capacity Utilization.

### Week 2: Data Collection, Cleaning, and Preprocessing
**File:** `data_preprocessing.py`
* **Objective:** Construct a robust data preprocessing pipeline to handle noisy supply chain telematics, manual logging errors, and sensor dropouts.
* **Key Features:**
  * **Missing Data Imputation:** Uses `KNNImputer` for continuous variables (fuel consumption) and Mode Imputation for categorical variables (traffic density).
  * **Outlier Capping:** Applies the Interquartile Range (IQR) method to cap anomalous delivery durations without discarding necessary data volume.
  * **Feature Scaling:** Normalizes numerical features utilizing `StandardScaler` to ensure balanced model training.
  * **Feature Engineering:** Converts categorical traffic data using One-Hot Encoding to ensure compatibility with numerical estimators.

## Requirements
To run the scripts in this repository, ensure the following Python libraries are installed:
* Python 3.8+
* pandas
* numpy
* scikit-learn
