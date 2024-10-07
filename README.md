## Download Synthetic data

https://synthea.mitre.org/downloads

download either "COVID-19 10K, CSV" or "COVID-19 100K, CSV" , unzip and extract to data folder

Walonoski J, Klaus S, Granger E, Hall D, Gregorowicz A, Neyarapally G, Watson A, Eastman J. Synthea™ Novel coronavirus (COVID-19) model and synthetic data set. Intelligence-Based Medicine. 2020 Nov;1:100007. https://doi.org/10.1016/j.ibmed.2020.100007

notebooks

contains jupyter notebooks for exploratory data analysis

src

data_loader.py
contains code for loading various data files

preprocessing.py
contains code for cleaning data. Missing data can be replaced with mean, median or mode. Anomalous values are detected and removed. Sparse columns are also removed.

feature_engineering.py
categorical variables are converted into binary features. Is_alive feature is created based on absence/presence of death date.

modeling.py

regression model for predicting healthcare expenses and healthcare coverages

Vanilla regression
Regression with feature selection
Decision tree regression
Gradient boosted regression
Regression with polynomial features








