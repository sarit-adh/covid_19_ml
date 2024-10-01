import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from data_loader import DataLoader
from feature_engineering import feature_engineering_conditions, feature_engineering_patients
from preprocessing import clean_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, RFE
from visualization import visualize_tree, visualize_true_vs_predicted

def run_linear_regression(X, y, y_label):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    linear_regression_model = LinearRegression()

    # Train the model
    linear_regression_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = linear_regression_model.predict(X_test)
    

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error ({y_label}): {mse}") 
    print(f"R-squared (R2 Score): {r2}")
    print(f"sample prediction: True Label : {y_test[0]} , Predicted Label : {y_pred[0]}")
    visualize_true_vs_predicted(y_test, y_pred, y_label)
    
def run_linear_regression_with_feature_selection(X, y, y_label):
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    linear_regression_model = LinearRegression()
    
    select_k_best = SelectKBest(score_func=f_regression)  # Use f_regression for regression tasks
    rfe = RFE(estimator=linear_regression_model)
    
    
    lasso = SelectFromModel(Lasso())

    # Create a pipeline where feature selection is followed by the linear regression model
    pipeline = Pipeline([
        ('feature_selection', select_k_best),  # Placeholder for feature selection
        ('model', linear_regression_model)     # Placeholder for model
    ])

    # Define parameter grid for GridSearchCV
    param_grid = [
        # Using SelectKBest with different values of 'k' (number of features)
        {
            'feature_selection': [select_k_best],
            'feature_selection__k': [5, 10, 15]  # Try different numbers of features
        },
        # Using Recursive Feature Elimination (RFE) with different numbers of features
        {
            'feature_selection': [rfe],
            'feature_selection__n_features_to_select': [5, 10, 15]  # Different numbers of features
        },
        # Using Lasso for feature selection with different alpha values
        {
            'feature_selection': [lasso],
            'feature_selection__estimator__alpha': [0.01, 0.1, 0.5]  # Different values of alpha for Lasso
        }
    ]
    
    # Initialize GridSearchCV with the pipeline and the parameter grid
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters found
    print("Best parameters found: ", grid_search.best_params_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate mean squared error on the test set
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test Set Mean Squared Error: {mse}")
    print(f"R-squared (R2 Score): {r2}")
    print(f"sample prediction: True Label : {y_test[0]} , Predicted Label : {y_pred[0]}")
    visualize_true_vs_predicted(y_test, y_pred, y_label)
    
    
def run_decision_tree_regression(X, y, y_label):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Decision Tree Regressor
    decision_tree_model = DecisionTreeRegressor(max_depth=10, random_state=42)

    # Fit the model to the training data
    decision_tree_model.fit(X_train, y_train)

    # Predicting on the test data
    y_pred = decision_tree_model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Printing evaluation results
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2 Score): {r2}")
    print(f"sample prediction: True Label : {y_test[0]} , Predicted Label : {y_pred[0]}")

    #visualize_tree(decision_tree_model, X.columns, y_label) # figure is legible only upto max depth 4
    visualize_true_vs_predicted(y_test, y_pred, y_label)
    
def run_gradient_boosting_regression(X, y, y_label):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Gradient Boosting Regressor
    gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Fit the model to the training data
    gradient_boosting_model.fit(X_train, y_train)

    # Predicting on the test data
    y_pred = gradient_boosting_model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Printing evaluation results
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2 Score): {r2}")
    print(f"sample prediction: True Label : {y_test[0]} , Predicted Label : {y_pred[0]}")

    visualize_true_vs_predicted(y_test, y_pred, y_label)
    
# Test
def main():
    data_loader = DataLoader("../data/")
    df = data_loader.load_file('patients.csv')
    df = clean_data(df)
    df = feature_engineering_patients(df)
    print(df.head())
    print(df.columns)
    data = df.drop(columns=['Id', 'BIRTHPLACE', 'ADDRESS', 'CITY', 'ZIP', 'LAT', 'LON'])
    
    
    '''
    # check for linearity (assumption #1 of linear regression model)
    g = sns.pairplot(data, x_vars='AGE', y_vars=['HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE'], height=5, aspect=3)
    plt.suptitle('Pairplot of AGE vs Healthcare Expenses and Coverage')
    plt.show()
    '''
    

    '''
    X = data.drop(columns=['HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE'])  
    y1 = data['HEALTHCARE_EXPENSES']  
    y2 = data['HEALTHCARE_COVERAGE']
    '''
    
    '''
    # Linear Regression Model for prediction of healthcare expenses
    run_linear_regression(X, y1,'HEALTHCARE_EXPENSES')
    
    # Linear Regression Model for prediction of healthcare coverage
    run_linear_regression(X, y2, 'HEALTHCARE_COVERAGE')
    '''
    
    '''
    # Decision Tree Model for prediction of healthcare expenses
    run_decision_tree_regression(X, y1,'HEALTHCARE_EXPENSES')
    
    # Decision Tree Model for prediction of healthcare coverage
    run_decision_tree_regression(X, y2, 'HEALTHCARE_COVERAGE')
    '''
    
    '''
    # Linear Regression Model with Feature Selection
    run_linear_regression_with_feature_selection(X, y1, 'HEALTHCARE_EXPENSES')
    '''
    
    '''
    # Linear Regression with gradient boosted regression
    run_gradient_boosting_regression(X, y1, 'HEALTHCARE_EXPENSES')
    '''
    
    '''
    # Polynomial features
    
    continuous_features = ['AGE']  # Add more continuous variables if needed
    discrete_features = [col for col in X.columns if col not in continuous_features]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_continuous_poly = poly.fit_transform(data[continuous_features])
    poly_feature_names = poly.get_feature_names_out(continuous_features)

    df_continuous_poly = pd.DataFrame(X_continuous_poly, columns=poly_feature_names)
    df_discrete = df[discrete_features].reset_index(drop=True)
    X_poly = pd.concat([df_continuous_poly, df_discrete], axis=1)
    
    print(X_poly.columns)
    
    run_linear_regression(X_poly, y1,'HEALTHCARE_EXPENSES')
    '''
    
    conditions_df = data_loader.load_file('conditions.csv')
    conditions_df = conditions_df[['PATIENT','DESCRIPTION']]
    conditions_df = clean_data(conditions_df)
    conditions_df = feature_engineering_conditions(conditions_df)
    patients_df = df.drop(columns=['BIRTHPLACE', 'ADDRESS', 'CITY', 'ZIP', 'LAT', 'LON'])
    combined_df = patients_df.merge(conditions_df, left_on="Id", right_on="PATIENT")
    combined_df = combined_df.drop(columns=['Id', 'PATIENT'])
    print(combined_df.columns)
    
    X = combined_df.drop(columns=['HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE'])  
    y1 = combined_df['HEALTHCARE_EXPENSES']  
    y2 = combined_df['HEALTHCARE_COVERAGE']
    
    #run_linear_regression(X, y1,'HEALTHCARE_EXPENSES')
    
    
    
    
    
    
    
    


if __name__ == "__main__": main()