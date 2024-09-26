from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from feature_engineering import feature_engineering_patients
from preprocessing import clean_data
import matplotlib.pyplot as plt
import seaborn as sns

from visualization import visualize_true_vs_predicted

def linear_regression(X, y):
    # Model for prediction of healthcare expenses
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

    
    


def main():
    data_loader = DataLoader("../data/")
    df = data_loader.load_file('patients.csv')
    df = clean_data(df)
    df = feature_engineering_patients(df)
    print(df.head())
    print(df.columns)
    data = df.drop(columns=['Id', 'BIRTHPLACE', 'ADDRESS', 'CITY', 'ZIP', 'LAT', 'LON'])


    X = data.drop(columns=['HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE'])  
    y1 = data['HEALTHCARE_EXPENSES']  
    y2 = data['HEALTHCARE_COVERAGE']
    
    
    # Model for prediction of healthcare expenses
    y_test, y_pred = linear_regression(X, y1)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Squared Error (healthcare expenses): {mse}") 
    print("sample prediction")
    print(y_test[0], y_pred[0])
    visualize_true_vs_predicted(y_test, y_pred, "healthcare expenses")
    
    
    # Model for prediction of healthcare coverage
    y_test, y_pred = linear_regression(X, y2)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Squared Error (healthcare coverage): {mse}") 
    print("sample prediction")
    print(y_test[0], y_pred[0])
    visualize_true_vs_predicted(y_test, y_pred, "healthcare coverage")


if __name__ == "__main__": main()