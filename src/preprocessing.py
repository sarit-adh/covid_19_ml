import pandas as pd
from data_loader import DataLoader

def drop_high_missing_cols(df, threshold=0.5):
    """
    Drops columns with a high percentage of missing values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The percentage threshold for missing values (default is 0.5).
    
    Returns:
    pd.DataFrame: The DataFrame with columns dropped.
    """
    missing_ratio = df.isnull().mean()
    print(missing_ratio)
    return df.loc[:, missing_ratio < threshold]


def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fills missing values in specified columns using a given strategy.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): The strategy to fill missing values ('mean', 'median', 'mode').
    columns (list): List of columns to fill (default is None, which fills all columns).
    
    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]) and strategy == 'mean':
            df.loc[df[col].isnull(), col] = df[col].mean()
        elif pd.api.types.is_numeric_dtype(df[col]) and strategy == 'median':
            df.loc[df[col].isnull(), col] = df[col].median()
        elif pd.api.types.is_numeric_dtype(df[col]) and strategy == 'mode':
            df.loc[df[col].isnull(), col] = df[col].mode()[0]
    return df


def clean_data(df):
    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    df = drop_high_missing_cols(df, threshold=0.5)
    df = fill_missing_values(df, strategy='mean')
    return df

def detect_anomalies(df, method='zscore', columns=None, threshold=3):
    """
    Detects anomalous values in specified numeric columns using the Z-score or IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    method (str): The method for detecting anomalies ('zscore' or 'iqr').
    columns (list): List of columns to check for anomalies (default is None, which checks all numeric columns).
    threshold (float): The threshold for defining anomalies (default is 3 for Z-score).
    
    Returns:
    pd.DataFrame: A DataFrame with boolean values indicating anomalies.
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
    
    anomalies = pd.DataFrame(False, index=df.index, columns=columns)  # Initialize with False
    print(anomalies)
    for col in columns:
        if method == 'zscore':
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            anomalies[col] = (abs(z_scores) > threshold)
        elif method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    return anomalies
    

# Test    
def main():
    data_loader = DataLoader("../data/")
    df = data_loader.load_file('patients.csv')
    print("Before cleaning: ", df.shape, df.isnull().sum())
    df = clean_data(df)
    print("After cleaning: ", df.shape, df.isnull().sum())
    
    anomaly_df = detect_anomalies(df, method='zscore', columns=["HEALTHCARE_COVERAGE"])
    print(anomaly_df)
    print(df[anomaly_df.HEALTHCARE_COVERAGE==True])
    


if __name__=="__main__":
    main()    