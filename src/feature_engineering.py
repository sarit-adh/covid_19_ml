import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import DataLoader
from preprocessing import clean_data



def feature_engineering_patients(df):
    """
    Perform feature engineering on the patient demographic dataset.
    :param df: Pandas DataFrame of the raw data
    :return: Processed DataFrame ready for modeling
    """
    # Drop columns that are identifiers
    df = df.drop(columns=["SSN", "DRIVERS", "PASSPORT", "PREFIX", "FIRST", "LAST", "MAIDEN"])
    
    # Convert BIRTHDATE and DEATHDATE to datetime
    df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'], errors='coerce')
    df['DEATHDATE'] = pd.to_datetime(df['DEATHDATE'], errors='coerce')

    # Calculate age from birthdate and deathdate
    df['AGE'] = (pd.to_datetime("today") - df['BIRTHDATE']).dt.days // 365
    # df['AGE'] = df['AGE'].fillna(df['AGE'].mean())  # Fill missing age values

    # Create a binary variable for whether the patient is alive or dead
    df['IS_ALIVE'] = df['DEATHDATE'].isna().astype(int)
    
    # Drop BIRTHDATE and DEATHDATE after feature extraction
    df = df.drop(columns=['BIRTHDATE', 'DEATHDATE'])
    
    # Handle missing values for MARITAL status
    df['MARITAL'] = df['MARITAL'].fillna('Unknown')  # Fill missing marital status with 'Unknown'
    
    
    # One-Hot Encode categorical columns: MARITAL, RACE, ETHNICITY, GENDER, STATE, COUNTY
    categorical_cols = ['MARITAL', 'RACE', 'ETHNICITY', 'GENDER', 'STATE', 'COUNTY']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Standardize numerical columns: HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, AGE
    numerical_cols = ['HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'AGE']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

#Test
def main():
    data_loader = DataLoader("../data/")
    df = data_loader.load_file('patients.csv')
    df = clean_data(df)
    df = feature_engineering_patients(df)
    print(df.head())

if __name__ == "__main__": main()