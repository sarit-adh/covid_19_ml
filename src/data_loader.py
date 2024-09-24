import os
import pandas as pd

class DataLoader:
    def __init__(self, base_path=None):
        """
        Initialize the DataLoader with an optional base path where the CSV files are located.
        :param base_path: Base directory path where the CSV files are located (optional).
        """
        self.base_path = base_path

    def load_file(self, file_name, use_columns=None, dtypes=None, chunksize=None, memory_efficient=False):
        """
        Load a single CSV file into a pandas DataFrame.
        :param file_name: Name of the file to load.
        :param use_columns: List of column names to load (optional).
        :param dtypes: Dictionary specifying column data types (optional).
        :param chunksize: Number of rows to load per chunk (optional).
        :param memory_efficient: If True, loads file in chunks for large files (optional).
        :return: DataFrame or iterator depending on memory_efficient setting.
        """
        file_path = self.get_file_path(file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        
        try:
            if memory_efficient:
                return pd.read_csv(file_path, usecols=use_columns, dtype=dtypes, chunksize=chunksize)
            else:
                return pd.read_csv(file_path, usecols=use_columns, dtype=dtypes)
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            return None

    def get_file_path(self, file_name):
        """
        Get the full file path for the given file name.
        :param file_name: Name of the file.
        :return: Full file path.
        """
        if self.base_path:
            return os.path.join(self.base_path, file_name)
        return file_name

    def load_and_merge_files(self, file_info_list, merge_on, how='inner', memory_efficient=False, chunksize=None):
        """
        Load multiple CSV files and merge them into a single DataFrame.
        :param file_info_list: List of tuples with file names and optional column names and dtypes to use.
        :param merge_on: Column name or list of columns to join on.
        :param how: Type of join to perform (default is 'inner').
        :param memory_efficient: If True, load files in chunks for large datasets (optional).
        :param chunksize: Number of rows to load per chunk (optional).
        :return: Merged DataFrame.
        """
        dfs = []
        for file_info in file_info_list:
            file_name, use_columns, dtypes = file_info
            df = self.load_file(file_name, use_columns=use_columns, dtypes=dtypes, memory_efficient=memory_efficient, chunksize=chunksize)
            if df is None:
                continue
            dfs.append(df)

        if not dfs:
            print("No files were loaded successfully.")
            return None
        
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on=merge_on, how=how)

        return merged_df



#TEST    
def main():    
    loader = DataLoader(base_path='../data/')

    #TEST to load single file
    df = loader.load_file('allergies.csv')
    print(df.head())
    
    '''
    # TEST to load allergies.csv and conditions.csv and merge them on 'PATIENT' key
    file_info = [
        ('allergies.csv', ['PATIENT', 'DESCRIPTION'], {'PATIENT': 'str', 'DESCRIPTION': 'str'}),
        ('conditions.csv', ['PATIENT', 'DESCRIPTION'], {'PATIENT': 'str', 'DESCRIPTION': 'str'})
    ]
    merged_df = loader.load_and_merge_files(file_info, merge_on='PATIENT').rename(columns={'DESCRIPTION_x' : 'ALLERGY', 'DESCRIPTION_y' : 'CONDITION'})
    
    if merged_df is not None:
        print(merged_df.head())
    '''
    
if __name__ == "__main__": main()