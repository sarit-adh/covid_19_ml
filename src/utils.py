import os
from data_loader import DataLoader

def get_column_names(base_path=None, filename=None):
    
    if not base_path:
        base_path = '../data/' 
    data_loader = DataLoader(base_path)
    if not filename:
        files = os.listdir(base_path)
    else:
        files = [filename]
    for file in files:
        print(file)
        print("###############################")
        df = data_loader.load_file(file)
        if df is not None:
            for column in df.columns:
                print(f"{column} : {df[column].dtype}")
        print("\n")
                
            
get_column_names()