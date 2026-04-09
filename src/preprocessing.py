import pandas as pd

def load_data(filepath):
    """Load dataset from the given filepath."""
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Handle missing values and drop unnecessary columns.
    Returns the cleaned DataFrame.
    """
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
        
    return df
