from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def get_preprocessor():
    """
    Returns the ColumnTransformer for scaling numeric columns
    and one-hot encoding categorical columns.
    """
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    cat_cols = [
        "SeniorCitizen", "gender", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols)
        ]
    )
    
    return preprocessor, num_cols, cat_cols

def split_data(df, target_col="Churn"):
    """
    Splits data into X_train, X_test, y_train, y_test.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Map target
    y = y.map({"Yes": 1, "No": 0})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
