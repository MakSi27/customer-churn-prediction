from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(preprocessor, model_type="logistic", class_weight="balanced"):
    """
    Builds a scikit-learn Pipeline with the given preprocessor and model.
    """
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            class_weight=class_weight,
            max_depth=8,
            min_samples_split=5
        )
    else:
        raise ValueError("Unsupported model type.")
        
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return clf

def train_model(pipeline, X_train, y_train):
    """
    Trains the given pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline
