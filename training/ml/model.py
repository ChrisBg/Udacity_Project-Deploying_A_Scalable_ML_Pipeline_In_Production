from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Initialize the model
    model = HistGradientBoostingClassifier()

    # Define parameter distributions
    param_distributions = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 3, 5, 10],
        'min_samples_leaf': [10, 20, 30],
        'max_iter': [100, 200, 300],
        'l2_regularization': [0.0, 1.0, 10.0],
    }

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='accuracy',
        n_jobs=-1,
        cv=5,
        verbose=2,
        random_state=42
    )

    # Fit the model
    random_search.fit(X_train, y_train) 

    # Output best parameters and evaluate
    print("Best Hyperparameters:", random_search.best_params_)
    
    best_model = random_search.best_estimator_
    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds