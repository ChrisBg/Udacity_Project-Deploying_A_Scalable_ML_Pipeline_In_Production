# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



# Add the necessary imports for the starter code.
def main():

    # Add code to load in the data.
    logger.info("Loading data")
    data_path = 'data/clean_data.csv'
    data = pd.read_csv(data_path)

    # First, let's check what columns are actually in your DataFrame
    logger.info(f"Available columns: {data.columns.tolist()}")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logger.info("Splitting data")
    train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data['salary'])
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    logger.info("Processing training data")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    logger.info("Processing test data")
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    logger.info("Training model")
    model = train_model(X_train, y_train)

    # Save the model
    logger.info("Saving model")
    joblib.dump(model, 'model/model.joblib')

    # Save the encoder
    logger.info("Saving encoder")
    joblib.dump(encoder, 'model/encoder.joblib')

    # Save the label binarizer
    logger.info("Saving label binarizer")
    joblib.dump(lb, 'model/lb.joblib')

    # Save the test data
    logger.info("Saving test data")
    joblib.dump(X_test, 'model/X_test.joblib')
    joblib.dump(y_test, 'model/y_test.joblib')

    # Evaluate the model
    logger.info("Evaluating model")
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    # Save the metrics
    logger.info("Saving metrics")
    joblib.dump(precision, 'model/precision.joblib')
    joblib.dump(recall, 'model/recall.joblib')
    joblib.dump(fbeta, 'model/fbeta.joblib')

if __name__ == "__main__":
    main()


