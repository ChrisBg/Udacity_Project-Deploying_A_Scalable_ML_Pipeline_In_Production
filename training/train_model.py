# Script to train machine learning model.

import os
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

    # Define paths for saving the model, encoder, label binarizer, test data, and metrics
    model_path = 'model/model.joblib'
    encoder_path = 'model/encoder.joblib'
    lb_path = 'model/lb.joblib'
    #X_test_path = 'model/X_test.joblib'
    #y_test_path = 'model/y_test.joblib'
    #precision_path = 'model/precision.joblib'
    #recall_path = 'model/recall.joblib'
    #fbeta_path = 'model/fbeta.joblib'

    if os.path.exists(model_path) and os.path.isfile(model_path):
        logger.info("Model already exists, skipping training, loading model")
        model = joblib.load(model_path)
        logger.info("Loading encoder")
        encoder = joblib.load(encoder_path)
        logger.info("Loading label binarizer")
        lb = joblib.load(lb_path)
    else:
        # Train and save a model.
        logger.info("Training model")
        model = train_model(X_train, y_train)
        # Save the model
        logger.info("Saving model")
        joblib.dump(model, model_path)
        # Save the encoder
        logger.info("Saving encoder")
        joblib.dump(encoder, encoder_path)
        # Save the label binarizer
        logger.info("Saving label binarizer")
        joblib.dump(lb, lb_path)

    # Save the test data
    #logger.info("Saving test data")
    #joblib.dump(X_test, X_test_path)
    #joblib.dump(y_test, y_test_path)

    # Evaluate the model
    logger.info("Evaluating model")
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    logger.info("Precision: %s, Recall: %s, Fbeta: %s", precision, recall, fbeta)

    # Save the metrics
    #logger.info("Saving metrics")
    #joblib.dump(precision, precision_path)
    #joblib.dump(recall, recall_path)
    #joblib.dump(fbeta, fbeta_path)

if __name__ == "__main__":
    main()


