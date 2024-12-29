import json
from datetime import datetime


def save_model_card_as_markdown(model_card, markdown_file_path):
    """
    Saves a model card as a Markdown file directly.

    Args:
        model_card (dict): Dictionary containing model card details.
        markdown_file_path (str): Path to save the generated Markdown file.

    Returns:
        None
    """
    # Generate Limitations and Ethical Considerations as bullet points
    limitations = "\n".join(f"- {lim}" for lim in model_card["limitations"])
    ethical_considerations = "\n".join(f"- {eth}" for eth in model_card["ethical_considerations"])
    
    # Create Markdown content
    markdown_content = f"""
# Model Card: {model_card['model_name']}
## Author
{model_card['author']}

## Date Created
{model_card['date_created']}

## Model Description
{model_card['model_description']}

## Intended Use
{model_card['intended_use']}

## Dataset Details
- **Name:** {model_card['dataset']['name']}
- **Source:** {model_card['dataset']['source']}
- **Size:** {model_card['dataset']['size']}
- **Columns:** {model_card['dataset']['columns']}
- **Target:** {model_card['dataset']['target']}
- **Features:** {model_card['dataset']['features']}
- **Categorical Features:** {model_card['dataset']['categorical_features']}
- **Numerical Features:** {model_card['dataset']['numerical_features']}
- **Preprocessing Steps:** {model_card['dataset']['preprocessing_steps']}

## Training Details
- **Algorithm:** {model_card['training_details']['algorithm']}
- **Hyperparameters:** {model_card['training_details']['hyperparameters']}
- **Validation Strategy:** {model_card['training_details']['validation_strategy']}
- **Hyperparameter Tuning:** {model_card['training_details']['hyperparameter_tuning']}
- **Slice Analysis:** {model_card['training_details']['slice_analysis']}

## Performance
- **Precision:** {model_card['performance']['precision']}
- **Recall:** {model_card['performance']['recall']}
- **Fbeta:** {model_card['performance']['fbeta']}
- **Test Set Performance:** {model_card['performance']['test_set_performance']}

## Limitations
{limitations}

## Ethical Considerations
{ethical_considerations}
    """
    
    # Write to Markdown file
    with open(markdown_file_path, 'w') as markdown_file:
        markdown_file.write(markdown_content.strip())
    
    print(f"Markdown model card saved to: {markdown_file_path}")


if __name__ == "__main__":
    
    model_card = {
        "model_name": "Census Income Classifier",
        "model_description": "A classification pipeline using HistGradientBoostingClassifier to predict salary class based on census data",
        "intended_use": "The model is trained to be used for predicting salary class based on census data in the context of a Udacity project for the Machine Learning Engineer DevOps Nanodegree program.",
        "author": "Christophe Bourgoin",
        "date_created": "2024-12-29",
        "additional_info": "For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf",
        "dataset": {
            "name": "Census Income Dataset",
            "source": "https://archive.ics.uci.edu/dataset/20/census+income",
            "size": "32,561 rows",
            "columns": "15 columns",
            "target": "salary",
            "features": "14 features",
            "categorical_features": "9 categorical features",
            "numerical_features": "6 numerical features",
            "preprocessing_steps": "Categorical features encoded using OneHotEncoder, the target feature is encoded using LabelBinarizer."
        },
        "training_details": {
            "algorithm": "HistGradientBoostingClassifier",
            "hyperparameters": {
                "loss": "log_loss",
                "learning_rate": 0.2,
                "max_iter": 100,
                "max_depth": 5,
                "min_samples_leaf": 20,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 10,
                "max_features": 1.0,
                "l2_regularization": 1.0,
                "max_bins": 255
            },
            "validation_strategy": "80-20 train-test split with 5-fold stratified cross-validation.",
            "hyperparameter_tuning": "The hyperparameters were tuned using RandomizedSearchCV with 5-fold stratified cross-validation and only 100 iterations.",
            "slice_analysis": "It's possible to evaluate also the model performance on different slices of the data. For example for the following features: education, age, occupation, race, sex, and native-country."
        },
        "performance": {
            "precision": 0.7893,
            "recall": 0.6760,
            "fbeta": 0.7283,
            "test_set_performance": "Metrics were evaluated on a hold-out test set of 20% of the data."
        },
        "limitations": [
            "Data came from an old census survey, so it's not representative of the actual population",
            "The model may not perform well on data distributions that differ significantly from the training set.",
            "Some categorical features might be underrepresented in the dataset.",
            "The trained model is not a production ready model, it's a proof of concept."
        ],
        "ethical_considerations": [
            "The dataset should not be used for any other purpose than the one specified in the intended use.",
            "Bias in the dataset could lead to unfair predictions for certain demographic groups."
        ]
    }

    # Save the model card as Markdown
    save_model_card_as_markdown(model_card, "model_card.md")
