# Model Card: 
- Author : Christophe Bourgoin
- Date Created : 2024-12-29
- Additional Information : For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Description
A classification pipeline using HistGradientBoostingClassifier to predict salary class based on census data

## Intended Use
The model is trained to be used for predicting salary class based on census data in the context of a Udacity project for the Machine Learning Engineer DevOps Nanodegree program.

## Dataset Details
- **Name:** Census Income Dataset
- **Source:** https://archive.ics.uci.edu/dataset/20/census+income
- **Size:** 32,561 rows
- **Columns:** 15 columns
- **Target:** salary
- **Features:** 14 features
- **Categorical Features:** 9 categorical features
- **Numerical Features:** 6 numerical features
- **Preprocessing Steps:** Categorical features encoded using OneHotEncoder, the target feature is encoded using LabelBinarizer.

## Training Details
- **Algorithm:** HistGradientBoostingClassifier
- **Hyperparameters:**   
  • "loss": log_loss,
  • "learning_rate": 0.2,
  • "max_iter": 100,
  • "max_depth": 5, 
  • "min_samples_leaf": 10,
  • "max_leaf_nodes": 31,
  • "max_features": 1.0,
  • "l2_regularization": 1.0,
  • "max_bins": 255,
- **Validation Strategy:** 80-20 train-test split with 5-fold stratified cross-validation.
- **Hyperparameter Tuning:** The hyperparameters were tuned using RandomizedSearchCV with 5-fold stratified cross-validation and only 100 iterations.
- **Slice Analysis:** It's possible to evaluate also the model performance on different slices of the data. For example for the following features: education, age, occupation, race, sex, and native-country.

## Performance
- **Precision:** 0.7893
- **Recall:** 0.676
- **Fbeta:** 0.7283
- **Test Set Performance:** Metrics were evaluated on a hold-out test set of 20% of the data.

## Limitations
- Data came from an old census survey, so it's not representative of the actual population
- The model may not perform well on data distributions that differ significantly from the training set.
- Some categorical features might be underrepresented in the dataset.
- The trained model is not a production ready model, it's a proof of concept.

## Ethical Considerations
- The dataset should not be used for any other purpose than the one specified in the intended use.
- Bias in the dataset could lead to unfair predictions for certain demographic groups.