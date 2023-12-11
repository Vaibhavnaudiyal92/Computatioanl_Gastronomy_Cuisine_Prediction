# Computational_Gastronomy_Cuisine_Prediction

# Introduction
RecipeDB is one of the popular datasets developed by Complex Systems Laboratory which contains details about roughly 50,000 recipes from all around the world in well documented manner. One of the complex problems that can be solved using this data is of Cuisine Prediction. Can a machine learning model be developed which can accurately predict the Cuisine to which a recipe belongs? In this project we will be trying to do exactly that! The dataset used is a part of RecipeDB dataset along with standard machine learning and mathematical libraries like sklearn, numpy, pandas, etc.

# Data Engineering
Here in this section, we will try to clean and process RecipeDB Five Cuisine dataset along with RecipeDB ingredient phrase dataset to create data for our prediction model.
For this, we will be going throught standard data processing techniques to check for data consistency.


1.   Check for null values in columns
2.   Check for non-related features

As part of data augmentation, we will be merging the two datasets. The merge will be done on the basis of Recipe ID under **inner** join process.

### Pivoting the dataset
For creating a pivoted dataset, we will have to drop all the columns except **Recipe_id** and **ingredient**. Due to One-Many relation of Recipe and ingredients, a single Recipe_id has many ingredients mapped to it. This is not the right way to feed data into training the machine learning model. Hence, we will be pivoting the data so that each ingredient becomes a column and acts as an one hot encoded feature, where values will be stored in binary format.

## Applying TF-IDF

We will be exploring TF-IDF approach as this can be mapped to Natural Language Processing (NLP) problem. Instead of words, we will be applying it to list of ingredients. TF-IDF will assign importance to set of most important ingredients which in turn can help the model to predict the Region.

# Machine Learning Model

In this section, we will be applying different sets of machine learning models on the prepared dataset. On the basis of validation, best working model will be selected which in turn will be fine tuned to give perfect results.
### Models Applied
1.  Random Forest
2.  Decision Trees
3.  Gradient Boosting
4.  Logistic Regression
5.  Naive Bayes

### Applying SMOTE
In the last part of pre-processing, we will be apply **SMOTE** which is a sampling technique. Inorder to remove any kind of bias or discrepency in the model training, we will be sampling the data on the basis of target variable so that equal number of examples are generated for each class of target variable.

# Conclusion
With hyperparameter tuning of n_estimators of Random Forest, we got a mean cross-validation score of **85.5%** which is **0.5%** better than base Random Forest model. Ensemble and Boosting algorithm seemed to outperform linear, probabilistic and tree based models. This is again confirmed by Precision and Recall curves.
