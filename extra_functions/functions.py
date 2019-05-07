import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import scale
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (precision_score, balanced_accuracy_score, 
                             classification_report, make_scorer, confusion_matrix)


def general_health_reclasser(x):
    """
    Used to reclassify the original levels of general health in the NHANES Data:
    
    Levels 1-2: 1
             3: 2
           4-5: 3
           nan: nan
           
    """
    if (x == 1 or x == 2):
        return 1
    
    if (x == 3):
        return 2

    if (x == 4 or x == 5):
        return 3

    # Keeps nans
    return x


def visualize(data, x, y):
    """
	Visualize the relationship of two variables (x and y) in a dataset.
	- Barplot
	- Violinplot
	- Boxplot
	- Stripplot
    """

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    sns.stripplot(x, y, data=data, jitter=True, size=4, ax=axes[0][0])
    sns.factorplot(x=x, y=y, data=data, kind='bar', ax=axes[0][1], sharex=False, sharey=False, legend=False)
    data.boxplot(column=y, by=x, ax=axes[1][0])
    sns.violinplot(x=x, y=y, data=data, ax=axes[1][1])
    
    # Remove the extra plot that factorplot draws
    plt.close(2)
    plt.show()
    
    
def visualize_categorical(data, x, y):
    """
	Visualize the relationship of two variables (x and y) in a dataset.
	- Barplot
	- Violinplot
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
    sns.factorplot(x=x, y=y, data=data, kind='bar', ax=axes[0], sharex=False, sharey=False, legend=False)
    sns.violinplot(x=x, y=y, data=data, ax=axes[1])
    
    # Remove the extra plot that factorplot draws
    plt.close(2)
    plt.show()

    

def kde_by_category(data, category, kde_column):
    """
    Plots a kde of a particular column of data (kde_column), but split by another column (category).
    Both data must come from one Pandas DataFrame (data)
    """
    plt.figure(figsize=(14,4))
    
    cats = sorted(data[category].unique())
    
    for cat in cats:
        sns.distplot(data[data[category]==cat][kde_column], hist=False, label=cat)
        
    plt.title('KDE of ' + kde_column + ' split by ' + category + ' levels')
    
    
def print_metrics(model, x_test, y_test, y_pred):
    """
    Prints the score of the model, the confusion matrix, and the classification report.
    """
    print("Score of the model is", model.score(x_test, y_test))

    print()
    print("Confusion Matrix: \n")
    print(sklearn.metrics.confusion_matrix(y_test, y_pred))
    print("\n----------------------------------------------------------------\n")
    print("Classification Report: \n")
    print(sklearn.metrics.classification_report(y_test, y_pred))
    
    
def data_splitter(X, y, _test_size=0.20, stratified=True):
    """
    Returns a split dataset from X and y using sklearn's train_test_split (defaul test_size is 20% of input)
        - Options for stratifying the data (True by default)
    """
    
    ############################################
    #### Stratified Data
    ############################################
    if (stratified):
        (X_train, X_test, 
         y_train, y_test) = train_test_split(X, y, test_size=_test_size, stratify=y)                                                   
    else:
        (X_train, X_test, 
         y_train, y_test) = train_test_split(X, y, test_size=_test_size)
    
    return (X_train, X_test, y_train, y_test)


def scaled_data_splitter(X, y, _test_size=0.20, stratified=True):
    """
    Returns a scaled split dataset from X and y using sklearn's train_test_split (default test_size is 20% of input)
        - Options for stratifying the data (True by default)
    """
    
    ############################################
    #### Stratified & Scaled Data
    ############################################
    if (stratified):
        (X_train, X_test, 
         y_train, y_test) = train_test_split(scale(X), y, test_size=_test_size, stratify=y)                                                   
    else:
        (X_train, X_test, 
         y_train, y_test) = train_test_split(scale(X), y, test_size=_test_size)
    
    return (X_train, X_test, y_train, y_test)


def predict_w_treshold(model, X_test, class_num, threshold):
    """
    Allows for a different percentage threshold to be used for classication
    
    Classification is typically based on the the class that has the highest percentage of
    predicted liklihood; however, sometimes this split biases against certain classes.
    """
    
    llph = model.predict_proba(X_test)

    llph_pred = []

    for pcts in llph:
        if pcts[class_num-1] > threshold:
            llph_pred.append(class_num)
        else:
            llph_pred.append(list(pcts).index(max(pcts)) + 1)

    return llph_pred


def grid_search_wrap(clf, gs_params, x_train, y_train, x_test, y_test, 
                     altered_threshold=False, class_num=3, threshold=.25,
                     metrics=True, conf_matrix=False, k_splits=5):
    """
    Uses sklearn's GridSearch to optimize hyperparamters of the inputted model (clf), based on the balanced accuracy
    score of the model (i.e. the unweighted class average for accuracy).  Uses a default Stratified 5-fold CV
    method for evaluating the performance of the model.
    
    If metrics is not set to false - returns both the balanced accuracy and the (macro-average -- same idea as 
    the balanced accuracy) precision of the best performing model
    
    Otherwise, the grid-searched model is returned
    """
    # Optimize the models for balanced_accuracy_score (the unweighted mean of each class)
    #    This metric allows for unbalanced minority classes to share an equal piece of the pie
    #    in the accuracy calculation
    optimize_for = make_scorer(balanced_accuracy_score)

    # GridSearch
    kfold_cv = StratifiedKFold(n_splits=k_splits)
    gs = GridSearchCV(clf, gs_params, n_jobs=-1, cv=kfold_cv,
                              refit=optimize_for)

    # Fitting/Predicting
    gs.fit(x_train, y_train)
    
    if not altered_threshold:
        gs_pred = gs.predict(x_test)
    else:
        gs_pred = predict_w_treshold(gs, x_test, class_num, threshold)
        
    # Confusion Matrix
    if (conf_matrix):
        print()
        print(confusion_matrix(y_test, gs_pred))
        print()
        
    # Metrics
    if (metrics):
        bal_acc = balanced_accuracy_score(y_test, gs_pred),
        prec = precision_score(y_test, gs_pred, average='macro')
        
        # if conf_matrix is called, then we need this to print out the metrics
        if conf_matrix:
            print(bal_acc, prec)

        return bal_acc, prec
    
    print("Best Parameters:")
    print(gs.best_params_)
    print("\n____________________________________\n")
    print(confusion_matrix(y_test, gs_pred))
    print("\n____________________________________\n")
    print(classification_report(y_test, gs_pred))
    return gs