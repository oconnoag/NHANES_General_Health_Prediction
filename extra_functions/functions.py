import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
