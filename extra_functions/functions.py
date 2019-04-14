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
