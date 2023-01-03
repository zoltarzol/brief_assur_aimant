from matplotlib.cbook import boxplot_stats
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro 
from scipy.stats import kstest
import pandas as pd


def get_outliers_length(data):
    # Extract the outlier values from the data
    for i in data.columns:
        if data[i].dtype in ["int64", "float64"]:
            outliers = [y for stat in boxplot_stats(data[i]) for y in stat['fliers']]
            if len(outliers) > 0 :
                print(f'Variable {i} has {len(outliers)} outliers')
            elif len(outliers) == 0 :
                print(f'Variable {i} has no outliers')
            else:
                print("Error!!!!!")
                break
        

def get_outliers_by_column(column):
  if column.dtype in ["int64", "float64"]:
          return [y for stat in boxplot_stats(column) for y in stat['fliers']]
  else:
    print("Error, data must be int or float")




def boxplot_all_numeric_columns(data):
    # Select only the numeric columns
    numeric_data = data.select_dtypes(include=['int64', 'float64'])

    names = list(numeric_data.columns)
    f, axes = plt.subplots(round(len(names)/2), 2,  figsize=(14, 4))  
    if numeric_data.shape[1]%2 == 0 :
        n_lignes = int(numeric_data.shape[1]/2) 
    else : 
        n_lignes = int(numeric_data.shape[1]/2+1) 
    y = 0;
    for name in names:
        i, j = divmod(y, n_lignes)
        sns.boxplot(x=numeric_data[name], ax=axes[i, j])
        y = y + 1

    plt.tight_layout()
    plt.show()



