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



import scipy.stats as stats

def correlation_test(data, alpha):
  # Get the column names
  cols = data.columns
  
  # Iterate over all pairs of columns
  for i in range(len(cols)):
    for j in range(i+1, len(cols)):
      # Get the column pair
      col1 = cols[i]
      col2 = cols[j]
      
      # Get the correlation and p-value
      corr, p_value = stats.pearsonr(data[col1], data[col2])
      
      # Print the interpretation of the correlation
      if corr > 0:
        print(f"There is a positive correlation between {col1} and {col2}")
      elif corr < 0:
        print(f"There is a negative correlation between {col1} and {col2}")
      else:
        print(f"There is no correlation between {col1} and {col2}")
      
      if abs(corr) > 0.9:
        print(f"This is a strong correlation (r = {round(corr,4)}).")
      elif abs(corr) > 0.7:
        print(f"This is a moderate correlation (r = {round(corr,4)}).")
      elif abs(corr) > 0.5:
        print(f"This is a weak correlation (r = {round(corr,4)}).")
      else:
        print(f"This is a very weak correlation (r = {round(corr,4)}).")
      
      # Print the interpretation of the p-value
      if p_value < alpha:
        print(f"There is a statistically significant association between {col1} and {col2} (p-value = {p_value:.4f})")
      else:
        print(f"There is no statistically significant association between {col1} and {col2} (p-value = {p_value:.4f})")
      
      print()
        



def make_jointplot(data, column1, column2, hue, kind ):
    # Create a jointplot
    plot = sns.jointplot(data=data, x=column1, y=column2, kind=kind, hue = hue)
    
    # Calculate the Pearson correlation coefficient and p value
    corr, p = stats.pearsonr(data[column1], data[column2])
    
    # Update the title with the correlation and p value
    plot.fig.suptitle(f"Correlation: {corr:.2f}, p-value: {p:.2e}")
    
    # Show the plot
    plt.show()
    

  
def calc_corr_pvalue(data, var1, var2, group_var):
  # Group the data by the group variable
  groups = data.groupby(group_var)
  
  # Create an empty list to store the results
  results = []
  
  # Iterate over the groups
  for name, group in groups:
    # Calculate the Pearson correlation coefficient and p value
    corr, p = stats.pearsonr(group[var1], group[var2])
    
    # Store the results in a tuple
    results.append((name, corr, p))
    
  # Create a DataFrame from the results
  df = pd.DataFrame(results, columns=[group_var, "correlation", "p_value"])
  
  return df









def calc_corr_pvalue(data, var1, var2, group_var, alpha):
  # Group the data by the group variable
  groups = data.groupby(group_var)
  
  # Create an empty list to store the results
  results = []
  
  # Iterate over the groups
  for name, group in groups:
    # Calculate the Pearson correlation coefficient and p value
    corr, p = stats.pearsonr(group[var1], group[var2])
    
    # Store the results in a tuple
    results.append((name, corr, p))
    
  # Create a DataFrame from the results
  df = pd.DataFrame(results, columns=[group_var, "correlation", "p_value"])
  
  # Create the title string
  title = f"Correlation between {var1} and {var2} grouped by {group_var}:"
  border = "-" * len(title)
  title_str = f"\n{border}\n{title}\n{border}\n"
  
  # Print the title
  print(title_str)

  print("")
  print(df.to_string(index=False))
  print("")
  # Iterate over the rows of the DataFrame
  for i, row in df.iterrows():
    # Get the p value
    p = row["p_value"]
    
    # Print the interpretation of the p value
    if p < alpha:
      print(f"{row[group_var]}: There is a statistically significant association (p-value = {p:.4f})")
    else:
      print(f"{row[group_var]}: There is no statistically significant association (p-value = {p:.4f})")