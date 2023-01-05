from matplotlib.cbook import boxplot_stats
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro 
from scipy.stats import kstest
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols



def get_outliers_length(data):
    """
    Get the outlier values in a numeric column.
    
    Parameters:
    column (pandas Series): The column to use for the calculations.
    
    Returns:
    list: A list of outlier values.
    """
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

  """
  Get the outlier values in a numeric column.
  
  Parameters:
  column (pandas Series): The column to use for the calculations.
  
  Returns:
  list: A list of outlier values.
  """
  if column.dtype in ["int64", "float64"]:
          return [y for stat in boxplot_stats(column) for y in stat['fliers']]
  else:
    print("Error, data must be int or float")




def boxplot_all_numeric_columns(data):
    """
    Create boxplots for all numeric columns in a DataFrame.
    
    Parameters:
    data (pandas DataFrame): The data to use for the plots.
    
    Returns:
    None: The function creates plots using Matplotlib.
    """

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




def correlation_test(data, alpha):

  """
  Test for correlations and statistical significance between all pairs of columns in a DataFrame.
  
  Parameters:
  data (pandas DataFrame): The data to use for the test.
  alpha (float): The significance level to use for the hypothesis test.
  
  Returns:
  None: The function prints the results to the console.
  """
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

    """
    Create a joint plot showing the relationship between two variables and the distribution of each.
    
    Parameters:
    data (pandas DataFrame): The data to use for the plot.
    column1 (str): The name of the first variable.
    column2 (str): The name of the second variable.
    hue (str): The name of the variable to use for the color of the points in the plot.
    kind (str): The type of plot to create. Can be "scatter", "reg", "resid", "kde", or "hex".
    
    Returns:
    None: The function creates a plot using Matplotlib.
    """

    # Create a jointplot
    plot = sns.jointplot(data=data, x=column1, y=column2, kind=kind, hue = hue)
    
    # Calculate the Pearson correlation coefficient and p value
    corr, p = stats.pearsonr(data[column1], data[column2])
    
    # Update the title with the correlation and p value
    plot.fig.suptitle(f"Correlation: {corr:.2f}, p-value: {p:.2e}")
    
    # Show the plot
    plt.show()
    

  










def calc_corr_pvalue(data, var1, var2, group_var, alpha):

  """
  Calculate the Pearson correlation coefficient and p value for the correlation between two variables, grouped by a third variable.
  
  Parameters:
  data (pandas DataFrame): The data to use for the calculations.
  var1 (str): The name of the first variable.
  var2 (str): The name of the second variable.
  group_var (str): The name of the variable to group the data by.
  alpha (float): The significance level to use for the hypothesis test.
  
  Returns:
  None: The function prints the results to the console.
  """

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










def point_biserial_correlation(data: pd.DataFrame, x_col: str, y_col: str, alpha: float = 0.05) -> None:
    """
    Calculate the point biserial correlation coefficient between two variables in a Pandas DataFrame.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the two variables.
    x_col (str): The name of the continuous variable.
    y_col (str): The name of the dichotomous variable.
    alpha (float): The significance level (default is 0.05).
    
    Returns:
    None

    Example:
    >>> data = pd.DataFrame({'charges': [1000, 2000, 3000, 4000, 5000], 'sex': ['male', 'male', 'female', 'female', 'male']})
    >>> point_biserial_correlation(data, 'charges', 'sex', alpha=0.05)
    ----------------------------
    | Correlation between "charges" and "sex" |
    ----------------------------
    There is a significant difference in the means of the continuous variable between the two groups defined by the dichotomous variable.
    
    """
    # Extract the variables from the DataFrame
    x = data[x_col]
    y = data[y_col]
    
    # Convert the dichotomous variable to a list of 0s and 1s
    y = [0 if yi == y.unique()[0] else 1 for yi in y]
    
    # Calculate the point biserial correlation coefficient and p-value
    rpb, p_value = stats.pointbiserialr(x, y)
    
    # Split the continuous variable data into two groups based on the dichotomous variable
    x_group1 = [x[i] for i in range(len(y)) if y[i] == 0]
    x_group2 = [x[i] for i in range(len(y)) if y[i] == 1]
    
    # Perform the t-test
    t, p_value_ttest = stats.ttest_ind(x_group1, x_group2)
    
    # Print the title in a boxed format
    title = f'Correlation between "{x_col}" and "{y_col}"'
    print('-' * (len(title) + 4))
    print(f'| {title} |')
    print('-' * (len(title) + 4))
    print(f'Point biserial correlation coefficient: {rpb:.3f}')
    print(f't-value: {t:.3f}')
    print(f'p-value: {p_value:.3f}')
    
    # Print the interpretation of the p-value
    if p_value_ttest < alpha:
        print('There is a significant difference in the means of the continuous variable between the two groups defined by the dichotomous variable.')
    else:
        print('There is not a significant difference in the means of the continuous variable between the two groups defined by the dichotomous variable.')















def one_way_anova(data: pd.DataFrame, y_col: str, x_col: str, alpha: float = 0.05) -> None:
    """
    Perform a one-way ANOVA on two variables in a Pandas DataFrame.
    Null hypothesis: Groups means are equal (no variation in means of groups)
    H0: μ1=μ2=…=μp
    Alternative hypothesis: At least, one group mean is different from other groups
    H1: All μ are not equal
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the two variables.
    y_col (str): The name of the dependent variable.
    x_col (str): The name of the independent variable.
    alpha (float): The significance level (default is 0.05).
    
    Returns:
    None

    Example:
    >>> one_way_anova(data, 'charges', 'children', alpha=0.05)
    """
      # Assert that the input is a Pandas DataFrame
    assert isinstance(data, pd.DataFrame), 'data must be a Pandas DataFrame'
    
    # Assert that the dependent and independent variables are columns in the DataFrame
    assert y_col in data.columns, f'{y_col} is not a column in the DataFrame'
    assert x_col in data.columns, f'{x_col} is not a column in the DataFrame'
    
    # Assert that the independent variable has more than two levels
    assert len(data[x_col].unique()) > 2, 'the independent variable must have more than two levels'
    
    # Assert that the significance level is between 0 and 1
    assert 0 < alpha < 1, 'alpha must be between 0 and 1'
    


    # Extract the dependent and independent variables
    y = data[y_col]
    x = data[x_col]
    
    # Split the data into separate arrays for each level of the independent variable
    groups = []
    levels = x.unique()
    for level in levels:
        group = y[x == level]
        groups.append(group)
    
    # Perform the one-way ANOVA
    f_value, p_value = stats.f_oneway(*groups)
    
    # Print the title in a boxed format
    title = f'ANOVA between "{y_col}" and "{x_col}"'
    print('-' * (len(title) + 4))
    print(f'| {title} |')
    print('-' * (len(title) + 4))
    
    # Print the results
    print(f'F-value: {f_value:.3f}')
    print(f'p-value: {p_value:.3f}')
    
    
    # Perform the one-way ANOVA
    model = ols('y ~ C(x)', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    print()


    """
    The function below was created specifically for the one-way ANOVA table results returned for Type II sum of squares
    """
    def anova_table_complet(anova_table):
        anova_table['mean_sq'] = anova_table[:]['sum_sq']/anova_table[:]['df']

        anova_table['eta_sq'] = anova_table[:-1]['sum_sq']/sum(anova_table['sum_sq'])

        anova_table['omega_sq'] = (anova_table[:-1]['sum_sq']-(anova_table[:-1]['df']*anova_table['mean_sq'][-1]))/(sum(anova_table['sum_sq'])+anova_table['mean_sq'][-1])

        cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
        anova_table = anova_table[cols]
        return anova_table

    print((anova_table_complet(anova_table)))

    # Print the interpretation of the p-value
    if p_value < alpha:
        print(f'The p-value of {p_value:.3f} is statistically significant at a level of {alpha}.')
        print(f'This suggests that there is a significant differences among the independent variable "{x_col}".')
        print(f"But we don't know which group is different from which.\nWe have to do post-hoc analysis using Tukey HSD (Honest Significant Difference) Test.")
        print()
        from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
        # compare the height between each diet, using 95% confidence interval 
        mc = MultiComparison(y, x)
        tukey_result = mc.tukeyhsd(alpha=0.05)

        print(tukey_result)
        print(f'Unique "{x_col}" groups: {mc.groupsunique}')
        print()
        print("Reject : True means there is statistically significant difference.")
    else:
        print(f'The p-value of {p_value:.3f} is not statistically significant at a level of {alpha}.')
        print(f'This suggests that there is not a significant differences among the independent variable "{x_col}".')
    
