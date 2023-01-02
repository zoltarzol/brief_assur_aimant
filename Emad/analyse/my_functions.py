from matplotlib.cbook import boxplot_stats

def get_outliers(data):
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
        