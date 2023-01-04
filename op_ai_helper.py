
# import necessary libraries
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# load the data 
data = pd.read_csv("dataset_brief_assur_aimant.csv")

def classify_bmi(row):
    if row["bmi"] < 18.5:
        return "underweight"
    elif row["bmi"] < 24.9:
        return "healthy"
    elif row["bmi"] < 29.9:
        return "overweight"
    else:
        return "obese"

data["bmi_class"] = data.apply(classify_bmi, axis=1)

print(data.describe())

# separate the features and target 
X = data.drop("charges", axis=1)
y = data["charges"]

# encode categorical variables 
encoder = LabelEncoder() 
X["sex"] = encoder.fit_transform(X["sex"]) 
X["smoker"] = encoder.fit_transform(X["smoker"]) 
X["region"] = encoder.fit_transform(X["region"]) 
X["bmi_class"] = encoder.fit_transform(X["bmi_class"])

# scale numerical variables 
scaler = StandardScaler() 
X[["age","bmi","children"]] = scaler.fit_transform(X[["age","bmi","children"]]) 

# split the data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Define the model pipeline 
pipe = Pipeline([("model", LinearRegression())])

# Define the hyperparameters for GridSearchCV 
grid_params = {'model': [LinearRegression(), Lasso(), Ridge(), ElasticNet()], 'model__normalize': [True, False]}

# Create the GridSearchCV object and fit it to the training data 
gridsearch = GridSearchCV(pipe, grid_params, cv=5)  
gridsearch.fit(X_train, y_train)  

# Print out the best parameters and score from GridSearchCV  
print("Best parameters: {}".format(gridsearch.best_params_))  
print("Best score: {:.2f}".format(gridsearch.best_score_))  

# Define the hyperparameters for RandomizedSearchCV  
randomized_params = {'model': [LinearRegression(), Lasso(), Ridge(), ElasticNet()], 'model__normalize': [True, False]}

# Create the RandomizedSearchCV object and fit it to the training data  
randomizedsearch = RandomizedSearchCV(pipe, randomized_params, cv=5)  
randomizedsearch.fit(X_train, y_train)  

# Print out the best parameters and score from RandomizedSearchCV  
print("Best parameters: {}".format(randomizedsearch.best_params_))  
print("Best score: {:.2f}".format(randomizedsearch.best_score_))  

# Fit the model with the best parameters to the test set  
model = randomizedsearch.best_estimator_.fit(X_test, y_test)  

# Make predictions on the test set and calculate r-squared and RMSE scores  
yhat = model.predict(X_test)  
r2 = r2_score(yhat, y_test)  
rmse = np.sqrt(mean_squared_error(yhat, y_test))

# Print out the results of r-squared and RMSE scores  
print("R-squared score: {:0.2f}".format(r2))  
print("RMSE score: {:0.2f}".format(rmse))

# Visualize the results using seaborn graphs   
sns.scatterplot(x=yhat, y=y_test)
plt.title('Predicted vs Residuals')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()