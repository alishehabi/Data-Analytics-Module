# Importing All Required Libraries

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Suppression of warning messages
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

%matplotlib inline

# Read the CSV file 'fifa_players.csv', load the data set into a dataFrame
df = pd.read_csv('fifa_players.csv')

# Print the shape of the DataFrame
print("Shape of the data= ", df.shape)

# Display the first five rows
df.head()
df.columns # Listing the names of all the columns in the dataset
df.dtypes

corr=df.corr() # gives us the correlation values
plt.figure(figsize=(15,6))
sns.heatmap(corr, annot = True, cmap="BuPu")  # let's visualise the correlation matrix
plt.show()

corr = df.corr()

# Isolate the 'value_euro' column
value_euro_corr = corr[['value_euro']]

# Plot the correlations with 'value_euro'
plt.figure(figsize=(10, 12))
sns.heatmap(value_euro_corr.sort_values(by='value_euro', ascending=False),
            annot=True,
            cmap="BuPu",
            cbar=True,
            vmin=-1,
            vmax=1)
plt.show()

df['physical'] = (df["strength"] + df["sprint_speed"] + df["agility"] + df["reactions"] + df["stamina"] +
                   df["jumping"] + df["acceleration"] + df["aggression"])/8

df['defence_rating'] = (df["sliding_tackle"] + df["standing_tackle"] + df["interceptions"] + df["marking"] +
                          df["positioning"])/5

df["skills"] = (df["ball_control"] + df["short_passing"] + df["long_passing"] + df["composure"] +
                          df["vision"] + df["dribbling"] + df["balance"])/7

df["attack_rating"] = (df["crossing"] + df["finishing"] + df["long_shots"] + df["volleys"] +
                           df["heading_accuracy"] + df['curve']+ df['freekick_accuracy']+ df['shot_power']+ df['penalties'])/9

plt.hist(df["value_euro"], bins=20, alpha=0.7, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Value in Euros')
plt.ylabel('Frequency')
plt.title('Distribution of Player Values in Euros')

# Applying a logarithmic transformation to 'value_euro' column
# Ensure there are no zero or negative values in 'value_euro' before this transformation
df["value_euro"] = np.log(df["value_euro"])

# Creating a histogram for the 'value_euro' column with customized bin colors
plt.figure(figsize=(14, 6))

# Histogram
plt.subplot(1, 3, 1)
plt.hist(df["value_euro"], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Log of Value in Euros')
plt.ylabel('Frequency')
plt.title('Histogram of Log-Transformed Player Values in Euros')

# Box Plot
plt.subplot(1, 3, 2)
plt.boxplot(df["value_euro"], vert=True, patch_artist=True, notch=True, widths=0.6)
plt.xticks([1], ['Value in Euros'])
plt.title('Box Plot of Log-Transformed Player Values in Euros')

# Density Plot
plt.subplot(1, 3, 3)
sns.kdeplot(df["value_euro"], fill=True, color="r")
plt.xlabel('Log of Value in Euros')
plt.ylabel('Density')
plt.title('Density Plot of Log-Transformed Player Values in Euros')

plt.tight_layout()
plt.show()

# Section 5: Data Cleaning - Dropping More Columns & Handling Missing Values
# Combine the two lists of columns to be removed
df = df[~df['positions'].str.contains('GK', na=False)]

columns_remove = [
    'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys',
    'dribbling', 'curve', 'freekick_accuracy', 'long_passing', 'ball_control',
    'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power',
    'jumping', 'stamina', 'strength', 'long_shots', 'aggression', 'interceptions',
    'positioning', 'vision', 'penalties', 'composure', 'marking', 'standing_tackle',
    'sliding_tackle','name', 'full_name', 'birth_date', 'nationality',
    'preferred_foot', 'body_type', "release_clause_euro", "national_team","national_team_position", "national_jersey_number",
    "body_type", "skill_moves(1-5)", "positions", 'height_cm', 'weight_kgs', 'age', 'national_rating'
]

# Now you can use this combined, deduplicated list to drop columns from your DataFrame
df = df.drop(columns=columns_remove)

# Further filtering to remove goalkeepers from the DataFrame, assuming 'GK' identifies goalkeepers in the 'positions' column

# Display the first few rows of the DataFrame without the specified columns and goalkeepers
df.head()

corr=df.corr() # gives us the correlation values
plt.figure(figsize=(15,6))
sns.heatmap(corr, annot = True, cmap="BuPu")  # let's visualise the correlation matrix
plt.show()

corr = df.corr()

# Isolate the 'value_euro' column
value_euro_corr = corr[['value_euro']]

# Plot the correlations with 'value_euro'
plt.figure(figsize=(6, 6))
sns.heatmap(value_euro_corr.sort_values(by='value_euro', ascending=False),
            annot=True,
            cmap="BuPu",
            cbar=True,
            vmin=-1,
            vmax=1)
plt.show()

df.isnull().sum()
# Fill missing values with the mean
df['value_euro'].fillna(df['value_euro'].mean(), inplace=True)
df['wage_euro'].fillna(df['wage_euro'].mean(), inplace=True)

# Verify that the null values have been filled
df.isnull().sum()
df.shape
df.describe()

# Section 4: Visualizing Relationships
figure, axis = plt.subplots(2, 2, figsize=(12, 8))
skills = ['skills', 'defence_rating', 'attack_rating', 'physical']
titles = ['Football skill vs Player value, euros', 'Defensive skill vs Player value, euros',
          'Offensive skills vs Player value, euros', 'Physical ability vs Player value, euros']

for i, ax in enumerate(axis.flat):
    ax.scatter(df[skills[i]], df["value_euro"], marker='x')
    ax.set_title(titles[i])
    ax.set_xlabel(skills[i])  # Set the label for the x-axis
    ax.set_ylabel('Player value, euros')  # Set the label for the y-axis

plt.tight_layout()
plt.show()

X = df.drop('value_euro', axis=1)
y = df['value_euro']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Adjust n_components as needed to retain 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Print the number of components
print("Original feature set size:", X.shape[1])
print("Reduced feature set size:", X_pca.shape[1])

y = np.log(df['value_euro'])  # Applying log transformation to the target variable to normalize its distribution

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

X_train.shape
y_train.shape
X_test.shape
y_test.shape

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Number of observations and number of features
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)
n_train = X_train.shape[0]
n_test = X_test.shape[0]
p = X_train.shape[1]

# Calculate and print the metrics for the training set
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
sse_train = np.sum((y_train - y_train_pred) ** 2)
r2_train = r2_score(y_train, y_train_pred)
adjusted_r2_train = 1 - (1-r2_train) * (n_train - 1) / (n_train - p - 1)

print(f"Training MSE: {mse_train}")
print(f"Training MAE: {mae_train}")
print(f"Training SSE: {sse_train}")
print(f"Training R-squared: {r2_train}")
print(f"Training Adjusted R-squared: {adjusted_r2_train} \n")

# Calculate and print the metrics for the test set
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
sse_test = np.sum((y_test - y_test_pred) ** 2)
r2_test = r2_score(y_test, y_test_pred)
adjusted_r2_test = 1 - (1-r2_test) * (n_test - 1) / (n_test - p - 1)

print(f"Test MSE: {mse_test}")
print(f"Test MAE: {mae_test}")
print(f"Test SSE: {sse_test}")
print(f"Test R-squared: {r2_test}")
print(f"Test Adjusted R-squared: {adjusted_r2_test}")

def train_and_evaluate(model, X, y):
    """
    Train and evaluate a regression model using both a single train/test split
    and cross-validation to compare performances, including evaluations on
    both training and test sets for R^2, MSE, and MAE.
    """
    print('Evaluating model:', model.__class__.__name__)

    # Perform a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Cross-validation for model evaluation
    cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_scores_mse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores_mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

    print('Cross-Validation Mean R^2:', np.mean(cv_scores_r2), 'with SD:', np.std(cv_scores_r2))
    print('Cross-Validation Mean MSE:', -np.mean(cv_scores_mse), 'with SD:', np.std(cv_scores_mse))
    print('Cross-Validation Mean MAE:', -np.mean(cv_scores_mae), 'with SD:', np.std(cv_scores_mae))

# Example usage of the function with Linear Regression
train_and_evaluate(LinearRegression(), X_pca, y)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
# Initialize and fit the LassoCV model
lasso_cv = LassoCV(alphas=None, cv=10, max_iter=10000)
lasso_cv.fit(X_train, y_train)

# Predictions
y_train_pred_lasso_cv = lasso_cv.predict(X_train)
y_test_pred_lasso_cv = lasso_cv.predict(X_test)

# Optimal alpha
print("Optimal alpha for LassoCV: ", lasso_cv.alpha_)

# R^2 scores
lasso_cv_train_score = lasso_cv.score(X_train, y_train)
lasso_cv_test_score = lasso_cv.score(X_test, y_test)

print(f"LassoCV Training Score (R^2): {lasso_cv_train_score}")
print(f"LassoCV Test Score (R^2): {lasso_cv_test_score}")

# MSE calculations
mse_train = mean_squared_error(y_train, y_train_pred_lasso_cv)
mse_test = mean_squared_error(y_test, y_test_pred_lasso_cv)

print(f"LassoCV Training MSE: {mse_train}")
print(f"LassoCV Test MSE: {mse_test}")

# Define a set of alphas to consider
alphas = np.logspace(-6, 6, 13)

# Initialize and fit the RidgeCV model
ridge_cv = RidgeCV(alphas=alphas, cv=10)
ridge_cv.fit(X_train, y_train)

# Predictions for training and testing sets
y_train_pred_ridge_cv = ridge_cv.predict(X_train)
y_test_pred_ridge_cv = ridge_cv.predict(X_test)

# Optimal alpha value
print("Optimal alpha for RidgeCV: ", ridge_cv.alpha_)

# R^2 scores for training and testing sets
ridge_cv_train_score = ridge_cv.score(X_train, y_train)
ridge_cv_test_score = ridge_cv.score(X_test, y_test)
print(f"RidgeCV Training Score (R^2): {ridge_cv_train_score}")
print(f"RidgeCV Test Score (R^2): {ridge_cv_test_score}")

# Calculating and printing the MSE for the training and testing sets
mse_train = mean_squared_error(y_train, y_train_pred_ridge_cv)
mse_test = mean_squared_error(y_test, y_test_pred_ridge_cv)
print(f"RidgeCV Training MSE: {mse_train}")
print(f"RidgeCV Test MSE: {mse_test}")


plt.figure(figsize=(8,15))

# Linear Regression plot
plt.subplot(3, 1, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs. Predicted Player Values (Test Set)')
plt.xlabel('Actual Values (Euro)')
plt.ylabel('Predicted Values (Euro)')

# LassoCV plot - regularisation 1
plt.subplot(3, 1, 2)
plt.scatter(y_test, y_test_pred_lasso_cv, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('LassoCV\nActual vs. Predicted Player Values')
plt.xlabel('Actual Values (Euro)')
plt.ylabel('Predicted Values (Euro)')

# RidgeCV plot - regularisation 2
plt.subplot(3, 1, 3)
plt.scatter(y_test, y_test_pred_ridge_cv, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('RidgeCV\nActual vs. Predicted Player Values')
plt.xlabel('Actual Values (Euro)')
plt.ylabel('Predicted Values (Euro)')

plt.tight_layout()
plt.show()

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define the parameter grid to search
param_grid = {
    'max_depth': range(1, 20),
    'min_samples_leaf': range(1, 20)
}

# Instantiate the regressor
dt_reg = DecisionTreeRegressor(random_state=0)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=dt_reg, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print(f"Best parameters: {grid_search.best_params_}")

# Initializing the Decision Tree Regressor with a random state for reproducibility
decision_tree = DecisionTreeRegressor(
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    random_state=0
)

# Training the model on the training set
decision_tree.fit(X_train, y_train)

# Predicting the target variable for the training and test sets
y_train_pred = decision_tree.predict(X_train)
y_test_pred = decision_tree.predict(X_test)

# Calculating the R-squared value and Mean Squared Error for both training and test sets
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Training R-squared (Accuracy): {train_r2}")
print(f"Test R-squared (Accuracy): {test_r2}")
print(f"Training Mean Squared Error: {train_mse}")
print(f"Test Mean Squared Error: {test_mse}")

# Plotting the tree (with limited depth for readability)
plt.figure(figsize=(20,10))
plot_tree(decision_tree, max_depth=3, feature_names=X.columns, filled=True, fontsize=12)
plt.show()

# Assuming 'X_pca' and 'y' are your PCA-transformed features and target variable
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Instantiate the regressor
rf_reg = RandomForestRegressor(random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print(f"Best parameters: {grid_search.best_params_}")

# Initializing the RandomForestRegressor with best parameters found
random_forest = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    random_state=42
)

# Training the model on the training set
random_forest.fit(X_train, y_train)

# Predicting the target variable for the training and test sets
y_train_pred = random_forest.predict(X_train)
y_test_pred = random_forest.predict(X_test)

# Calculating the R-squared value and Mean Squared Error for both training and test sets
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Training R-squared (Accuracy): {train_r2}")
print(f"Test R-squared (Accuracy): {test_r2}")
print(f"Training Mean Squared Error: {train_mse}")
print(f"Test Mean Squared Error: {test_mse}")
