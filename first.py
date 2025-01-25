import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load datasets
medal_counts = pd.read_csv('/data/2025_Problem_C_Data/summerOly_medal_counts.csv', encoding='latin1')
hosts = pd.read_csv('data/2025_Problem_C_Data/summerOly_hosts.csv', encoding='latin1')
programs = pd.read_csv('data/2025_Problem_C_Data/summerOly_programs.csv', encoding='latin1')
athletes = pd.read_csv('data/2025_Problem_C_Data/summerOly_athletes.csv', encoding='latin1')

# Data exploration
def preview_data():
    print("Medal Counts Dataset:\n", medal_counts.head())
    print("Hosts Dataset:\n", hosts.head())
    print("Programs Dataset:\n", programs.head())
    print("Athletes Dataset:\n", athletes.head())

preview_data()

#Data preprocessing
# Extract relevant features from medal_counts and hosts
medal_counts['Host'] = medal_counts['Year'].map(
    hosts.set_index('Year')['Host'].to_dict()
)

# Fill NaN with 0 for medals
medal_counts.fillna(0, inplace=True)

# Compute historical medal trends for each country
historical_medals = medal_counts.groupby('NOC')[['Gold', 'Silver', 'Bronze', 'Total']].mean().reset_index()
historical_medals.rename(columns={
    'Gold': 'HistoricalGold',
    'Silver': 'HistoricalSilver',
    'Bronze': 'HistoricalBronze',
    'Total': 'HistoricalTotal'
}, inplace=True)

# Merge historical data
medal_counts = medal_counts.merge(historical_medals, on='NOC', how='left')

# Add host effect as binary feature
medal_counts['IsHost'] = medal_counts['NOC'] == medal_counts['Host']

# Prepare features and labels for modeling
features = ['HistoricalGold', 'HistoricalSilver', 'HistoricalBronze', 'HistoricalTotal', 'IsHost']
labels_gold = medal_counts['Gold']
labels_total = medal_counts['Total']
# Train-test split
X_train, X_test, y_train_gold, y_test_gold = train_test_split(X, labels_gold, test_size=0.2, random_state=42)
X_train, X_test, y_train_total, y_test_total = train_test_split(X, labels_total, test_size=0.2, random_state=42)

# Model training
rf_gold = RandomForestRegressor(random_state=42)
rf_gold.fit(X_train, y_train_gold)

rf_total = RandomForestRegressor(random_state=42)
rf_total.fit(X_train, y_train_total)

# Model evaluation
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MSE: {mse}, MAE: {mae}")
    return predictions

gold_predictions = evaluate_model(rf_gold, X_test, y_test_gold)
total_predictions = evaluate_model(rf_total, X_test, y_test_total)
# Visualization: Actual vs Predicted
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test_gold, y=gold_predictions, color='blue', label='Gold Medals')
plt.plot([0, max(y_test_gold)], [0, max(gold_predictions)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('Actual Gold Medals')
plt.ylabel('Predicted Gold Medals')
plt.title('Actual vs Predicted Gold Medals')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))

sns.scatterplot(x=y_test_total, y=total_predictions, color='green', label='Total Medals')
plt.plot([0, max(y_test_total)], [0, max(total_predictions)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('Actual Total Medals')
plt.ylabel('Predicted Total Medals')
plt.title('Actual vs Predicted Total Medals')
plt.legend()
plt.show()

#Feature importance
importance_gold = rf_gold.feature_importances_
importance_total = rf_total.feature_importances_

plt.figure(figsize=(10, 5))
plt.bar(features, importance_gold, color='blue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance for Gold Medals Prediction')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(features, importance_total, color='green')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance for Total Medals Prediction')
plt.show()