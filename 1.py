import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data from Excel file
df = pd.read_excel('1.xlsx', engine='openpyxl', skiprows=1)
#print(df.columns)
# Replace missing values with the mean
df.fillna(df.mean(), inplace=True)

# Split data into features and target
X = df[['Lх_мм']]
y = df['Нагр_кг']

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict the target values
y_pred = model.predict(X)

# Print the predicted values
print(y_pred)
