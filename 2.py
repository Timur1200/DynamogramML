import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
np.set_printoptions(threshold=np.inf)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#Получить текущий рабочий каталог 
cwd = os.getcwd()

# Получить список всех файлов Excel в текущем каталоге
files = [f for f in os.listdir(cwd) if f.endswith('.xlsx')]

# Initialize an empty DataFrame
df = pd.DataFrame()

# Loop over each file and append its contents to the DataFrame
for file in files:
    data = pd.read_excel(file, header=1, engine='openpyxl')   
    df = df.append(data)
    df.dropna(inplace=True)
# Print the resulting DataFrame
print(df)

X = df[['Lх_мм']]
y = df['Нагр_кг']

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

# Прогнозирование целевых значений
y_pred = model.predict(X)

# Print the predicted values

print(y_pred)
