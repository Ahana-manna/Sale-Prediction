import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv') 

print("Dataset Info:")
print(data.info())
print("\nFirst 5 Rows of the Dataset:")
print(data.head())

data = data.dropna() 

# Convert categorical columns to dummy variables
data_encoded = pd.get_dummies(data, columns=['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category'], drop_first=True)

# Define features (X) and target variable (y)
X = data_encoded.drop(columns=['Sales', 'Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Customer ID', 
                               'Customer Name', 'Country', 'City', 'State', 'Product ID', 'Product Name', 'Postal Code'])
y = data_encoded['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Optional: Visualize Actual vs Predicted Sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

new_data = pd.DataFrame({
    'Region_Central': [0], 
    'Region_East': [1],
    'Segment_Corporate': [0],
    'Segment_Consumer': [1],
    'Sub-Category_Binders': [1],
    'Sub-Category_Labels': [0],
}, index=[0])

new_data = new_data.reindex(columns=X.columns, fill_value=0) 
predicted_sales = model.predict(new_data)
print(f"\nPredicted Sales: {predicted_sales[0]:.2f}")
