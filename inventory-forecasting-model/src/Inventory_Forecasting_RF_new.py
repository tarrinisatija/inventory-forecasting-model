import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

#generating the synthetic dataset

num_days = 365 * 3  # 3 years
num_products = 100 #100 products
start_date = pd.to_datetime("2021-01-01")

product_ids = [f"P{str(i).zfill(3)}" for i in range(1, num_products + 1)] #assigning product ids starting from P001
product_names = [f"Product_{i}" for i in range(1, num_products + 1)] #assigning product names starting from Product_1
categories = ["Electronics", "Clothing", "Groceries", "Stationery"]
warehouses = ["W1", "W2", "W3"]
weather_options = ["Sunny", "Rainy", "Cloudy"]

data = []

for day in range(num_days):
    date = start_date + pd.Timedelta(days=day)
    weekday = date.weekday()

    is_holiday = 1 if weekday in [5, 6] or random.random() < 0.05 else 0 #randomly assigning holidays
    weather = random.choice(weather_options)

    for pid, pname in zip(product_ids, product_names):
        category = random.choice(categories)
        warehouse = random.choice(warehouses)
        promotion = round(random.uniform(0, 0.3), 2)  # up to 30% discount(it can be increased or decreased)

        base_demand = np.random.randint(20, 60) if is_holiday else np.random.randint(10, 40)
        fluctuation = np.random.normal(0, 7)
        units_sold = max(0, int(base_demand * (1 + promotion) + fluctuation))
        inventory_left = max(0, 100 - units_sold + np.random.randint(-10, 10))
#based on the type of product assigning the prices
        if category == "Electronics":
            price = round(np.random.uniform(200, 1000), 2)
        elif category == "Clothing":
            price = round(np.random.uniform(30, 300), 2)
        elif category == "Groceries":
            price = round(np.random.uniform(5, 100), 2)
        else:
            price = round(np.random.uniform(10, 150), 2)

        data.append([
            date.strftime("%Y-%m-%d"), pid, pname, category, warehouse,
            units_sold, inventory_left, price, base_demand,
            is_holiday, weather, promotion
        ])

df = pd.DataFrame(data, columns=[
    "Date", "Product_ID", "Product_Name", "Category", "Warehouse",
    "Units_Sold", "Inventory_Left", "Price", "Demand_Forecast",
    "Holiday", "Weather", "Promotion"
])

df.to_csv("synthetic_inventory_data_extended.csv", index=False)

#training the model and evaluating it

df = pd.read_csv("synthetic_inventory_data_extended.csv")

#encoding all the categorical elements so that they are represented by numbers
le_category = LabelEncoder()
le_warehouse = LabelEncoder()
le_product = LabelEncoder()
le_weather = LabelEncoder()

df["Category"] = le_category.fit_transform(df["Category"])
df["Warehouse"] = le_warehouse.fit_transform(df["Warehouse"])
df["Product_Name"] = le_product.fit_transform(df["Product_Name"])
df["Weather"] = le_weather.fit_transform(df["Weather"])


df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.weekday

#defining the input features as well as the output feature(units_sold)
features = [
    "Product_Name", "Category", "Warehouse", "Price", "Inventory_Left",
    "Demand_Forecast", "Holiday", "Weather", "Promotion",
    "Day", "Month", "Weekday"
]
X = df[features]
y = df["Units_Sold"]

#splitting the data(80:20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

#training the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

#making the prediction 
y_pred = model.predict(X_test)

#evaluating the model by calculating the mse,r2 and the rmse
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")

#plotting the result(actual vs predicted)
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
}).reset_index(drop=True)

results_sorted = results.sort_values(by="Actual").reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(results_sorted["Actual"], label="Actual", linewidth=2)
plt.plot(results_sorted["Predicted"], label="Predicted", linewidth=2)
plt.title("Actual vs Predicted Units Sold")
plt.xlabel("Sample Index")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
