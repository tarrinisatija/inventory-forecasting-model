import pandas as pd
import numpy as np
import random
from prophet import Prophet
import matplotlib.pyplot as plt

num_days = 365
num_products = 50
start_date = pd.to_datetime("2024-01-01")
product_ids = [f"P{str(i).zfill(3)}" for i in range(1, num_products + 1)]
product_names = [f"Product_{i}" for i in range(1, num_products + 1)]
categories = ["Electronics", "Clothing", "Groceries", "Stationery"]
warehouses = ["W1", "W2", "W3"]
data = []
for day in range(num_days):
    date = start_date + pd.Timedelta(days=day)
    weekday = date.weekday()
    for pid, pname in zip(product_ids, product_names):
        category = random.choice(categories)
        warehouse = random.choice(warehouses)
        base_demand = np.random.randint(20, 60) if weekday in [5, 6] else np.random.randint(10, 40)
        fluctuation = np.random.normal(0, 7)
        units_sold = max(0, int(base_demand + fluctuation))
        inventory_left = max(0, 100 - units_sold + np.random.randint(-10, 10))
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
            units_sold, inventory_left, price, base_demand
        ])
df = pd.DataFrame(data, columns=[
    "Date", "Product_ID", "Product_Name", "Category", "Warehouse",
    "Units_Sold", "Inventory_Left", "Price", "Demand_Forecast"
])
df.to_csv("synthetic_inventory_data.csv", index=False)
df = pd.read_csv("synthetic_inventory_data.csv")
df_product = df[df["Product_Name"] == "Product_20"]
df_prophet = df_product[["Date", "Units_Sold"]].rename(columns={"Date": "ds", "Units_Sold": "y"})
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
figure=model.plot(forecast)
plt.figure(figsize=(8, 4))
plt.show()
print(forecast[["ds", "yhat"]].tail(5))