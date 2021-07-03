import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv('iphone_price.csv')
# print(data.head())
# plt.scatter(data['version'],data['price'])
# plt.show()

model = LinearRegression()
# print(type(data['version']))
# print(type(data[['version']]))
model.fit(data[['version']],data[['price']])
print(model.predict([[14],[20],[100]]))