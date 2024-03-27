import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, SGDRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# data
real_estate_data = pd.read_csv("realtor-data.csv")
real_estate_data.head()
# dropping the unnecessary features
new_data = real_estate_data.drop(['status', 'full_address', 'street', 'sold_date'], axis=1)
#new_data.head(10)

# Removing null values from the dataset
new_data.isnull().sum()
new_data.dropna(how='any', inplace=True)
new_data.info()



# Question 1) how many houses are available to purchase in each state ?
sb.set_style('darkgrid')
ch=new_data['state'].value_counts()[:10]
print(" see ")
print(new_data['state'].value_counts()[:10])
sb.barplot(x=ch.index,y=ch,palette='viridis')
plt.xlabel('state')
plt.ylabel('Available Houses')
plt.title("Houses availability of Top 10 states")
plt.show()


############################################################################################################
print(" question 2: \n")
# Question 2) what states has the most expensive houses?
count = new_data.groupby(by=['state'])['price'].median().sort_values(ascending=False)
sb.set_style('darkgrid')
#state=new_data['state'].value_counts()[:10]

sb.barplot(x=count.index,y=count,palette='viridis')
plt.xlabel('states')
plt.ylabel('price')
plt.title("Most Expensive Houses")
plt.show()

############################################################################################################

# Question 3) what feutures affect the price of a house?
print(" question 3: \n")
print(new_data.corr())

# plotting correlation heatmap
dataplot = sb.heatmap(new_data.corr(), cmap="YlGnBu", annot=True)

# displaying heatmap
plt.show()


########################################    Model          ######


# dividing data into training and testing
X = new_data[['house_size']]
y = new_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(y_train.describe())
##########

# creating our model to predict the price of houses

# linear regression

model = LinearRegression()
model.fit(X_train, y_train)

price_prediction = model.predict(X_test)

print(price_prediction)

# evaluating our model
r2 = r2_score(y_test, price_prediction)
rmse = mean_squared_error(y_test, price_prediction, squared=False)

print(" Linear Regression: \n")
print('The r2 is: ', r2)
print('The rmse is: ', rmse)

# random forest regression

# create model
random_forest_model = RandomForestRegressor()


# fit the model
random_forest_model.fit(X_train, y_train)

price_prediction_2 = model.predict(X_test)

print(price_prediction_2)

# evaluating our model
r2_2 = r2_score(y_test, price_prediction)
rmse_2 = mean_squared_error(y_test, price_prediction, squared=False)

print("Random Forest: \n")
print('The r2 is: ', r2_2)
print('The rmse is: ', rmse_2)
