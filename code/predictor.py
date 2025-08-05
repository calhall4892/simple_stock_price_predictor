import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

simple = pd.read_csv(r"C:\Users\Callum\Documents\ML Researcher\Simple stock price\simple_stock_price_predictor\data\Download Data - INDEX_UK_FTSE UK_UKX.csv")
print(simple.head())
print(simple.columns) # to check the names of columns
simple['Date'] = pd.to_datetime(simple['Date']) # change to date
simple = simple.set_index('Date') # set index

print(simple.head) # check correct index

simple['N_D_close'] = simple['Close'].shift(-1) # shifting the dates
simple = simple.dropna() # remove nan

# This is the input data. This is what the model will be trained on
X = simple[['Close']]

# This is the output data. This is what the model is attempting to predict.
y = simple['N_D_close']

"""
To understand the link between the code and the maths, the fundamentals
need to be understood. When assigning X ready for training the data, this
is creating a matrix of the data assigned. The movement from X to y is through
the manipulation of the matrix as seen in ML mathematics books. Therefore next step
is to review matrix manipulations to ensure link between the code and the maths.
"""

split_point = int(len(X) * 0.8) # 80% number to be applied.

# This is the splitting between testing and training data
X_train = X[split_point:]
X_test = X[:split_point]
y_train = y[split_point:]
y_test = y[:split_point]

# This sets up the model. Almost all scikitlearn models use this basic setup.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred) # This is the MSE used for regression.
rmse = mse ** 0.5 # This is the RMSE used for regression

print(f'The MSE for this model is {mse}')
print(f'The RMSE value for this model is {rmse}')