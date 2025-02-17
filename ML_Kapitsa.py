
from Dataset_ML_kapitsa import df
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = df.drop(['MathScore'], axis= 1)
Y = df['MathScore']

X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size= 0.15,random_state= 35 )

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
print(f'MSE: {mean_squared_error(Y_test, y_pred)}')
print(f'r2 score: {r2_score(Y_test, y_pred)}')

