
from Dataset import df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib



X = df.drop('Price', axis= 1)
Y = df['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.1, random_state= 42)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
print("Mean squared error:", mean_squared_error(Y_test, y_pred))
print(f"R^2 Score: {r2_score(Y_test, y_pred)}")

joblib.dump(model, "First_ML.pkl")