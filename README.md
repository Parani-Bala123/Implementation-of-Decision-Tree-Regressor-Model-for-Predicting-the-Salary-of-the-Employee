# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Parani Bala M
RegisterNumber: 212224230192
*/


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

data = pd.read_csv("Salary.csv")
print(data.head(), "\n")
print(data.info(), "\n")
print(data.isnull().sum(), "\n")

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head(), "\n")

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor(random_state=2)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("Mean square erroe (MSE):",mse)
print("R-squared score (R2):",r2)

print("Predicted salary for input[5,6]:",dt.predict([[5, 6]])[0])

```

## Output:

<img width="463" height="147" alt="image" src="https://github.com/user-attachments/assets/b64aa843-7438-471f-a79d-22c925e5d39c" />

<img width="493" height="243" alt="image" src="https://github.com/user-attachments/assets/031e2962-f7cf-4714-b6f1-88c15322dbb5" />

<img width="414" height="140" alt="image" src="https://github.com/user-attachments/assets/3dca7e65-0b94-4f3b-a5c2-03e6088a71ff" />

<img width="482" height="234" alt="image" src="https://github.com/user-attachments/assets/44a7a572-8eb0-461e-b34d-cb02ebaec16e" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
