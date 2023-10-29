# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
### step 1:
Use the standard libraries in python.

### step 2:
Set variables for assigning dataset values.

### step 3:
Import LinearRegression from the sklearn.

### step 4:
Assign the points for representing the graph.

### step 5:
Predict the regression for marks by using the representation of graph.

### step 6:
Compare the graphs and hence we obtain the LinearRegression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 212222230081
RegisterNumber: MOHAMED ATHIL B
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)  

```

## Output:

### df.head() :
![266958556-eaca1325-1d7b-490a-8fd7-cc2546601f8a](https://github.com/Bmohamedathil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119560261/8368e116-4be5-4508-90eb-58b3373811bb)

### df.tail() :
![266958691-9fef3a6b-405e-4cc7-baaa-d3dda81bf987](https://github.com/Bmohamedathil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119560261/f2a085d3-bc9b-4de7-a173-0cb521656cd2)

### Array value of X :
![262857976-43a6d7cf-8d85-43de-a85f-93decc96890b](https://github.com/Bmohamedathil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119560261/d5188005-9a25-4651-b752-ce8c5ef35fc0)

### Array value of Y :
![262858010-a5970ea9-eb92-4ab4-be47-531b0eb92022](https://github.com/Bmohamedathil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119560261/e7731c01-1802-474b-b02b-eb01e2a8afee)

### Values of y prediction :
![262858039-963bfe0c-06f4-45d8-860e-be0045fc2d68](https://github.com/Bmohamedathil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119560261/ed59169c-7f2e-4783-9b9d-3cb134e4dc45)

### Array values of Y test :
![262858039-963bfe0c-06f4-45d8-860e-be0045fc2d68](https://github.com/Bmohamedathil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119560261/b271580e-70a4-4f56-8e11-31f108d453b3)

### Training set graph :
![262858079-2765e3ec-2276-4997-bb05-a400fac0bba9](https://github.com/Bmohamedathil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119560261/419a596f-2d1c-4165-8a11-332c316b14e2)

### Test set graph :
![262858111-f6a5c582-0bcf-4f96-83bd-ae870a866e52](https://github.com/Bmohamedathil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119560261/84d024c9-8b9d-48d1-be6f-65b7d7c9212a)

### Values of MSE,MAE and RMSE :
![262858131-e2fecb84-d830-4b0b-bd1a-44e9d3f88bab](https://github.com/Bmohamedathil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119560261/9eb60fa6-a135-4c40-bd41-d30e9fcdcd86)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
