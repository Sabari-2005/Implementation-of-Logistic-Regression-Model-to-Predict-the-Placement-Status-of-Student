# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SABARINATH R 
RegisterNumber:  212223100048
*/
import pandas as pd 
data=pd.read_csv("Placement_Data.csv") 
data.head() 
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()      
data1.duplicated().sum() 
from sklearn .preprocessing import LabelEncoder
le=LabelEncoder() 
data1["gender"]=le.fit_transform(data1["gender"]) 
data1["ssc_b"]=le.fit_transform(data1["ssc_b"]) 
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"]) 
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"]) 
data1["specialisation"]=le.fit_transform(data1["specialisation"]) 
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1] 
x 
y=data1["status"] 
y 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)    #Predicts placement (0 or 1) for the test dataset (x_test).
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print(classification_report1) 
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## Data :
![image](https://github.com/user-attachments/assets/a1fe20ce-d517-453a-b163-e807f9302667)


## Transformed Data:
![image](https://github.com/user-attachments/assets/2f12a6e5-f70d-47b0-b15e-2e0b65c085f0)


## X and Y values:
![image](https://github.com/user-attachments/assets/6ab25c8f-b6a7-4167-8ccb-a8485f1904aa)

![image](https://github.com/user-attachments/assets/1784d3b2-ad99-44f7-b41b-1e719df311d7)


## Model:
![image](https://github.com/user-attachments/assets/347acfc9-727d-484b-92bf-db927c228c00)

## Accuracy :
![image](https://github.com/user-attachments/assets/0597ad9f-1c51-4030-bbd7-91b22a8f43c9)


## Classifiaction:
![image](https://github.com/user-attachments/assets/3bbfabcc-ff40-465d-803d-665ee1fe0d1e)


## Prediction:
![image](https://github.com/user-attachments/assets/5c9202b2-1e81-4a40-8030-e286a4765044)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
