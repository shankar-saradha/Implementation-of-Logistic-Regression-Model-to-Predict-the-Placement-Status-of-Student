# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Shankar 
RegisterNumber:  212221240052
*/

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1["gender"])
data1['ssc_b']=le.fit_transform(data1["ssc_b"])
data1['hsc_b']=le.fit_transform(data1["hsc_b"])
data1['hsc_s']=le.fit_transform(data1["hsc_s"])
data1['degree_t']=le.fit_transform(data1["degree_t"])
data1['workex']=le.fit_transform(data1["workex"])
data1['specialisation']=le.fit_transform(data1["specialisation"])
data1['status']=le.fit_transform(data1["status"])
print(data1)

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
Original data(first five columns):

![output1](https://user-images.githubusercontent.com/93978702/173190082-98a2de2c-8ca7-49f6-ad8b-36431e686900.png)

Data after dropping unwanted columns(first five):

![output2](https://user-images.githubusercontent.com/93978702/173190087-e61cf43d-f0c9-4335-b102-30e504a5605b.png)

Checking the presence of null values:

![output3](https://user-images.githubusercontent.com/93978702/173190091-f8b0bddc-1ebe-4c02-b692-2258445c22f6.png)

Checking the presence of duplicated values:

![output4](https://user-images.githubusercontent.com/93978702/173190101-68ded8b7-9a66-4ecb-990b-7844882d92cc.jpg)

Data after Encoding:

![output5](https://user-images.githubusercontent.com/93978702/173190124-0f4c7a2b-51ed-428e-8ed7-6593508454a8.jpg)

X Data:

![output6](https://user-images.githubusercontent.com/93978702/173190139-7444e46b-33b4-4303-a4b7-e2b57c31e8c2.jpg)

Y Data: 

![output7](https://user-images.githubusercontent.com/93978702/173190168-b5a6ac5f-d68b-4925-b659-a1906898dfd8.jpg)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
