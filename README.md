# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the spam dataset, detect encoding, and handle missing values.
2. Split the dataset into input texts (x) and labels (y), then into training and testing sets.
3. Transform the text data into numerical vectors using CountVectorizer.
4. Train an SVC model, predict labels for test data, and calculate the accuracy score.

## Program:

Program to implement the SVM For Spam Mail Detection..
Developed by: Venkatachalam S
RegisterNumber:  212224220121
```
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

file = '/content/spam.csv'
with open(file, 'rb') as f:
    result = chardet.detect(f.read(100000))
result

df = pd.read_csv('/content/spam.csv',encoding='Windows-1252')

df.head()

df.info()

df.isnull().sum()

x = df['v1'].values
y = df['v2'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

accuracy = accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/5742fbda-ee0e-410e-9d9f-00fa1d231d20)
![image](https://github.com/user-attachments/assets/47db3c94-cf8d-4403-a49e-c413addfe82f)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
