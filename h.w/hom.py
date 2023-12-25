import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("cars_pridections.csv")
print(data.isnull().sum())

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

model=LogisticRegression()
model.fit(x_train, y_train) 
y_pred=model.predict(x_test)

print("train_Acc is ", model.score(x_train,y_train))
print("test_Acc  is ", model.score(x_test,y_test))
print(accuracy_score(y_test,y_pred))


# رسم البيانات بشكل دائري

plt.pie([np.sum(y_test == 0), np.sum(y_test == 1)], labels=['Class 0', 'Class 1'], colors=['blue', 'red'], autopct='%1.1f%%')
plt.title('Distribution of Classes in Actual Data')

# إظهار الشكل
plt.show()


# حساب confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# رسم confusion matrix باستخدام seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()










