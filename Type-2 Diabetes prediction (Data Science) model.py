#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pwd


# In[3]:


variable=pd.read_csv(r"C:\Users\Vidushi\Type2-Diabetes-prediction-master\Dataset\diabetes.csv")
variable


# In[4]:


variable.describe()


# In[5]:


variable.info()


# In[6]:


#Check for all null values
variable.isnull().values.any()


# In[7]:


#histogram
variable.hist(bins=10, figsize=(10,10))
plt.show()


# In[8]:


#Correlation
sns.heatmap(variable.corr())
# we see that skin thickness, age, insulin and pregnancies are fully independent on each other
#age and pregnanacies have negative correlation


# In[9]:


#lets count total outcome in each target 0 1
#0 means no diabeted
#1 means patient with diabtes
sns.countplot(y=variable['OUTCOME'],palette='Set1')


# In[10]:


sns.set(style="ticks")
sns.pairplot(variable, hue="OUTCOME")


# In[11]:


sns.set(style="whitegrid")
variable.boxplot(figsize=(15,6))


# In[12]:


#box plot
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(8,4)})
sns.boxplot(x=variable['INSULIN'])
plt.show()
sns.boxplot(x=variable['BLOOD PRESSURE'])
plt.show()
sns.boxplot(x=variable['DIABETES PEDIGREE FUNCTION'])
plt.show()


# In[13]:


#outlier remove
Q1=variable.quantile(0.25)
Q3=variable.quantile(0.75)
IQR=Q3-Q1
print("---Q1--- \n",Q1)
print("\n---Q3--- \n",Q3)
print("\n---IQR---\n",IQR)

#print((df < (Q1 - 1.5 * IQR))|(df > (Q3 + 1.5 * IQR)))


# In[14]:


#outlier remove
variable_out = variable[~((variable < (Q1 - 1.5 * IQR)) |(variable > (Q3 + 1.5 * IQR))).any(axis=1)]
variable.shape,variable_out.shape
#more than 80 records deleted


# In[15]:


#Scatter matrix after removing outlier
sns.set(style="ticks")
sns.pairplot(variable_out, hue="OUTCOME")
plt.show()


# In[16]:


#lets extract features and targets
X=variable_out.drop(columns=['OUTCOME'])
y=variable_out['OUTCOME']


# In[17]:


#Splitting train test data 80 20 ratio
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)


# In[18]:


train_X.shape,test_X.shape,train_y.shape,test_y.shape


# In[19]:


from sklearn.metrics import roc_curve
def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[20]:


#Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#creating model
reg = LinearRegression()

#fitting training data
reg= reg.fit(X,y)

#Y prediction
Y_pred = reg.predict(X)

#calculating R2 score
r2_score = reg.score(X,y)
print(r2_score)


# In[21]:


#Logisitic Regression
from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(train_X, train_y)
Y_pred_LR = logmodel.predict(test_X)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(test_y, Y_pred_LR)
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

print("Accuracy of LogReg:", accuracy_score(test_y, Y_pred_LR))
print("ROC AUC of LogReg:", roc_auc_score(test_y, Y_pred_LR))
print("Classification report of LogReg:", classification_report(test_y, Y_pred_LR))
print("confusion matrix of LogReg:", cm)
fper, tper, thresholds = roc_curve(test_y, Y_pred_LR) 
plot_roc_curve(fper, tper)


# In[22]:


#KNN 

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(train_X, train_y)
Y_pred_KNN= classifier.predict(test_X)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(test_y, Y_pred_KNN)
from sklearn.metrics import accuracy_score

print("Accuracy of KNN:", accuracy_score(test_y, Y_pred_KNN))
from sklearn.metrics import roc_auc_score
print("ROC AUC of KNN:", roc_auc_score(test_y, Y_pred_KNN))
print("Classification report of KNN:", classification_report(test_y, Y_pred_KNN))
print("confusion matrix of KNN:", cm)

fper, tper, thresholds = roc_curve(test_y, Y_pred_KNN) 
plot_roc_curve(fper, tper)


# In[23]:


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators = 270)
rfc.fit(train_X, train_y)
Y_pred_RFC= rfc.predict(test_X)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(test_y, Y_pred_RFC)
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

print("Accuracy of RFC:", accuracy_score(test_y, Y_pred_RFC))
print("ROC AUC of RFC:", roc_auc_score(test_y, Y_pred_RFC))
print("Classification report of RFC:", classification_report(test_y, Y_pred_RFC))
print("confusion matrix of :", cm)

fper, tper, thresholds = roc_curve(test_y, Y_pred_RFC) 
plot_roc_curve(fper, tper)


# In[24]:


#adaptive boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
model= DecisionTreeClassifier(criterion= 'entropy', max_depth=1)
AdaBoost= AdaBoostClassifier(base_estimator= model, n_estimators=50, learning_rate=1)
boostmodel= AdaBoost.fit(train_X, train_y)
Y_pred_AB= boostmodel.predict(test_X)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(test_y, Y_pred_AB)
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

print("Accuracy of AdaBoost:", accuracy_score(test_y, Y_pred_AB))
print("ROC AUC of AdaBoost :", roc_auc_score(test_y, Y_pred_AB))
print("Classification report of AdaBoost:", classification_report(test_y, Y_pred_AB))
print("confusion matrix of AdaBoost:", cm)

fper, tper, thresholds = roc_curve(test_y, Y_pred_AB) 
plot_roc_curve(fper, tper)


# In[25]:


#gradient boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
modelgb = GradientBoostingClassifier()
modelgb.fit(train_X,train_y)
Y_pred_GB = modelgb.predict(test_X)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(test_y, Y_pred_GB)
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

print("Accuracy of GradBoost:", accuracy_score(test_y, Y_pred_GB))
print("ROC AUC of GradBoost :", roc_auc_score(test_y, Y_pred_GB))
print("Classification report of GradBoost:", classification_report(test_y, Y_pred_GB))
print("confusion matrix of GradBoost:", cm)
fper, tper, thresholds = roc_curve(test_y, Y_pred_GB) 
plot_roc_curve(fper, tper)


# In[ ]:




