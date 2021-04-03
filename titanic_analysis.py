#!/usr/bin/env python
# coding: utf-8

# # Problem Statement : Find survival rate on Titanic

# In[1]:


#import necessary libraries for doing data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Acquisition

# In[2]:


# Load the data from the csv
# Dataset taken from https://www.kaggle.com/hesh97/titanicdataset-traincsv
titanic_df = pd.read_csv(r"C:\Users\ankushsharma\Desktop\archive\train.csv")
titanic_df.head()


# # Data Processing

# In[3]:


titanic_df.info()


# In[4]:


#Drop the cabin column as there are many null values and it does not help in making predictions
titanic_df.drop('Cabin',axis=1,inplace=True)


# In[5]:


#Age also contains lot of empty/null/n.a values and so we can replace them with mean of age for better analysis
#Fill empty or NA values of Age with mean value
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())


# In[6]:


#Filling the null values in the Embarked column with 'S' as those are more frequent
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')


# In[7]:


#Checking if there are any null values in dataframe
titanic_df.isnull().values.any()


# # Data Visualization & Data Exploration

# In[8]:


# using seaborn lib methods to show plots betwwen Age and Survival
sns.lmplot('Age','Survived',data=titanic_df)
#Looks like with increasing Age, chances of survival tends to be less comparatively


# In[9]:


sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass')


# In[11]:


titanic_df['Alone']=titanic_df.Parch + titanic_df.SibSp


# In[12]:


#If Alone is > 0 then with family oe else without family
titanic_df['Alone'].loc[titanic_df['Alone'] > 0]='With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0]='Without Family'


# In[13]:


#Who was alone or with family
sns.catplot('Alone',kind='count',data=titanic_df)


# In[14]:


#Who was alone or with family according to class
sns.catplot('Alone',kind='count',data=titanic_df,hue='Pclass')


# In[15]:


#Who survived and who didn't
sns.catplot('Survived',kind='count',data=titanic_df)


# In[16]:


sns.factorplot('Survived',kind='count',data=titanic_df,hue='Pclass')


# In[17]:


#Check how many children were there
def titanic_children(passenger):
    age,sex=passenger
    if age < 18:
        return 'child'
    else:
        return sex    


# In[21]:


titanic_df['person']=titanic_df[['Age','Sex']].apply(titanic_children,axis=1)


# In[22]:


titanic_df.info()


# # Correlation between different variables

# In[19]:


corr=titanic_df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,vmax=.8,linewidths=0.01,square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between variables');


# # Data Modelling

# In[30]:


person_dummies = pd.get_dummies(titanic_df['person'])
alone_dummies = pd.get_dummies(titanic_df['Alone'])
embarked_dummies = pd.get_dummies(titanic_df['Embarked'])


# In[31]:


embarked_dummies.drop('Q',axis=1,inplace=True)


# In[32]:


pclass_dummies = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies.columns=['class 1','class 2','class 3']


# In[33]:


import math
titanic_df['Age']=titanic_df['Age'].apply(math.ceil)
titanic_df['Fare']=titanic_df['Fare'].apply(math.ceil)


# In[34]:


titanic_df = pd.concat([titanic_df,pclass_dummies,person_dummies,alone_dummies,embarked_dummies],axis=1)


# In[35]:


titanic_df.drop(['PassengerId','Name','Sex','SibSp','Parch','Ticket','Embarked'],axis=1,inplace=True)


# In[36]:


titanic_df.drop(['Alone','person','Pclass','Without Family'],axis=1,inplace=True)


# In[37]:


titanic_df.head()


# In[38]:


#needed to convert all values into numeric as  modelling identifies only numeric data


# In[39]:


titanic_train=titanic_df.drop('Survived',axis=1)


# In[40]:


#modelling algos need only numerical data 
titanic_survived = titanic_df.Survived


# In[44]:


x_train,x_test,y_train,y_test = train_test_split(titanic_train,titanic_survived,test_size=0.2)


# In[45]:


log_model = LogisticRegression()
log_model.fit(x_train,y_train)
train_survival=log_model.predict(x_test)


# In[43]:


print("Accuracy score of logistical model is ",metrics.accuracy_score(y_true=y_test,y_pred=train_survival))


# In[ ]:


#our analysis predicted nearly 80% correct results

