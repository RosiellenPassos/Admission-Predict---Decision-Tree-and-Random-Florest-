#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ### Stetement of the problem : 
# Admission into graduate school is competitive and various institution uses different criteria for their admission. This kernel will help students measure their chances of getting admission into a master program.
# 
# ### Dataset : 
# The dataset used in this kernel can be found here https://www.kaggle.com/mohansacharya/graduate-admissions .
# 
# ### Feature : 
# The dataset consist of the following features
# 
#     GRE Scores (290 to 340)
#     TOEFL Scores (92 to 120)
#     University Rating (1 to 5)
#     Statement of Purpose (1 to 5)
#     Letter of Recommendation Strength (1 to 5)
#     Undergraduate CGPA (6.8 to 9.92)
#     Research Experience (0 or 1)
#     Chance of Admit (0 for <70% and 1 for >=70%)
# 

# # Exploratory Analysis and Preprocessing
# 
# 

# In[1]:


#Importing library for analytics and visualization

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[2]:


#Reading the file 

student_adms = pd.read_csv('/home/thiago/StudentData.csv')
student_adms


# *  __Drop Collums__
# 
# 

# In[3]:


#Droping the first two collums

student_adms.drop("Unnamed: 0", axis=1, inplace=True)
student_adms.drop("Serial No.", axis=1, inplace=True)


# In[4]:


student_adms


# *  __Visualization__
# 
# 

# In[5]:


grafico = px.scatter_matrix(student_adms, dimensions = ['GRE Score','TOEFL Score','University Rating'] , color ='Chance of Admit ')
grafico.show()


# In[6]:


grafico = px.scatter_matrix(student_adms, dimensions = ['SOP', 'LOR ', 'CGPA', 'Research'] , color ='Chance of Admit ')
grafico.show()


# *  __Missing Values__
# 
# 

# In[7]:


student_adms.isnull() #missing values


# In[8]:


student_adms.isnull().sum() #Sum of the missing values


# *  __Statistcs__
# 
# The data has no missing values, now lets check some of the statitics for the features.

# In[9]:


student_adms.describe() #Statitics


# *  __Separating the features__
# 
# For this classification, we'll store the predict features on a variable X and the classifier one on a Y .

# In[10]:


x_student = student_adms.iloc[:,0:7].values #Varible X with predict features
y_student = student_adms.iloc[:,7].values #Variable Y with claffifier features


# *  __Standard Scaler__
# 
# 
# 
# 

# In[11]:


from sklearn.preprocessing import StandardScaler #Importing the lib to Standard scaler the values

scaler_student= StandardScaler() #Creating the variaable to StandardScaler


# In[12]:


x_student = scaler_student.fit_transform(x_student) #Standarding values on the X variable


# *  __Train and Test Split__
# 
# We'll  split the data in train and test values. 

# In[13]:


from sklearn.model_selection import train_test_split #Importing library


# In[14]:


x_student_train, x_student_test, y_student_train, y_student_test = train_test_split(x_student, y_student, test_size = 0.25, random_state = 0)


# In[15]:


import pickle #saving tha trains and tests datas on the drive

with open('student_train_test.pkl', mode = 'wb') as f:
    pickle.dump([x_student_train, x_student_test, y_student_train, y_student_test], f)


# # Desicion Tree Predict
# 
# 

# In[16]:


from sklearn.tree import DecisionTreeClassifier #Importing library


# In[17]:


tree_student = DecisionTreeClassifier (criterion = 'entropy', random_state = 0) #Creating the variaable to DecisionTreeClassifier
tree_student.fit(x_student_train, y_student_train)


# In[18]:


predict = tree_student.predict(x_student_test) #Algorithm prediction
predict


# *  __Acuracy Score__
# 
# After the prediction, we'll se the acuracy of the algorithm.

# In[19]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
accuracy_score(y_student_test, predict) 


# * The algorithm has __77%__ of acuracy.  
# 
# *  In the stataments below we can the for students  classifficated as 0.0 (less than 70% with chance of admit), the algorithm acuracy is 64%. And when it's found a student in this case the algorithm is right in 74% of te times.
# 
# *  For students  classifficated as 1.0(More than 70% with chance of admit), the algorithm acuracy is 85%. And when it's found a student in this case the algorithm is right in 79% of the times.

# In[20]:


print(classification_report(y_student_test, predict))


# For visualyze this acuracy report, we can import the Yellow library and use the Confusion Matrix again

# In[21]:


from yellowbrick.classifier import ConfusionMatrix 


# In[22]:


cm = ConfusionMatrix(tree_student)
cm.fit(x_student_train, y_student_train)
cm.score(x_student_test, y_student_test)


# *  __Features Importances__
# 

# In[23]:


tree_student.feature_importances_


# The feature more important is the GRE Score with 46,6%. This make sense cause the graduate record examination (GRE) is standardized exam used to measure one's aptitude for abstract thinking in the areas of analytical writing, mathematics, and vocabulary. The GRE is commonly used by many graduate schools in the U.S. and Canada to determine an applicant's eligibility for the program. 

# In[24]:


print( 'GRE Scores:', tree_student.feature_importances_[0]*100 ,'%', 
      '\nTOEFL Scores:' ,tree_student.feature_importances_[1]*100 ,'%',  
      '\nUniversity Rating:', tree_student.feature_importances_[2]*100 ,'%', 
      '\nStatement of Purpose:',tree_student.feature_importances_[3]*100 ,'%', 
      '\nLetter of Recommendation Strength:', tree_student.feature_importances_[4]*100 ,'%', 
      '\nUndergraduate CGPA:', tree_student.feature_importances_[5]*100 ,'%', 
      '\nResearch Experience:',tree_student.feature_importances_[6]*100 ,'%')


# # Randon Forest Predict

# In[33]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


with open('student_train_test.pkl', mode = 'rb') as f:
    x_student_train, x_student_test, y_student_train, y_student_test = pickle.load(f)


# In[35]:


random_forest = RandomForestClassifier (n_estimators = 100, criterion= 'entropy', random_state = 0)


# In[36]:


random_forest.fit(x_student_train, y_student_train)


# In[37]:


predict_random = random_forest.predict(x_student_test)
predict_random 


# In[38]:



accuracy_score(y_student_test, predict_random) 


# In[39]:


print(classification_report(y_student_test, predict_random))


# In[40]:


cm = ConfusionMatrix(random_forest)
cm.fit(x_student_train, y_student_train)
cm.score(x_student_test, y_student_test)

