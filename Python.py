#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Analysis is carried out on a marketing campaign dataset based on a case of a
#retailer company in computer accessories. The dataset consists of 19 variables and 1500 cases.
#The data will first be prepared by identifying missing values and errors in data entry. The
#variables which are not related to the analysis will be eliminated and some variable will be
#transformed according to the needs of the analysis. After preparing the data, Python
#programmes will be developed to analyse the summary statistics, correlation and Euclidean
#distance. Finally, a logistic regression model will be built using Python and the model will be
#checked for adequacy.

#Loading the library
import pandas as pd
import numpy as np

#Loading the data
data= pd.read_csv('Marketing Campaign data.csv')


# In[2]:


# obtain a summary statistics, from which the min value can be 
#observed to check for missing values for certain variables.
data.describe() 


# In[3]:


# checking for empty entry in the dataset
data.isnull().sum() 


# In[4]:


#It was observed that the variable ‘OCCUPATION' has missing values which were recorded as ‘?'.
#This cannot be used in the analysis and hence these cases have to be deleted. 
#The variable ‘COMMENTS’ have missing values, these cases will not be removed as it was decided
#that variable would be eliminated.
# Deleting '?' in variable OCCUPATION and NAN values
data=data.replace({'?':np.nan}).dropna()
# altenative method to droping NAN values
#data1 = data[~pd.isnull(data)] 


# In[5]:


# Dropping variables
data=data.drop(['PRINTER_SUPPLIES','COMMENTS','OS_DOC_SET_KANJI'],axis=1)
data
# alternative method
#list_drop = ['PRINTER_SUPPLIES','COMMENTS','OS_DOC_SET_KANJI']
#data.drop(list_drop, axis=1, inplace=True)


# In[6]:


## Variable transformation
# changing customer gender into 1 and 0 for M and F respectively
gender = {'M': 1,'F': 0}
data.CUST_GENDER = [gender[item] for item in data.CUST_GENDER]
################################################################################
# Checking frequency of each country in the data
pd.value_counts(data['COUNTRY_NAME'])
# Country name into ordinal numbers
country_code = { 
                'United States of America':1,
                'Argentina':2,
                'Italy':3,
                'Brazil':4, 
                'Germany':5,
                'Poland':6, 
                'Canada':7,
                'United Kingdom':8,
                'Saudi Arabia':9,
                'Denmark':10,
                'China':11, 
                'Singapore':11,
                'New Zealand':11,
                'Japan':11,
                'Spain':11,
                'Turkey':11,
                'Australia':11,
                'France':11,
                'South Africa':1}
data.COUNTRY_NAME = [country_code[item] for item in data.COUNTRY_NAME]
################################################################################
# Checking number of categories that is already classified
pd.value_counts(data['CUST_INCOME_LEVEL'])
# Customer income level into ordinal
income_level= {
'J: 190,000 - 249,999':4,
'L: 300,000 and above':5,
'I: 170,000 - 189,999':4,
'K: 250,000 - 299,999':4,
'F: 110,000 - 129,999':3,
'G: 130,000 - 149,999':3,
'E: 90,000 - 109,999':2,
'H: 150,000 - 169,999':4,
'B: 30,000 - 49,999':1,
'C: 50,000 - 69,999':2,
'D: 70,000 - 89,999':2,
'A: Below 30,000':1}
data.CUST_INCOME_LEVEL= [income_level[item] for item in data.CUST_INCOME_LEVEL]
################################################################################
# Checking current classification of education
pd.value_counts(data['EDUCATION'])
# Education into ordinal level
education= {
'HS-grad':4, 
'< Bach.':4,
'Bach.':5,
'Masters':5,
'Assoc-V':4,
'Assoc-A':4,
'10th':2,
'11th':3,
'Profsc':5,
'7th-8th':2,
'9th':2,
'PhD':5,
'12th':3,
'5th-6th':1,
'Presch.':1,
'1st-4th':1,}
data.EDUCATION = [education[item] for item in data.EDUCATION]
################################################################################
# Identify how the data entry error was identified by python
pd.value_counts(data['HOUSEHOLD_SIZE'])
# household into ordinal level
household= {'1':1,'2':2,'3':3,'4-5':4, '6-8':5, '9+':6}
data.HOUSEHOLD_SIZE = [household[item] for item in data.HOUSEHOLD_SIZE]
################################################################################
pd.value_counts(data['OCCUPATION'])
occupation= {
'Exec.':1,
'Crafts':2,
'Sales':3,
'Cleric.':4,
'Prof.':5, 
'Other':6,
'Machine':7,
'Transp.':8,
'Handler':9,
'TechSup':10,
'Farming':11,
'Protec.':12,
'House-s':13,    
'Armed-F':14,}
data.OCCUPATION=[occupation[item] for item in data.OCCUPATION]
#################################################################################
pd.value_counts(data['CUST_MARITAL_STATUS'])
marital={
'Married':1,
'NeverM':2,
'Divorc.':3,
'Separ.':4,
'Widowed':5,
'Mabsent':6,
'Mar-AF':7,}
data.CUST_MARITAL_STATUS=[marital[item] for item in data.CUST_MARITAL_STATUS]


# In[7]:


#Python code designed to calculate the summary statistics of any variables

#Summary statistics
# create a dictonary for all variables
dic={ 1:'CUST_GENDER',
    2:'AGE',
    3:'CUST_MARITAL_STATUS',
    4:'COUNTRY_NAME',
    5:'CUST_INCOME_LEVEL',
    6:'EDUCATION',
    7:'OCCUPATION',
    8:'HOUSEHOLD_SIZE',
    9:'YRS_RESIDENCE',
    10:'AFFINITY_CARD',
    11:'BULK_PACK_DISKETTES',
    12:'FLAT_PANEL_MONITOR',
    13:'HOME_THEATER_PACKAGE',
    14:'BOOKKEEPING_APPLICATION',
    15:'Y_BOX_GAMES',}
# provide a list of variable number to choose from
print('Choose the variables number from list shown',
    'customer gender          - 1',
    'Age                      - 2',
    'Marital status           - 3',
    'Country name             - 4',
    'Income level             - 5',
    'Education                - 6',
    'Occupation               - 7',
    'Household size           - 8',
    'Yrs residence            - 9',
    'Affinity card            - 10',
    'Bulk Pack Diskettes      - 11',
    'Flat panel monitor       - 12',
    'Home theater package     - 13',
    'Bookkeeping application  - 14',
    'Y box games              - 15',
    sep="\n")  
# store the user choice  
x=int(input(
'Enter the respective number of variable to obtain summary statistics:'))
# Calculation 
SUM=data[dic[x]].sum() 
MEAN=data[dic[x]].mean()
Standard_deviation=data[dic[x]].std()
Skewness=data[dic[x]].skew()
Kurtosis=data[dic[x]].kurt()
print('The sum is %d.' % SUM, 
       'The Mean is %f' %MEAN,
       'The Standard Deviation is %f' %Standard_deviation,
       'The Skewness is %f' % Skewness,
       'The Kurtosis is %f' %Kurtosis,
        sep="\n")


# In[8]:


# Correlation of target variable with other variables
Correlation= data.corr()['AFFINITY_CARD']
print(Correlation.sort_values(ascending=False))
################################################################################
# Euclidean Distance
from scipy.spatial import distance
# prompt to enter Customer ID
Customer_ID_1= int(input(
'Enter ID number of customer:'))
Customer_ID_2= int(input(
'Ebter ID number of next customer:'))
# Euclidean distance calculation
euc_dst= distance.euclidean(data.loc[Customer_ID_1],data.loc[Customer_ID_2])
print('The Euclidean distance is %f.' % euc_dst)
###################################################


# In[9]:


# Histogram for chosen variable
import matplotlib.pyplot as plt

# create a dictonary for all variables
dic={
1:'CUST_GENDER',
2:'AGE',
3:'CUST_MARITAL_STATUS',
4:'COUNTRY_NAME',
5:'CUST_INCOME_LEVEL',
6:'EDUCATION',
7:'OCCUPATION',
8:'HOUSEHOLD_SIZE',
9:'YRS_RESIDENCE',
10:'AFFINITY_CARD',
11:'BULK_PACK_DISKETTES',
12:'FLAT_PANEL_MONITOR',
13:'HOME_THEATER_PACKAGE',
14:'BOOKKEEPING_APPLICATION',
15:'Y_BOX_GAMES',}
# provide a list of variable number to choose from

print('Choose the variables number from list shown',
    'customer gender          - 1',
    'Age                      - 2',
    'Marital status           - 3',
    'Country name             - 4',
    'Income level             - 5',
    'Education                - 6',
    'Occupation               - 7',
    'Household size           - 8',
    'Yrs residence            - 9',
    'Affinity card            - 10',
    'Bulk Pack Diskettes      - 11',
    'Flat panel monitor       - 12',
    'Home theater package     - 13',
    'Bookkeeping application  - 14',
    'Y box games              - 15',
    sep="\n")
    
# store the user choice  
num=int(input('Enter the respective number of the variable:'))

#plot the histogram
data[dic[num]].plot(kind='hist',stacked=True,bins=10)
# label x axis according the user choice
plt.xlabel(dic[num])


# In[25]:


# scatter plot of any 2 chosen variables
# create a dictonary for all variables
dic={
1:'CUST_GENDER',
2:'AGE',
3:'CUST_MARITAL_STATUS',
4:'COUNTRY_NAME',
5:'CUST_INCOME_LEVEL',
6:'EDUCATION',
7:'OCCUPATION',
8:'HOUSEHOLD_SIZE',
9:'YRS_RESIDENCE',
10:'AFFINITY_CARD',
11:'BULK_PACK_DISKETTES',
12:'FLAT_PANEL_MONITOR',
13:'HOME_THEATER_PACKAGE',
14:'BOOKKEEPING_APPLICATION',
15:'Y_BOX_GAMES',}
# provide a list of variable number to choose from

print('Choose the variables number from list shown',
    'customer gender          - 1',
    'Age                      - 2',
    'Marital status           - 3',
    'Country name             - 4',
    'Income level             - 5',
    'Education                - 6',
    'Occupation               - 7',
    'Household size           - 8',
    'Yrs residence            - 9',
    'Affinity card            - 10',
    'Bulk Pack Diskettes      - 11',
    'Flat panel monitor       - 12',
    'Home theater package     - 13',
    'Bookkeeping application  - 14',
    'Y box games              - 15',
    sep="\n")
    
# store the user choice  
x=int(input('Enter the respective number of the variable for x axis:'))
y=int(input('Enter the respective number of the variable for y axis:'))

#plot the scatter plot
plt.scatter(data[dic[x]],data[dic[y]])
# label x and y axis
plt.xlabel(dic[x])
plt.ylabel(dic[y])


# In[18]:


# Data modelling LOGISTIC REGRESSION 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# classifying explanatory variables into x and target variable into y
y = data.iloc[:,[10]]
X = data.iloc[:,[1,2,3,4,5,6,7,8,9,11,12,13,14,15]]
# spliting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#checking training and test data partition
X_train.shape
X_test.shape
# implementing the model
import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
#checking summary statistics to check for significance of variable
result=logit_model.fit()
print(result.summary())
# Fit logistic regression to the training set
logic = LogisticRegression(random_state=0,max_iter=1000)
logic.fit(X_train, np.ravel(y_train,order='C'))


# In[19]:


# Predict test set results and confusion matrix
y_pred = logic.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
# Accuracy of model
print
('Accuracy on test data:{:.2f}'.format(logic.score(X_test, y_test)))

# Creating ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logic.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logic.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# plot base line
plt.plot([0, 1], [0, 1],'r--')
# Set axis limit for x-axis and y-axis
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# label axis
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
# Dispay AUC at lower right
plt.legend(loc="lower right")


# In[ ]:


# Detailed Explanation can be found in Python.pdf

