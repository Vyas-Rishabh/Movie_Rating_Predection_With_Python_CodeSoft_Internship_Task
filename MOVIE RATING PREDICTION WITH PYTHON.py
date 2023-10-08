#!/usr/bin/env python
# coding: utf-8

# # MOVIE RATING PREDICTION WITH PYTHON

# ### Importing Liabraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')
df.head()


# ### Data Preprocessing

# In[3]:


#Number of Rows
df.shape[0]


# In[4]:


#Number of Columns
df.shape[1]


# In[5]:


print(df.columns.tolist()) #Number of Columns in List


# In[6]:


#Missing values in Columns
df.isnull().sum()


# In[7]:


#Total Number of Missing Values
df.isnull().sum().values.sum()


# In[8]:


#Unique Values
df.nunique()


# In[9]:


df.info()


# In[10]:


#actors value count
df['Actor 1'].value_counts()


# In[11]:


# directors value count
df['Director'].value_counts()


# In[12]:


#genre value count
df['Genre'].value_counts()


# In[13]:


df.head(10)


# In[14]:


# Predict movie ratings based on features, and remove null values from features
df.dropna(subset=['Name', 'Year', 'Duration', 'Rating', 'Votes'], inplace=True)


# In[15]:


df.isna().sum()


# In[16]:


df.head()


# In[17]:


#Dataset Overview after clearning null values


# In[18]:


df.shape[0] #Number of rows


# In[19]:


df.shape[1] #Number of columns


# In[20]:


df.isna().sum().values.sum() #Total number of missing values


# In[21]:


df.nunique() #total number of unique values


# In[22]:


# Remove ("-2019") parentheses from YEAR column and we will convert to INT
df['Year'] = df['Year'].str.strip('()').astype(int)


# In[23]:


# Remove ("1,086") commas from Votes column and we will convert to INT
df['Votes'] = df['Votes'].str.replace(',','').astype(int)


# In[24]:


# Remove (109 min) min from Duration and we will convert to INT
df['Duration'] = df['Duration'].str.replace('min','').astype(int)


# In[25]:


df.info()


# In[26]:


df.describe()


# In[27]:


# Drop the Genre column
df.drop('Genre', axis=1, inplace=True)


# In[34]:


df.head()


# In[35]:


import warnings
warnings.filterwarnings('ignore')


# ### Exploratory Data Analysis (EDA)

# In[43]:


plt.figure(figsize=(14, 7))
plt.subplot(2, 2, 1)
sns.boxplot(x='Votes', data=df)

plt.subplot(2, 2, 2)
sns.distplot(df['Year'], color='g')

plt.subplot(2, 2, 3)
sns.distplot(df['Rating'], color='g')

plt.subplot(2, 2, 4)
sns.scatterplot(x=df['Duration'], y=df['Rating'], data=df)

plt.show()


# In[44]:


#Histogram
df.hist(figsize=(30, 15))


# In[46]:


# Heatmap for Correlation Matrix
corrmat = df.corr()
fig = plt.figure(figsize= (20, 5))

sns.heatmap(corrmat, vmax = .8, square = True, annot = True)
plt.show()


# In[47]:


df.head()


# In[49]:


# Now we will drop another columns
df.drop(['Name', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis = 1, inplace=True)
df.head()


# In[51]:


X = df[['Year','Duration','Votes']]
y = df['Rating']


# In[53]:


X.head()


# In[54]:


y.head()


# In[56]:


# Now we will split data into Training and Testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)


# ### Building a Model

# In[58]:


# Create a pipeline with SGDRegressor and standard scalling
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[59]:


pipeline = Pipeline([('Scaler', StandardScaler()), ('sgd', SGDRegressor(max_iter=1000, random_state=1000))])


# In[60]:


pipeline.fit(X_train, y_train)


# In[61]:


# Now Predict ratings on the test set
y_pred_pipeline = pipeline.predict(X_test)


# In[64]:


y_pred_pipeline


# ### Model Evaluation

# In[66]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluation Metrics for the Pipeline
mae_pipeline = mean_absolute_error(y_test, y_pred_pipeline)
mse_pipeline = mean_squared_error(y_test, y_pred_pipeline)
r2_pipeline = r2_score(y_test, y_pred_pipeline)


# In[67]:


print("Pipeline Mean Absolute Error:", mae_pipeline)
print("Pipeline Mean Squared Error:", mse_pipeline)
print("Pipeline R-square:", r2_pipeline)


# ### Model Deployment

# In[70]:


new_input = pd.DataFrame({'Year':[2022], 'Duration':[135], 'Votes':[10120]})

#Use trained pipeline to make predictons on the new_input
predicted_rating = pipeline.predict(new_input)
print("Predicted Rating:", predicted_rating)


# You can find the Project on <a href="https://github.com/Vyas-Rishabh/Movie_Rating_Predection_With_Python_CodeSoft_Internship_Task"><b>GitHub.</b></a>
