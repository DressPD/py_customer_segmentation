#!/usr/bin/env python
# coding: utf-8

# # Market Segmentation exercise
# #### Aris Dressino, October 2019, Big Data and Management at Luiss Business School
# 
# - for scooter company
# - identify clusters to perform marketing
# - scooters are taken and given to special stations
# - 30 free minutes for unlimited trips with membership

# ## Cleaning and Normalizing data for sex identification

# In[1]:


# general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


mkt = pd.read_csv('MarketSegmentation.csv')


# In[3]:


# dimensions of dataframe
mkt.shape


# In[4]:


# extracting values for classification problem
x = mkt.iloc[:, :6].values
y = mkt.iloc[:, -1].values


# In[5]:


print(x.shape)    # independent variables
y    # dependent variable = sex of the user


# In[6]:


# train-test split at 20%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# normalization of numerical variables (1b)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)    # standardization mean = 0, sd=1
x_test = sc.fit_transform(x_test)


# In[7]:


print(x.shape)
print(x_test.shape)
# y_test


# ## KNN Classification of sex <- More Precise

# In[8]:


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'euclidean', p = 2)    # or 'minkowski'
classifier.fit(x_train, y_train)

# Predicting the Test set results about sex of customers
y_pred = classifier.predict(x_test)


# In[9]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# accuracy of 65%


# ## Random Forest Classification of sex

# In[10]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm    # accuracy of 58%


# ## Clustering analysis
# 
# - study of clusters group

# In[11]:


mkt = pd.read_csv('MarketSegmentation.csv')

x = mkt.iloc[:].values

# # standardization mean = 0, sd=1 (1b)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)    # features' scaling

x.shape


# In[25]:


scal = dict()
for n in x[:,2]: 
    if (n in scal): 
         scal[n] += 1
    else: 
         scal[n] = 1
scal = dict(sorted(scal.items()))
for key, value in scal.items(): 
    print (" scaled value %.6f  : %.6f"%(key, value))
    
print(x[:,2])

# theoretically, it is not always a good idea to standardize categorical variables, but we can observe that the binary relation
# 0 and 1 of one-hot encoded variables is maintained between the observations although we have different values representing it
# I have chosen to use standardized distributions for categories in order to fit them better with other the other dimensions


# ## Elbow Method on multiple dimensions

# In[13]:


# Using the elbow method to find the optimal number of clusters (2)
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 6 clusters could be the point where the decreasing function is stabilized


# ## K-Means fitting
# 
# - probably overfitting due to outliers and complex dimensionality

# In[14]:


# y_km

# function to count observations in the clusters
def count_clusters(ls):
    km = {} 
    for n in ls: 
        if (n in km): 
             km[n] += 1
        else: 
            km[n] = 1
    km = dict(sorted(km.items()))
    for key, value in km.items(): 
        print (" cluster % d : % d"%(key, value))


# In[15]:


#### Fitting K-Means = 10 to the dataset

km10 = KMeans(n_clusters = 10, init = 'k-means++', random_state = 40)

y_km10 = km10.fit_predict(x)


# In[16]:


y_km10


# In[17]:


count_clusters(y_km10)    # overfitting and groups with too little observations


# In[18]:


# 10 centroids of 7 dimensions
km10.cluster_centers_


# In[19]:


#### Fitting K-Means = 6 to the dataset <- good clustering

km6 = KMeans(n_clusters = 6, init = 'k-means++', random_state = 40)    # or 'random'

y_km6 = km6.fit_predict(x)

count_clusters(y_km6)


# In[20]:


# 6 centroids of 7 dimensions
km6.cluster_centers_


# In[21]:


#### Fitting K-Means = 3 to the dataset <- appropriate clustering with clearly separated groups
# with similar amount of observations

km3 = KMeans(
    n_clusters=3, init='k-means++',    # or 'random'
    n_init=12, max_iter=400, 
    tol=1e-04, random_state=40)

y_km3 = km3.fit_predict(x)

count_clusters(y_km3)


# In[22]:


# 3 centroids of 7 dimensions
km3.cluster_centers_


# In[23]:


#### Fitting K-Means = 2 to the dataset

km2 = KMeans(
    n_clusters=2, init='k-means++',    # or 'random'
    n_init=12, max_iter=400, 
    tol=1e-04, random_state=40)

y_km2 = km2.fit_predict(x)

count_clusters(y_km2)


# In[24]:


# 2 centroids of 7 dimensions
km2.cluster_centers_


# In[ ]:




