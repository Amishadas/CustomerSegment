# -*- coding: utf-8 -*-
"""

@author: AmishaDas


"""


#importing all the necessary libraries
import numpy as np         # np for mathematical functions 
import pandas as pd        # pd for dataprocessing, dataframes and opening CSV files
import matplotlib          # extension of numpy
import matplotlib.pyplot as plt     #Data visualization
import seaborn as sns      # Data visualization

#version of the libraries
print(pd.__version__)     #version of pandas used is 0.25.1 
print(np.__version__)     #version of numpy used is 1.15.4
print(matplotlib.__version__)   #version of matplotlib used is 3.0.2
print(sns.__version__)          #version of seaborn used is 0.9.0

#importing the absolute path on my system
import os
os.chdir('D:/Datascience')

#importing dataset 'Mall_Customers.csv'
data=pd.read_csv("Mall_Customers.csv")

#Creating a copy of the original dataset
data1=data.copy()
data1.head(10)     #Printing the first 10 rows of the dataset

data1.describe() #shows some statistical data
#To get the total rows and columns  in the dataset
data1.shape      #there are 200 rows and 5 columns

#To get the missing values if exist in the dataset
data1.info() #there are no missing values 

#to get the missing values computation
data1.isnull().sum()    # it is 0

#########################################################################################
########################################################################################
#Data Visualization of Spending Score and Annual Income(Box Plot)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(y=data1["Spending Score (1-100)"], color="blue")
plt.subplot(1,2,2)
sns.boxplot(y=data1["Annual Income (k$)"])
plt.show()
#From the figure it is clear that the range of spending score is more than
#annual income


#bar plot for distribution of male and female population
plt.figure(figsize=(10,4))
genders = data1.Gender.value_counts()
sns.set_style("darkgrid")
sns.barplot(x=genders.index, y=genders.values)
plt.show()
#From the graph it is clear that female has more on part.


#the Below code is for distribution of ages and to know which age group 
#has more contribution
#We have divided the age group 
age18_25 = data1.Age[(data1.Age <= 25) & (data1.Age >= 18)]
age26_35 = data1.Age[(data1.Age <= 35) & (data1.Age >= 26)]
age36_45 = data1.Age[(data1.Age <= 45) & (data1.Age >= 36)]
age46_55 = data1.Age[(data1.Age <= 55) & (data1 .Age >= 46)]
age55above = data1.Age[data1.Age >= 56]

#here on x-axis or xlabel we are showing the age groups
#and on y-axis the number of customers
x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y, palette="deep")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()
#From the graph we can see that age group 26-35 has more contribution


#The max score in spending score is 99 and we distribute it like wise
data1.describe()

spen1_20 = data1["Spending Score (1-100)"][(data1["Spending Score (1-100)"] >= 1) & (data1["Spending Score (1-100)"] <= 20)]
spen21_40 = data1["Spending Score (1-100)"][(data1["Spending Score (1-100)"] >= 21) & (data1["Spending Score (1-100)"] <= 40)]
spen41_60 = data1["Spending Score (1-100)"][(data1["Spending Score (1-100)"] >= 41) & (data1["Spending Score (1-100)"] <= 60)]
spen61_80 = data1["Spending Score (1-100)"][(data1["Spending Score (1-100)"] >= 61) & (data1["Spending Score (1-100)"] <= 80)]
spen81_100 = data1["Spending Score (1-100)"][(data1["Spending Score (1-100)"] >= 81) & (data1["Spending Score (1-100)"] <= 100)]
# xlabel are showing the spending score
# ylabel are showing the no. of customers having the scores
spenx = ["1-20", "21-40", "41-60", "61-80", "81-100"]
speny = [len(spen1_20.values), len(spen21_40.values), len(spen41_60.values), len(spen61_80.values), len(spen81_100.values)]

plt.figure(figsize=(14,5))
sns.barplot(x=spenx, y=speny, palette="deep")
plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")
plt.show()
#from the graph, the maximum customers having the spending range between 41-60



#Data visualization of number of customers with their annual income
ann0_30 = data1["Annual Income (k$)"][(data1["Annual Income (k$)"] >= 0) & (data1["Annual Income (k$)"] <= 30)]
ann31_60 = data1["Annual Income (k$)"][(data1["Annual Income (k$)"] >= 31) & (data1["Annual Income (k$)"] <= 60)]
ann61_90 = data1["Annual Income (k$)"][(data1["Annual Income (k$)"] >= 61) & (data1["Annual Income (k$)"] <= 90)]
ann91_120 = data1["Annual Income (k$)"][(data1["Annual Income (k$)"] >= 91) & (data1["Annual Income (k$)"] <= 120)]
ann121_150 = data1["Annual Income (k$)"][(data1["Annual Income (k$)"] >= 121) & (data1["Annual Income (k$)"] <= 150)]

annx = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
anny = [len(ann0_30.values), len(ann31_60.values), len(ann61_90.values), len(ann91_120.values), len(ann121_150.values)]

plt.figure(figsize=(14,5))
sns.barplot(x=annx, y=anny, palette="deep")
plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of Customer")
plt.show()
#From the bar plot we can see that majority of the customers have annual income 60,001-90,000


#########################################################################################
#########################################################################################
X= data1.iloc[:, [3,4]].values
#Kmeans algorithm to decide the optimum cluster number
from sklearn.cluster import KMeans
wcss = []
#let us assume the max number of cluster would be 10

###Selecting the features for model because it annual income and spending income 
#Considering only 2 features (Annual income and Spending Score) and no Label available
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)       #inertia_ is the formula used to segregate the
#Visualizing the elbow method to get optimal value of k
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11), wcss,linewidth=2, color="red", marker ="8")

plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()
#We can see from the graph that the last elbow comes at k=5

##################################################################################
##################################################################################
#Model Building
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X) 
#y_kmeans is the final model

#Visualizing he cluster 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'black', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#We have have 5 clusters like
#cluster 1 shows annual income high and spending less
#cluster 2 shows it's is an equalizing factor of annual income and spending score
#cluster 3 shows annual income is high and spending score is also high
#cluster 4 shows annual income is less and spending score is high
#cluster 5 shows annual income is less and spending score is also low
#And there are some centroids in betwwn the cluster
