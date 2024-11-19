import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import os

working_dir = os.getcwd()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#//data base preparation
df = pd.read_csv(f"{working_dir}/data/Oxygen_Plant_24Days.csv")
df.drop(labels='time',axis=1,inplace=True)
#print(df.info())

scale = MinMaxScaler()
normal_data = scale.fit_transform(df)
#print(normal_data)

pca = PCA(n_components=2)
pca_output = pca.fit_transform(normal_data)
print(pca.explained_variance_ratio_)

km = KMeans(n_clusters= 4,random_state= 0)
km.fit(pca_output)
clust_lable_dict = {0:'Good',1:'out',2:'Bad',3:'Very Good',4:'l5',5:'l6',6:'l7',7:'l8'}
labels = [clust_lable_dict[i] for i in km.labels_]

#//Air turbine outlet pressure VS products
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Tubine Outlet Pressure VS Air Production', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('bar', fontsize=16,fontweight='bold')
sns.scatterplot(
    #data=df,
    x=pca_output[:,0],
    #y='AI4 Oxygen content in waste nitrogen',
    y=pca_output[:,1],
    hue=labels,
    alpha = 0.9,
)
fig.tight_layout()
#plt.savefig(f'{working_dir}/temp/temp.jpg')
#plt.show()
fig.clear()



df['kmeans_labels'] = labels
print(df.head())

#//Air turbine outlet pressure VS products
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Tubine Outlet Pressure VS Air Production', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('bar', fontsize=16,fontweight='bold')
sns.scatterplot(
    data=df,
    x=df.index,
    #y='AI4 Oxygen content in waste nitrogen',
    y='AI1 Product gaseous (liquid) oxygen purity',
    hue='kmeans_labels',
    alpha = 0.9,
)
fig.tight_layout()
#plt.savefig(f'{working_dir}/temp/temp.jpg')
#plt.show()
fig.clear()

#// turbine Speed VS O2 Purity
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Turbine Speed VS O2 Purity', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('RPM', fontsize=16,fontweight='bold')
sns.kdeplot(
    data=df,
    x='AI1 Product gaseous (liquid) oxygen purity',
    hue='kmeans_labels',
    fill=True,
    alpha = 0.05,
)
fig.tight_layout()
#plt.savefig(f'{working_dir}/temp/temp.jpg')
#plt.show()
fig.clear()


#// turbine Speed VS O2 Purity
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Turbine Speed VS O2 Purity', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('RPM', fontsize=16,fontweight='bold')
sns.kdeplot(
    data=df,
    x='SIC401B - PV Air turbine/booster T/C2000B speed',
    hue='kmeans_labels',
    fill=True,
    alpha = 0.05,
)
plt.axvline(27793, c='green')
plt.annotate('27793 RPM', xy =(27787, 0.0023),rotation = 90,ha='center', fontsize=18,alpha = 0.8) 
#plt.axvline(210, c='red')
#plt.annotate('224', xy =(224, 0.009),rotation = 90,ha='center', fontsize=16) 
fig.tight_layout()
plt.savefig(f'{working_dir}/temp/temp.jpg')
#plt.show()
fig.clear()
