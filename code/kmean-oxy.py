import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans

import os

working_dir = os.getcwd()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#//data base preparation
df = pd.read_csv(f"{working_dir}/data/Oxygen_Plant_24Days.csv")
df.drop(labels='time',axis=1,inplace=True)
#print(df.info())


km_df = df.filter(['AI1 Product gaseous (liquid) oxygen purity','Analys Argon (AI7) Product argon purity (O2 content)','AI4 Oxygen content in waste nitrogen'],axis=1) #,'AI4 Oxygen content in waste nitrogen'
#km_df = km_df.reindex(['AI4 Oxygen content in waste nitrogen','AI1 Product gaseous (liquid) oxygen purity'], axis=1)
print(km_df.head())
km_analysis = KMeans(n_clusters=8,init='k-means++', n_init=10)
clust_lable_dict = {0:'l1',1:'l2',2:'l3',3:'l4',4:'l5',5:'l6',6:'l7',7:'l8'}
km_analysis.fit(km_df)
labels = [clust_lable_dict[i] for i in km_analysis.labels_]
df['kmeans_labels'] = labels
#print(df.head())


#//Air turbine outlet pressure VS products
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Tubine Outlet Pressure VS Air Production', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('bar', fontsize=16,fontweight='bold')
sns.scatterplot(
    data=df,
    #x=df.index,
    x='AI4 Oxygen content in waste nitrogen',
    y='AI1 Product gaseous (liquid) oxygen purity',
    hue='kmeans_labels',
    alpha = 0.9,
)
fig.tight_layout()
plt.savefig(f'{working_dir}/temp/temp.jpg')
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
#plt.savefig(f'{working_dir}/temp/temp.jpg')
#plt.show()
fig.clear()