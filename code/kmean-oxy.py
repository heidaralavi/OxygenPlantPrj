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
#print(df.shape)

scale = MinMaxScaler()
normal_df = pd.DataFrame(data = scale.fit_transform(df) , columns= df.columns)
#print(normal_df.shape)

corr_df = normal_df.corr().abs()
corr_df = corr_df.where(np.triu(np.ones(corr_df.shape),k= 1).astype(bool))
high_corr_columns_name = pd.DataFrame(corr_df.ge(.95).stack().loc[lambda corr_df: corr_df].index.to_list())
#print(high_corr_columns_name)
#high_corr_columns_name.to_excel(f"{working_dir}/temp/high_corr_features.xlsx",index=False)

high_corr_to_drop = pd.read_csv(f"{working_dir}/data/high_corr_features.csv")
df.drop(labels=high_corr_to_drop['high_corr_features'] ,axis=1,inplace=True)
print(df.shape)

scale = MinMaxScaler()
normal_df = pd.DataFrame(data = scale.fit_transform(df) , columns= df.columns)
print(normal_df.shape)

pca = PCA(n_components=2,random_state=0)
pca_output = pca.fit_transform(normal_df)
print(pca.explained_variance_ratio_)

km = KMeans(n_clusters= 5,random_state= 0)
km.fit(pca_output)
clust_lable_dict = {0:'Good',1:'out',2:'Bad',3:'Very Good',4:'out',5:'l6',6:'l7',7:'l8'}
labels = [clust_lable_dict[i] for i in km.labels_]

#//Vitualize PCA-Kmeans Clustering
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('PCA Kmeans Clustering Vitualizations', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('PCA1', fontsize=16,fontweight='bold')
ax1.set_ylabel('PCA2', fontsize=16,fontweight='bold')
sns.scatterplot(
    #data=df,
    x=pca_output[:,0],
    #y='AI4 Oxygen content in waste nitrogen',
    y=pca_output[:,1],
    hue=labels,
    alpha = 0.9,
)
fig.tight_layout()
plt.savefig(f'{working_dir}/kmeans-output/pca_kmeans.jpg')
#plt.show()
fig.clear()

# add C5000 Total Current and Remove Outlaier
df['kmeans_labels'] = labels
df['C5000_total_current'] = df['CT001 Motor current (C5000A)']+df['CT001 Motor current (C5000B)']+df['CT001 Motor current (C5000C)']
df = df[df['kmeans_labels'] != 'out']
print(df.shape)

#// Histogram of O2 Purity By Kmeans Clusters
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('O2 Purity Groupby Kmeans', fontsize=18,fontweight='bold')
ax1,ax2,ax3 = fig.subplots(1,3,sharey=True)
ax1.set_xlabel('O2 Purity (%)', fontsize=16,fontweight='bold')
ax2.set_xlabel('O2 Purity (%)', fontsize=16,fontweight='bold')
ax3.set_xlabel('O2 Purity (%)', fontsize=16,fontweight='bold')
sns.histplot(
                data=df,
                x= 'AI1 Product gaseous (liquid) oxygen purity',
                hue= 'kmeans_labels',
                hue_order= ['Very Good'],
                alpha = 0.5,
                kde= True,
                ax= ax1
            )
sns.histplot(
                data=df,
                x= 'AI1 Product gaseous (liquid) oxygen purity',
                hue= 'kmeans_labels',
                hue_order= ['Good'],
                alpha = 0.5,
                kde= True,
                ax= ax2
            )
sns.histplot(
                data=df,
                x= 'AI1 Product gaseous (liquid) oxygen purity',
                hue= 'kmeans_labels',
                hue_order= ['Bad'],
                alpha = 0.5,
                kde= True,
                ax= ax3
            )
ax1.axvline(99.845, c='red')
ax1.annotate('99.84', xy =(99.84, 75),rotation = 90,ha='center', fontsize=18,alpha = 0.8)
ax2.axvline(99.815, c='red')
ax2.annotate('99.81', xy =(99.81, 75),rotation = 90,ha='center', fontsize=18,alpha = 0.8)
ax3.axvline(99.795, c='red')
ax3.annotate('99.79', xy =(99.79, 75),rotation = 90,ha='center', fontsize=18,alpha = 0.8)
fig.tight_layout()
plt.savefig(f'{working_dir}/kmeans-output/histogram-O2-purity-groupby-kmeans.jpg')
#plt.show()
fig.clear()

#// Histogram of Nitrogen Purity By Kmeans Clusters
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('nitrogen Purity Groupby Kmeans', fontsize=18,fontweight='bold')
ax1,ax2,ax3 = fig.subplots(1,3,sharey=True)
ax1.set_xlabel('N2 Purity (%)', fontsize=16,fontweight='bold')
ax2.set_xlabel('N2 Purity (%)', fontsize=16,fontweight='bold')
ax3.set_xlabel('N2 Purity (%)', fontsize=16,fontweight='bold')
sns.histplot(
                data=df,
                x= 'AI4 Oxygen content in waste nitrogen',
                hue= 'kmeans_labels',
                hue_order= ['Very Good'],
                alpha = 0.5,
                kde= True,
                ax= ax1
            )
sns.histplot(
                data=df,
                x= 'AI4 Oxygen content in waste nitrogen',
                hue= 'kmeans_labels',
                hue_order= ['Good'],
                alpha = 0.5,
                kde= True,
                ax= ax2
            )
sns.histplot(
                data=df,
                x= 'AI4 Oxygen content in waste nitrogen',
                hue= 'kmeans_labels',
                hue_order= ['Bad'],
                alpha = 0.5,
                kde= True,
                ax= ax3
            )
ax1.axvline(90.3, c='red')
ax1.annotate('90.3', xy =(90.2, 75),rotation = 90,ha='center', fontsize=18,alpha = 0.8)
ax2.axvline(90, c='red')
ax2.annotate('90', xy =(89.9, 75),rotation = 90,ha='center', fontsize=18,alpha = 0.8)
ax3.axvline(90.15, c='red')
ax3.annotate('90.15', xy =(90.1, 75),rotation = 90,ha='center', fontsize=18,alpha = 0.8)
fig.tight_layout()
plt.savefig(f'{working_dir}/kmeans-output/histogram-N2-purity-groupby-kmeans.jpg')
#plt.show()
fig.clear()

#// C5000 total Amp VS O2 Purity
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('C5000 Amp. VS O2 Purity', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('Amp.', fontsize=16,fontweight='bold')
sns.kdeplot(
    data=df,
    x='C5000_total_current',
    hue='kmeans_labels',
    fill=True,
    alpha = 0.05,
)
fig.tight_layout()
plt.savefig(f'{working_dir}/kmeans-output/C5000-Amp-VS-O2-Purity.jpg')
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
fig.tight_layout()
plt.savefig(f'{working_dir}/kmeans-output/tubine-speed-O2-purity.jpg')
#plt.show()
fig.clear()

#//Turbine Speed VS O2 purity groupby kmeans
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Trbine Speed Kmeans Clustering Vitualizations', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('O2 Purity', fontsize=16,fontweight='bold')
ax1.set_ylabel('RPM', fontsize=16,fontweight='bold')
sns.scatterplot(
    data=df,
    y='SIC401B - PV Air turbine/booster T/C2000B speed',
    x='AI4 Oxygen content in waste nitrogen',
    hue='kmeans_labels',
    alpha = 0.9,
)
fig.tight_layout()
plt.axhline(27793, c='green',alpha=0.4)
plt.savefig(f'{working_dir}/kmeans-output/turbine-speed-vs-o2-purity.jpg')
#plt.show()
fig.clear()