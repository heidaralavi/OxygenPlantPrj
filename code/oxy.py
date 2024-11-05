import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import MinMaxScaler



def date_time_add(x):
    start_time = datetime.datetime(2024,9,6,18,42,48)
    delta_time = datetime.timedelta( minutes= x )
    return  start_time + delta_time 

def normalization(dataframe):
    normalized_df=(dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
    #normalized_df.to_excel("../fig/Normalized.xlsx",index=False)
    return normalized_df

#//data base preparation
df = pd.read_csv("../data/oxygen-plant.csv")

#// change dataframe columns name 
df_colums = pd.read_csv("../data/tag-des.csv")
colums_dict = df_colums.set_index('Tag_ID').to_dict()
df.columns = df.columns.to_series().map(colums_dict['Des'])

#// add tarikh column
df["tarikh"] = df["Minuts_From_Start"].apply(date_time_add)
print(df.tail())



normal_data = normalization(df.drop("tarikh" ,axis= 1))
#print(normal_data.head())

corr = normal_data.corr()
plt.figure(figsize=(15,11),dpi=300)
plt.title('Correlation Coefficient', fontsize=20,fontweight='bold')
sns.heatmap(corr, cbar=False,square= True, fmt='.1f', annot=True, annot_kws={'size':6}, cmap='Greens')
#plt.savefig('../fig/pic.jpg')
plt.show()

plt.figure(figsize=(15,11),dpi=300)
plt.title('Moisture', fontsize=20,fontweight='bold')
sns.kdeplot(data=normal_data, x=normal_data.iloc[:,2] , y=normal_data.iloc[:,2],fill=True, levels=10,)
plt.show()


plt.figure(figsize=(15,11),dpi=300)
plt.title('Moisture', fontsize=20,fontweight='bold')
sns.displot(
    data=normal_data,
    x=normal_data.iloc[:,2],
    hue=normal_data.iloc[:,5],
    #hue_order=['Very Good','Good','Bad'],
    kind="kde", height=6,
    fill=True,
    alpha = 0.05,
)
plt.show()

#df.to_excel("../fig/output.xlsx",index=False)