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

def set_oxygen_produce_levels(x):
    if x < 29500 :
        return None
    if x <= 29600 :
        return "Level1"
    if x <= 29700 :
        return "Level2"
    if x <= 29800 :
        return "Level3"
    if x <= 29900 :
        return "Level4"
    if x <= 30000 :
        return "Level5"
    if x > 30000 :
        return None



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

df["levels"] = df["FI580 Vessel V5000B pressure"].apply(set_oxygen_produce_levels)
print(df.tail())



corr = normal_data.corr()
plt.figure(figsize=(15,11),dpi=300)
plt.title('Correlation Coefficient', fontsize=20,fontweight='bold')
sns.heatmap(corr, cbar=False,square= True, fmt='.1f', annot=True, annot_kws={'size':6}, cmap='Greens')
#plt.savefig('../fig/pic.jpg')
#plt.show()

plt.figure(figsize=(15,11),dpi=300)
plt.title('Moisture', fontsize=20,fontweight='bold')
sns.histplot(data=df.iloc[:,3],binwidth=100 )
plt.show()


plt.figure(figsize=(15,11),dpi=300)
plt.title('Moisture', fontsize=20,fontweight='bold')
sns.kdeplot(data=df,  x=df.iloc[:,4],fill=True,hue=df['levels'],hue_order =['Level1','Level2','Level3','Level4','Level5'])
plt.show()

'''
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
#plt.show()

#df.to_excel("../fig/output.xlsx",index=False)
'''