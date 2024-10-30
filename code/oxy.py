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
plt.savefig('../fig/pic.jpg')
plt.show()

#print(df["[0:20]"])
#plt.title('Oxygen Plant', fontsize=20,fontweight='bold')
#plt.xlim(datetime.datetime(2024,1,25,15,15,00),datetime.datetime(2024,1,25,20,0,0))
#plt.plot(df["tarikh"],df["[0:367]"])

#print (df.info())
df.to_excel("../fig/output.xlsx",index=False)