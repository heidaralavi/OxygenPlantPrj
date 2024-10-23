import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import MinMaxScaler

def date_time_add(x):
    start_time = datetime.datetime(2023,12,30,23,15,0)
    delta_time = datetime.timedelta( minutes= x )
    return  start_time + delta_time 

def normalization(dataframe):
    normalized_df=(dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
    return normalized_df

#//data base preparation
df = pd.read_csv("/app/data/oxygen-plant.csv")
df["tarikh"] = df["Minuts_From_Start"].apply(date_time_add)
#print(df.tail())



normal_data = normalization(df.drop("tarikh" ,axis= 1))
print(normal_data.head())

corr = normal_data.corr()
plt.figure(figsize=(15,11),dpi=80)
plt.title('Correlation Coefficient', fontsize=20,fontweight='bold')
sns.heatmap(corr, cbar=False,square= True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Greens')


#print(df["[0:20]"])
#plt.title('Oxygen Plant', fontsize=20,fontweight='bold')
#plt.xlim(datetime.datetime(2024,1,25,15,15,00),datetime.datetime(2024,1,25,20,0,0))
#plt.plot(df["tarikh"],df["[0:367]"])
plt.savefig('/app/fig/pic.jpg')
#print (df.info())
