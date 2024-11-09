import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


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
        return "L1 (29500-29600)Nm3/h"
    if x <= 29700 :
        return "L2 (29600-29700)Nm3/h"
    if x <= 29800 :
        return "L3 (29700-29800)Nm3/h"
    if x <= 29900 :
        return "L4 (29800-29900)Nm3/h"
    if x <= 30000 :
        return "L5 (29900-30000)Nm3/h"
    if x > 30000 :
        return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#//data base preparation
df = pd.read_csv("../data/oxygen-plant.csv")

#// change dataframe columns name 
df_colums = pd.read_csv("../data/tag-des.csv")
colums_dict = df_colums.set_index('Tag_ID').to_dict()
df.columns = df.columns.to_series().map(colums_dict['Des'])

#// add tarikh column
df["tarikh"] = df["Minuts_From_Start"].apply(date_time_add)
#print(df.tail())

#// Normaliz Data
normal_data = normalization(df.drop("tarikh" ,axis= 1))
#print(normal_data.head())

#// add products levels
df["levels"] = df["FI580 Vessel V5000B pressure"].apply(set_oxygen_produce_levels)
#print(df.tail())


#// calculating Correlation Coefficient
corr = normal_data.corr()
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Correlation Coefficient', fontsize=16,fontweight='bold')
ax1 = fig.subplots(1,1)
sns.heatmap(corr, cbar=False,square= True, fmt='.1f', annot=True, annot_kws={'size':3}, cmap='Greens',ax= ax1)
fig.tight_layout()
#plt.savefig('../fig/Correlation_Coefficient.jpg')
plt.show()

#// Histogram of O2 production
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('O2 Production', fontsize=16,fontweight='bold')
ax1 = fig.subplots(1,1)
sns.histplot(data=df["FI580 Vessel V5000B pressure"], binwidth=100 ,ax= ax1)
fig.tight_layout()
#plt.savefig('../fig/Production_Hist.jpg')
plt.show()

#// Compressors C5000 Motor Current
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Compressors C5000 Motor Current', fontsize=16,fontweight='bold')
(ax1,ax2,ax3) = fig.subplots(1,3,sharex=True,sharey=True)
ax1.scatter(x=df['Minuts_From_Start'], y= df['CT001 Motor current (C5000A)'], color='blue', label='y1',alpha=0.5)
ax1.set_title('C5000A')
ax1.set_ylabel('Amp.')
ax2.scatter(x=df['Minuts_From_Start'], y= df['CT001 Motor current (C5000B)'], color='red', label='y2',alpha=0.5)
ax2.set_title('C5000B')
ax2.set_xlabel('time (minutes)')
ax3.scatter(x=df['Minuts_From_Start'], y= df['CT001 Motor current (C5000C)'], color='green', label='y3',alpha=0.5)
ax3.set_title('C5000C')
fig.tight_layout()
#plt.savefig('../fig/C5000_Motor_Currents.jpg')
plt.show()

#// Product VS C5000 Compressor Motor Currents
sns.set(rc = {'figure.figsize':(15,11),'figure.dpi':300})
sns.displot(
    data=df,
    x=df['CT001 Motor current (C5000A)'],
    hue=df['levels'],
    hue_order=['L1 (29500-29600)Nm3/h','L2 (29600-29700)Nm3/h','L3 (29700-29800)Nm3/h','L4 (29800-29900)Nm3/h','L5 (29900-30000)Nm3/h'],
    kind="kde", height=6,
    fill=True,
    alpha = 0.05,
)
#plt.savefig('../fig/Product-Current_C5000A.jpg')
plt.show()

#//Air turbine outlet pressure VS products
sns.set(rc = {'figure.figsize':(15,11),'figure.dpi':300})
sns.displot(
    data=df,
    x=df['PI210 Air turbine/booster outlet pressure'],
    hue=df['levels'],
    hue_order=['L1 (29500-29600)Nm3/h','L2 (29600-29700)Nm3/h','L3 (29700-29800)Nm3/h','L4 (29800-29900)Nm3/h','L5 (29900-30000)Nm3/h'],
    kind="kde", height=6,
    fill=True,
    alpha = 0.05,
)
#plt.savefig('../fig/trubine pressure Vs Products.jpg')
plt.show()

#// Air turbine outlet Temprature VS pressure
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Air turbine outlet', fontsize=16,fontweight='bold')
(ax1,ax2) = fig.subplots(1,2,sharex=True,sharey=True)
ax1.scatter(x=df['PI210 Air turbine/booster outlet pressure'], y= df['TI210A Air turbine T2000A outlet temperature'], color='blue', label='y1',alpha=0.5)
ax1.set_title('Turbine A')
ax1.set_ylabel('Temp.')
ax2.scatter(x=df['PI210 Air turbine/booster outlet pressure'], y= df['TI210B Air turbine T2000A outlet temperature'], color='red', label='y2',alpha=0.5)
ax2.set_title('Turbine B')
ax2.set_xlabel('Pressure (bar)')
fig.tight_layout()
#plt.savefig('../fig/trubine pressure Vs Temprature.jpg')
plt.show()



#df.to_excel("../fig/output.xlsx",index=False)
