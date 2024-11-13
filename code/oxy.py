import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
import os

working_dir = os.getcwd()
def date_time_add(x):
    datetime_object = datetime.strptime(str(x), '%d.%m.%Y %H:%M:%S')
    return  datetime_object 

def normalization(dataframe):
    normalized_df=(dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
    #normalized_df.to_excel(f"{working_dir}/fig/Normalized.xlsx",index=False)
    return normalized_df

def set_oxygen_produce_levels(x):
    if x < 29500 :
        return "under levels"
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
        return "over levels"

def set_oxygen_purity_levels(x):
    if x < 99.75 :
        return "Purity < 99.75"
    if x <= 99.8 :
        return "99.75 < Purity < 99.80"
    if x <= 99.85 :
        return "99.80 < Purity < 99.85"
    if x > 99.85 :
        return "Purity > 99.85"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#//data base preparation

df = pd.read_csv(f"{working_dir}/data/Oxygen_Plant_24Days.csv")
#print(df.info())


#// add tarikh column
df["tarikh"] = df["time"].apply(date_time_add)
#print(df.info())

#// add C5000 total current
df['C5000_total_current'] = df['CT001 Motor current (C5000A)']+df['CT001 Motor current (C5000B)']+df['CT001 Motor current (C5000C)']
#print(df.tail())


#// Normaliz Data
normal_data = normalization(df.drop(["time","tarikh"] ,axis= 1))
#print(normal_data.head())

#// add products levels
df["levels"] = df["FI580 Vessel V5000B pressure"].apply(set_oxygen_produce_levels)
#print(df.tail())

#// add O2 Purity levels
df["O2_Purity"] = df["AI1 Product gaseous (liquid) oxygen purity"].apply(set_oxygen_purity_levels)
#print(df.tail())

#// calculating Correlation Coefficient
corr = normal_data.corr()
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Correlation Coefficient', fontsize=16,fontweight='bold')
ax1 = fig.subplots(1,1)
sns.heatmap(corr, cbar=False,square= False, fmt='.1f', annot=True, annot_kws={'size':3}, cmap='Greens',ax= ax1)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/Correlation_Coefficient.jpg')
plt.show()

#// Histogram of O2 production
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('O2 Production', fontsize=16,fontweight='bold')
ax1 = fig.subplots(1,1)
sns.histplot(data=df["FI580 Vessel V5000B pressure"], binwidth=100 ,ax= ax1)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/Production_Hist.jpg')
plt.show()

#// Compressors C5000 Motor Current
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Compressors C5000 Motor Current', fontsize=16,fontweight='bold')
gs = GridSpec(2,3)
ax1=fig.add_subplot(gs[0,0])
ax2=fig.add_subplot(gs[0,1], sharey=ax1)
ax3=fig.add_subplot(gs[0,2], sharey=ax1)
ax4=fig.add_subplot(gs[1,:])
ax1.scatter(x=df['tarikh'], y= df['CT001 Motor current (C5000A)'], color='blue', label='y1',alpha=0.5)
ax1.set_title('C5000A')
ax1.set_ylabel('Amp.')
ax1.tick_params(axis="x",labelrotation=45)
ax2.scatter(x=df['tarikh'], y= df['CT001 Motor current (C5000B)'], color='red', label='y2',alpha=0.5)
ax2.set_title('C5000B')
ax2.tick_params(axis="x",labelrotation=45)
ax3.scatter(x=df['tarikh'], y= df['CT001 Motor current (C5000C)'], color='green', label='y3',alpha=0.5)
ax3.set_title('C5000C')
ax3.tick_params(axis="x",labelrotation=45)
ax4.scatter(x=df['tarikh'], y= df['C5000_total_current'], color='green', label='y3',alpha=0.5)
ax4.set_title('Consumption C5000')
ax4.tick_params(axis="x",labelrotation=45)
ax4.set_xlabel('time', fontsize=16,fontweight='bold')
ax4.set_ylabel('Amp.')
fig.tight_layout()
plt.savefig(f'{working_dir}/fig/C5000_Motor_Currents.jpg')
plt.show()

#// Compressors C5000 Cumsuntion Motor Current
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Compressors C5000 Motor Current', fontsize=16,fontweight='bold')
ax1 = fig.subplots(1,1,sharex=True,sharey=True)
ax1.scatter(x=df['tarikh'], y= df['CT001 Motor current (C5000A)'], color='blue', label='y1',alpha=0.5)
ax1.set_title('C5000A')
ax1.set_ylabel('Amp.')
ax1.tick_params(axis="x",labelrotation=45)
ax3.tick_params(axis="x",labelrotation=45)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/C5000_Motor_Currents.jpg')
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
#plt.savefig(f'{working_dir}/fig/Product-Current_C5000A.jpg')
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
#plt.savefig(f'{working_dir}/fig/trubine pressure Vs Products.jpg')
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
#plt.savefig(f'{working_dir}/fig/trubine pressure Vs Temprature.jpg')
plt.show()


#//pie chart for product-current
plt.figure(figsize=(10,7),dpi=300)
df.groupby("levels")['C5000_total_current'].mean().plot.pie(autopct='%.2f', textprops={'fontsize': 18})
#plt.savefig(f'{working_dir}/fig/product_current_pie.jpg')
plt.show()

#// Histogram of O2 purity
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('O2 Production', fontsize=16,fontweight='bold')
ax1 = fig.subplots(1,1)
sns.histplot(data=df["AI1 Product gaseous (liquid) oxygen purity"] ,ax= ax1)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/O2_Purity_Hist.jpg')
plt.show()

#//pie chart of O2 Purity conditions
plt.figure(figsize=(10,7),dpi=300)
df['O2_Purity'].value_counts().plot.pie(autopct = '%.2f', textprops={'fontsize': 18})
#plt.savefig(f'{working_dir}/fig/O2_purity_pie.jpg')
plt.show()

#// Product VS C5000 Compressor Motor Currents
sns.set(rc = {'figure.figsize':(15,11),'figure.dpi':300})
sns.displot(
    data=df,
    x=df['SIC401B - PV Air turbine/booster T/C2000B speed'],
    hue=df['O2_Purity'],
    #hue_order=['L1 (29500-29600)Nm3/h','L2 (29600-29700)Nm3/h','L3 (29700-29800)Nm3/h','L4 (29800-29900)Nm3/h','L5 (29900-30000)Nm3/h'],
    kind="kde", height=6,
    fill=True,
    alpha = 0.05,
)
plt.axvline(27793, c='green')
plt.annotate('27793 RPM', xy =(27795, 0.0023),rotation = 90,ha='center', fontsize=6,alpha = 0.8) 
#plt.axvline(210, c='red')
#plt.annotate('224', xy =(224, 0.009),rotation = 90,ha='center', fontsize=16) 
#plt.savefig(f'{working_dir}/fig/tubine-speed-O2-purity.jpg')
plt.show()




#df.to_excel(f"{working_dir}/fig/output.xlsx",index=False)
