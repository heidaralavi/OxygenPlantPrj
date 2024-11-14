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
        return "Product < 29500 Nm3/h"
    if x <= 29600 :
        return "29500 < Product < 29600 Nm3/h"
    if x <= 29700 :
        return "29600 < Product < 29700 Nm3/h"
    if x <= 29800 :
        return "29700 < Product < 29800 Nm3/h"
    if x <= 29900 :
        return "29800 < Product < 29900 Nm3/h"
    if x <= 30000 :
        return "29900 < Product < 30000 Nm3/h"
    if x > 30000 :
        return "Product > 30000 Nm3/h"

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#// calculating Correlation Coefficient
corr = normal_data.corr()
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Correlation Coefficient', fontsize=16,fontweight='bold')
ax1 = fig.subplots(1,1)
sns.heatmap(corr, cbar=False,square= False, fmt='.1f', annot=True, annot_kws={'size':3}, cmap='Greens',ax= ax1)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/Correlation_Coefficient.jpg')
#plt.show()
fig.clear()

#// Histogram of Air production
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Air Production After Dust Filter', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('FI580   (Nm3/h)', fontsize=16,fontweight='bold')
sns.histplot(data=df["FI580 Vessel V5000B pressure"], binwidth=100 ,ax= ax1)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/Production_Hist.jpg')
#plt.show()
fig.clear()

#// Compressors C5000 Motor Current
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Compressors C5000 Motor Current', fontsize=16,fontweight='bold')
gs = GridSpec(2,3)
ax1=fig.add_subplot(gs[0,0])
ax2=fig.add_subplot(gs[0,1], sharey=ax1)
ax3=fig.add_subplot(gs[0,2], sharey=ax1)
ax4=fig.add_subplot(gs[1,:])
ax1.scatter(x=df['tarikh'], y= df['CT001 Motor current (C5000A)'], color='blue',alpha=0.5)
ax1.set_title('C5000A')
ax1.set_ylabel('Amp.')
ax1.tick_params(axis="x",labelrotation=45)
ax2.scatter(x=df['tarikh'], y= df['CT001 Motor current (C5000B)'], color='red',alpha=0.5)
ax2.set_title('C5000B')
ax2.tick_params(axis="x",labelrotation=45)
ax3.scatter(x=df['tarikh'], y= df['CT001 Motor current (C5000C)'], color='green',alpha=0.5)
ax3.set_title('C5000C')
ax3.tick_params(axis="x",labelrotation=45)
ax4.scatter(x=df['tarikh'], y= df['C5000_total_current'], color='green',alpha=0.5)
ax4.set_title('Consumption C5000')
ax4.tick_params(axis="x",labelrotation=45)
ax4.set_xlabel('time', fontsize=16,fontweight='bold')
ax4.set_ylabel('Amp.')
fig.tight_layout()
plt.axhline(df['C5000_total_current'].mean(), c='red')
plt.axhline(df['C5000_total_current'].mean()+df['C5000_total_current'].std(), c='red',linestyle='--')
plt.axhline(df['C5000_total_current'].mean()-df['C5000_total_current'].std(), c='red',linestyle='--')
#plt.annotate('27793 RPM', xy =(27795, 0.0023),rotation = 90,ha='center', fontsize=6,alpha = 0.8)
#plt.savefig(f'{working_dir}/fig/C5000_Motor_Currents.jpg')
#plt.show()
fig.clear()

#// Product VS C5000 Compressor Motor Currents
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('C5000A Comp. VS Air Production', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('Amp.', fontsize=16,fontweight='bold')
ax1.set_ylabel('FI580   (Nm3/h)', fontsize=16,fontweight='bold')
sns.kdeplot(data=df, x='CT001 Motor current (C5000A)', y='FI580 Vessel V5000B pressure',fill=True, levels=10)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/Product-Current_C5000A.jpg')
#plt.show()
fig.clear()

#//Air turbine outlet pressure VS products
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Tubine Outlet Pressure VS Air Production', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('bar', fontsize=16,fontweight='bold')
sns.kdeplot(
    data=df,
    x='PI210 Air turbine/booster outlet pressure',
    hue='levels',
    hue_order=['Product < 29500 Nm3/h','29500 < Product < 29600 Nm3/h',
               '29600 < Product < 29700 Nm3/h','29700 < Product < 29800 Nm3/h',
               '29800 < Product < 29900 Nm3/h','29900 < Product < 30000 Nm3/h',
               'Product > 30000 Nm3/h'],
    fill=True,
    alpha = 0.05,
)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/trubine pressure Vs Products.jpg')
#plt.show()
fig.clear()

#//C5000 Compressor VS O2 Purity
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Amp. C5000 Comp. VS O2 Purity', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('Amp.', fontsize=16,fontweight='bold')
sns.kdeplot(
    data=df,
    x='C5000_total_current',
    hue='O2_Purity',
    fill=True,
    alpha = 0.05,
)
plt.axvline(290, c='green')
plt.annotate('290 Amp.', xy =(289.5, 0.05),rotation = 90,ha='center', fontsize=20,alpha = 0.8)
plt.axvline(292.5, c='green')
plt.annotate('292.5 Amp.', xy =(292, 0.05),rotation = 90,ha='center', fontsize=20,alpha = 0.8)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/C5000 Amp VS O2 purity.jpg')
#plt.show()
fig.clear()

#// Air turbine outlet Temprature VS pressure
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Air turbine outlet', fontsize=18,fontweight='bold')
(ax1,ax2) = fig.subplots(1,2,sharex=True,sharey=True)
ax1.scatter(x=df['PI210 Air turbine/booster outlet pressure'], y= df['TI210A Air turbine T2000A outlet temperature'], color='blue', label='y1',alpha=0.5)
ax1.set_title('Turbine A', fontsize=16,fontweight='bold')
ax1.set_ylabel('Temp.', fontsize=16,fontweight='bold')
ax2.scatter(x=df['PI210 Air turbine/booster outlet pressure'], y= df['TI210B Air turbine T2000A outlet temperature'], color='red', label='y2',alpha=0.5)
ax2.set_title('Turbine B', fontsize=16,fontweight='bold')
ax2.set_xlabel('Pressure (bar)', fontsize=16,fontweight='bold')
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/trubine pressure Vs Temprature.jpg')
#plt.show()
fig.clear()


#//pie chart for product-current
fig = plt.figure(figsize=(20,11),dpi=300)
fig.suptitle('Amp. C5000 Comp. VS Air Production', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
#print(df.groupby("levels")['C5000_total_current'].mean().index)
ax1.pie(
    x=df.groupby("levels")['C5000_total_current'].mean(),
    autopct = '%.2f%%',
    textprops={'fontsize': 18},
    labels=df.groupby("levels")['C5000_total_current'].mean().index,
    )
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/product_current_pie.jpg')
#plt.show()
fig.clear()


#// Histogram of O2 purity
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('O2 Purity', fontsize=20,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('AI1 Product gaseous (liquid) oxygen purity', fontsize=16,fontweight='bold')
sns.histplot(data=df["AI1 Product gaseous (liquid) oxygen purity"] ,ax= ax1)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/O2_Purity_Hist.jpg')
#plt.show()
fig.clear()

#//pie chart of O2 Purity conditions
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('O2 Purity', fontsize=30,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.pie(
    x=df['O2_Purity'].value_counts(),
    autopct = '%.2f%%',
    textprops={'fontsize': 28},
    labels=df['O2_Purity'].value_counts().index,
    )
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/O2_purity_pie.jpg')
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
    hue='O2_Purity',
    fill=True,
    alpha = 0.05,
)
plt.axvline(27793, c='green')
plt.annotate('27793 RPM', xy =(27787, 0.0023),rotation = 90,ha='center', fontsize=18,alpha = 0.8) 
#plt.axvline(210, c='red')
#plt.annotate('224', xy =(224, 0.009),rotation = 90,ha='center', fontsize=16) 
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/tubine-speed-O2-purity.jpg')
#plt.show()
fig.clear()


#//pie chart for product levels
fig = plt.figure(figsize=(20,11),dpi=300)
fig.suptitle('Air Production Levels', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
#print(df.groupby("levels")['C5000_total_current'].mean().index)
ax1.pie(
    x=df.groupby("levels")['FI580 Vessel V5000B pressure'].sum(),
    autopct = '%.2f%%',
    textprops={'fontsize': 18},
    labels=df.groupby("levels")['FI580 Vessel V5000B pressure'].sum().index,
    )
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/product_levels_pie.jpg')
#plt.show()
fig.clear()

#//C5000 Amp. VS Production
fig = plt.figure(figsize=(15,11),dpi=300)
fig.suptitle('Amp. C5000 Comp. VS Production', fontsize=18,fontweight='bold')
ax1 = fig.subplots(1,1)
ax1.set_xlabel('Amp.', fontsize=16,fontweight='bold')
sns.kdeplot(
    data=df,
    x='C5000_total_current',
    hue='levels',
    hue_order=['Product < 29500 Nm3/h','29500 < Product < 29600 Nm3/h',
               '29600 < Product < 29700 Nm3/h','29700 < Product < 29800 Nm3/h',
               '29800 < Product < 29900 Nm3/h','29900 < Product < 30000 Nm3/h',
               'Product > 30000 Nm3/h'],
    fill=True,
    alpha = 0.05,
)
plt.axvline(289.2, c='green')
plt.annotate('289.2 Amp.', xy =(288.5, 0.035),rotation = 90,ha='center', fontsize=20,alpha = 0.8)
plt.axvline(292.5, c='green')
plt.annotate('292.5 Amp.', xy =(292, 0.035),rotation = 90,ha='center', fontsize=20,alpha = 0.8)
fig.tight_layout()
#plt.savefig(f'{working_dir}/fig/C5000 Amp VS O2 production.jpg')
#plt.show()
fig.clear()

#df.to_excel(f"{working_dir}/fig/output.xlsx",index=False)

