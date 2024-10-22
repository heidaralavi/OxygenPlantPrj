import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/app/data/oxygen-plant.csv")
print(df.info())
#print(df["[0:20]"])
plt.title('Oxygen Plant', fontsize=20,fontweight='bold')
plt.xlim(190,260)
plt.plot(df.index,df["[0:367]"])
plt.savefig('/app/fig/pic.jpg')
#print (df.info())