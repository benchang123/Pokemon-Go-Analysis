import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn import metrics

# Read in data
df = pd.read_csv('C:/Users/bench/Documents/Python Scripts/Scripts/Pokemon/pokemongo.csv', index_col=0, encoding= 'unicode_escape')
df["Total"]=df['Att']+df['Def']+df['Sta']
df.describe()

df.sort_values("Total",ascending=False)

calclist=['Att','Def','Sta','Total']

for stat in calclist:
    slope_stat, intercept_stat, rvalue_stat, pvalue_stat, std_stat = scipy.stats.linregress(df[stat],df['MaxCP'])
    r_squared_stat=rvalue_stat**2
    ax=sns.lmplot(x=stat, y='MaxCP', data=df)
    ax.fig.suptitle(stat+" vs Max CP")
    y_pred=slope_stat*df[stat]+intercept_stat
    
    print("Statistical Values for "+stat)
    print("Slope: "+str(round(slope_stat,2)))
    print("Intercept: "+str(round(intercept_stat,2)))
    print("R^2: "+str(round(r_squared_stat,2)))
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(df['MaxCP'], y_pred),2)) 
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(df['MaxCP'], y_pred)),2))
    print('\n')

plt.figure(figsize=(15,5))
sns.distplot(df.MaxCP).set_title('Distribution of Max CP')

plt.figure(figsize=(15,5))
sns.distplot(df.Total).set_title('Distribution of Total')

plt.figure(figsize=(15,5))
sns.boxplot(df['Type1'],df['MaxCP'],data=df)
plt.figure(figsize=(15,5))
sns.boxplot(df['Type2'],df['MaxCP'],data=df)