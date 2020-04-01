import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Read in data
df = pd.read_csv('pokemongo.csv', index_col=0, encoding= 'unicode_escape')
df.describe()
df["Total"]=df['Att']+df['Def']+df['Sta']
df

plt.figure(0,figsize=(15,5))
sns.swarmplot(x='Att', y='Max CP', data=df)

plt.figure(1,figsize=(15,5))
sns.swarmplot(x='Def', y='Max CP', data=df)

plt.figure(2,figsize=(15,5))
sns.swarmplot(x='Sta', y='Max CP', data=df)

plt.figure(3,figsize=(15,5))
sns.swarmplot(x='Total', y='Max CP', data=df)


