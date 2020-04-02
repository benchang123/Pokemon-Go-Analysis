import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# Read in data
df = pd.read_csv('C:/Users/bench/Documents/Python Scripts/Scripts/Pokemon/pokemongo.csv', index_col=0, encoding= 'unicode_escape')
df.describe()
df["Total"]=df['Att']+df['Def']+df['Sta']

df.sort_values("Total",ascending=False)

plt.figure(0,figsize=(15,5))
sns.lmplot(x='Att', y='MaxCP', data=df)

plt.figure(1,figsize=(15,5))
sns.lmplot(x='Def', y='MaxCP', data=df)

plt.figure(2,figsize=(15,5))
sns.lmplot(x='Sta', y='MaxCP', data=df)

plt.figure(3,figsize=(15,5))
sns.lmplot(x='Total', y='MaxCP', data=df)

sns.distplot(df.MaxCP)

# model = LinearRegression()
# X, y = df[['NumberofEmployees','ValueofContract']], df.AverageNumberofTickets
# model.fit(X, y)