import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn import linear_model as m

#set working path
default_path='/Users/bench/Documents/Python Scripts/Pokemon'
os.chdir(default_path)


# Read in data
#https://gist.github.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6
poke_game = pd.read_csv('pokemoncomplete.csv', index_col=0, encoding= 'unicode_escape')
#https://www.kaggle.com/netzuel/pokmon-go-dataset-15-generations
poke_go = pd.read_csv('pogo.csv', index_col=0, encoding= 'unicode_escape')

# EDA
poke_go=poke_go.reset_index()
poke_game.drop(columns=['Type 1','Type 2','Legendary'], inplace=True)
poke_game.rename(columns={'Attack':'Attack_g','Defense':'Defense_g'}, inplace=True)

# Inspect NaN
poke_go.isna().sum()
poke_game.isna().sum()

poke_go['Name']=poke_go['Name'].str.lower()
poke_game['Name']=poke_game['Name'].str.lower()

# Combine Dataframes

poke_go_compare=poke_go.merge(poke_game,left_on='Name',right_on='Name', how='inner')
poke_go_compare.drop(columns=['Generation_y'], inplace=True)
poke_go_compare.rename(columns={'Generation_x':'Generation'}, inplace=True)

# Find Missing Pokemons during Merge
missing=poke_go.loc[poke_go['Pokedex'].isin(poke_go_compare['Pokedex'])==False,'Name']

# Eventually try to add missing pokemons!!!

poke_go_complete=poke_go.merge(poke_game,left_on='Name',right_on='Name', how='inner')
poke_go_complete.drop(columns=['Generation_y'], inplace=True)
poke_go_complete.rename(columns={'Generation_x':'Generation'}, inplace=True)

# Visualizations
colInterestgame=['MaxCP','HP','Attack_g','Defense_g','Sp. Atk','Sp. Def','Speed']
colInterestgo=['MaxCP','Attack','Defense','Stamina']

sns.heatmap(poke_go_complete[colInterestgo].corr(),annot=True)
sns.heatmap(poke_go_complete[colInterestgame].corr(),annot=True)

sns.pairplot(poke_go_complete.loc[:,colInterestgo])

sns.pairplot(poke_go_complete.loc[:,colInterestgame])

sns.distplot(poke_go_complete.MaxCP).set_title('Distribution of Max CP')

ax=sns.boxplot(x='Primary',y='MaxCP',data=poke_go)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax2=sns.boxplot(x='Secondary',y='MaxCP',data=poke_go)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)

#Legendary Pokemon by Type
primarycount=poke_go_complete.groupby('Primary').size().sort_values(ascending=False)
ax3=sns.barplot(x=primarycount.index,y=primarycount)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)

primarycount=poke_go_complete.groupby('Secondary').size().sort_values(ascending=False)
ax3=sns.barplot(x=primarycount.index,y=primarycount)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)

# Modeling using Pokemon Game Data

def rmse(y_pred,y_actual):
    return np.mean((y_pred-y_actual)**2)

model=m.LinearRegression()
model.fit(poke_go_complete.loc[:,'HP':'Speed'],poke_go_complete['MaxCP'])
poke_go_complete['predCP']=model.predict(poke_go_complete.loc[:,'HP':'Speed'])

sns.lmplot(data=poke_go_complete,x='predCP',y='MaxCP')

error=rmse(poke_go_complete['predCP'],poke_go_complete['MaxCP'])
error

model.coef_

# Model using Pokemon Go Stats

model2=m.LinearRegression()
model2.fit(poke_go_complete.loc[:,['Attack','Defense','Stamina']],poke_go_complete['MaxCP'])
poke_go_complete['pokegopred']=model2.predict(poke_go_complete.loc[:,['Attack','Defense','Stamina']])

sns.lmplot(x='pokegopred',y='MaxCP',data=poke_go_complete)

error2=rmse(poke_go_complete['pokegopred'],poke_go_complete['MaxCP'])
error2

model2.coef_

# Determine Pokemon with Biggest Error

biggestDiff=poke_go_complete.loc[:,['MaxCP','predCP']].diff(axis=1).sort_values('predCP',ascending=False).head(20).index

poke_go_complete.loc[biggestDiff,'Name']

# Predict CP for Gen 6

gen6poke=poke_game.loc[poke_game['Generation']==6,:]

gen6poke['CP']=model.predict(gen6poke.loc[:,'HP':'Speed'])

## Determine strongest Pokemon for Pokemon Games
# Pokemon GO

#Standardize stats
def normalization(data):
    return (data-np.mean(data))/np.std(data)
poke_go_complete_scaled = pd.DataFrame(normalization(poke_go_complete.loc[:,['Attack','Defense','Stamina']]))

poke_go_complete_scaled['OverallStrength']=np.sum(poke_go_complete_scaled,axis=1)
poke_go_complete_scaled['Name']=poke_go_complete['Name']
poke_go_complete_scaled['Primary']=poke_go_complete['Primary']
poke_go_complete_scaled['Secondary']=poke_go_complete['Secondary']
    