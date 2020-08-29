import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn import linear_model as m

#Set filepath to read in data
default_path='/Users/bench/Documents/Python Scripts/Pokemon'
os.chdir(default_path)

# Read in data
poke_game = pd.read_csv('pokemoncomplete.csv', index_col=0, encoding= 'unicode_escape')

poke_go = pd.read_csv('pogo.csv', index_col=0, encoding= 'unicode_escape')

# EDA
poke_go=poke_go.reset_index()
poke_game.drop(columns=['Type 1','Type 2'], inplace=True)
poke_game.rename(columns={'Attack':'Attack_g','Defense':'Defense_g'}, inplace=True)
poke_go.drop(columns=['Legendary'], inplace=True)
poke_game.loc[poke_game.index.isin([647,648,649]),['Legendary']]=True

# Inspect NaN
display(poke_go.isna().sum())
poke_game.isna().sum()

poke_go['Name']=poke_go['Name'].str.lower()
poke_game['Name']=poke_game['Name'].str.lower()

# Combine Dataframes

poke_go_compare=poke_go.merge(poke_game,left_on='Name',right_on='Name', how='inner')
poke_go_compare.drop(columns=['Generation_y'], inplace=True)
poke_go_compare.rename(columns={'Generation_x':'Generation'}, inplace=True)

# Find Missing Pokemons during Merge
missing=poke_go.loc[poke_go['Pokedex'].isin(poke_go_compare['Pokedex'])==False,'Name']
missing

poke_game1=poke_game.reset_index()
poke_go_missing=poke_go.merge(poke_game1,left_on='Pokedex',right_on='#', how='inner')
poke_go_missing.drop_duplicates('Name_x', keep='first', inplace=True)

poke_go_missing=poke_go_missing[poke_go_missing['Name_x'].isin(missing)]

poke_go_missing.drop(columns=['Generation_y','Name_y','#'], inplace=True)
poke_go_missing.rename(columns={'Generation_x':'Generation','Name_x':'Name'}, inplace=True)

poke_go_complete=pd.concat([poke_go_compare, poke_go_missing], axis=0)

# Eventually try to add missing pokemons!!!

poke_go_complete=poke_go.merge(poke_game,left_on='Name',right_on='Name', how='inner')
poke_go_complete.drop(columns=['Generation_y'], inplace=True)
poke_go_complete.rename(columns={'Generation_x':'Generation'}, inplace=True)

# Visualizations
colInterestgame=['MaxCP','HP','Attack_g','Defense_g','Sp. Atk','Sp. Def','Speed']
colInterestgo=['MaxCP','Attack','Defense','Stamina']

sns.heatmap(poke_go_complete[colInterestgo].corr(),annot=True)

sns.pairplot(poke_go_complete.loc[:,colInterestgo])

sns.heatmap(poke_go_complete[colInterestgame].corr(),annot=True)

sns.pairplot(poke_go_complete.loc[:,colInterestgame])

sns.distplot(poke_go_complete.MaxCP).set_title('Distribution of Max CP')

ax=sns.boxplot(x='Primary',y='MaxCP',data=poke_go)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax2=sns.boxplot(x='Secondary',y='MaxCP',data=poke_go)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)

# CP by Type
ax=sns.boxplot(x='Primary',y='MaxCP',data=poke_go)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.figure()
ax2=sns.boxplot(x='Secondary',y='MaxCP',data=poke_go)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)

# Freq by Type
primarycount=poke_go_complete.groupby('Primary').size().sort_values(ascending=False)
ax3=sns.barplot(x=primarycount.index,y=primarycount)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)

primarycount=poke_go_complete.groupby('Secondary').size().sort_values(ascending=False)
ax3=sns.barplot(x=primarycount.index,y=primarycount)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)

# Modeling using Pokemon Game Data

def rmse(y_pred,y_actual):
    return np.sqrt(np.mean((y_pred-y_actual)**2))

model=m.LinearRegression()
model.fit(poke_go_complete.loc[:,'HP':'Speed'],poke_go_complete['MaxCP'])
poke_go_complete['predCP']=model.predict(poke_go_complete.loc[:,'HP':'Speed'])

sns.lmplot(data=poke_go_complete,x='predCP',y='MaxCP')

error=round(rmse(poke_go_complete['predCP'],poke_go_complete['MaxCP']),3)
print(f'The RMSE is {error}')

print(f'The coefficient for the HP, Attack, Defense, Special Attack, Special Defense, and Speed are {np.around(model.coef_,2)} respectively')

#Residual Plot

residual=poke_go_complete.loc[:,['MaxCP','predCP']].diff(axis=1)
residual.drop(columns='MaxCP',inplace=True)
poke_go_complete['Game Residual']=residual.rename(columns={'predCP':'Residual'})
residual

sns.lmplot(data=poke_go_complete,x='predCP',y='Game Residual')

# Model using Pokemon Go Stats

model2=m.LinearRegression()
model2.fit(poke_go_complete.loc[:,['Attack','Defense','Stamina']],poke_go_complete['MaxCP'])
poke_go_complete['pokegopred']=model2.predict(poke_go_complete.loc[:,['Attack','Defense','Stamina']])

sns.lmplot(x='pokegopred',y='MaxCP',data=poke_go_complete)

error2=rmse(poke_go_complete['pokegopred'],poke_go_complete['MaxCP'])
print(f'The RMSE is {round(error2,2)}')

print(f'The coefficient for Attack, Defense, Stamina are {np.around(model2.coef_,2)} respectively')

#Residual Plot

residual=poke_go_complete.loc[:,['MaxCP','pokegopred']].diff(axis=1)
residual.drop(columns='MaxCP',inplace=True)
poke_go_complete['Go Residual']=residual.rename(columns={'pokegopred':'Residual'})

sns.lmplot(data=poke_go_complete,x='pokegopred',y='Go Residual')

# 2nd Model using Pokemon Go Stats

ohc=pd.get_dummies(poke_go_complete['Primary'],drop_first=True)
poke_go_complete_ohc=pd.concat([poke_go_complete, ohc], axis=1)
interestedcolumns=poke_go_complete_ohc.iloc[:,np.r_[3,4,2,8,9,26:42]]

model3=m.LinearRegression()
model3.fit(interestedcolumns,poke_go_complete_ohc['MaxCP'])
poke_go_complete_ohc['pokegopred2']=model3.predict(interestedcolumns)

sns.lmplot(x='pokegopred2',y='MaxCP',data=poke_go_complete_ohc)

error3=rmse(poke_go_complete_ohc['pokegopred2'],poke_go_complete_ohc['MaxCP'])
print(f'The RMSE is {round(error3,2)}')

residual=poke_go_complete_ohc.loc[:,['MaxCP','pokegopred2']].diff(axis=1)
residual.drop(columns='MaxCP',inplace=True)
poke_go_complete_ohc['Go Residual 2']=residual.rename(columns={'pokegopred2':'Residual'})

sns.lmplot(data=poke_go_complete_ohc,x='pokegopred',y='Go Residual 2')

# Determine Pokemon with Biggest Error

biggestDiff=poke_go_complete.loc[:,['MaxCP','predCP']].diff(axis=1).sort_values('predCP',ascending=False).head(10).index

poke_go_complete.loc[biggestDiff,'Name']

# Predict CP for Gen 6

gen6poke=poke_game.loc[poke_game['Generation']==6,:]

gen6poke['CP']=model.predict(gen6poke.loc[:,'HP':'Speed'])

gen6poke.sort_values('CP',ascending=False).loc[:,['Name','CP']].head(10)

## Determine strongest Pokemon for Pokemon Games
# Pokemon GO

#Standardize stats
def standardize(data):
    return (data-np.mean(data))/np.std(data)
poke_go_complete_scaled = pd.DataFrame(standardize(poke_go_complete.loc[:,['Attack','Defense','Stamina']]))

poke_go_complete_scaled['Strength']= poke_go_complete_scaled.loc[:,'Attack':'Stamina']@model2.coef_.T
poke_go_complete_scaled['Name']=poke_go_complete['Name']
poke_go_complete_scaled['Primary']=poke_go_complete['Primary']
poke_go_complete_scaled['Secondary']=poke_go_complete['Secondary']
poke_go_complete_scaled['Legendary']=poke_go_complete['Legendary']

# Strongest Pokemons
poke_go_complete_scaled.sort_values("Strength", ascending=False).head()
poke_go_complete_scaled[poke_go_complete_scaled['Legendary']==False].sort_values("Strength", ascending=False).head()

# Weakest Pokemons
poke_go_complete_scaled.sort_values("Strength", ascending=True).head()
