import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from matplotlib import pyplot as plt
import os
from sklearn import metrics
from sklearn import linear_model

#set working path
default_path='C:/Users/bench/Documents/Python Scripts/Scripts/Pokemon'
os.chdir(default_path)


# Read in data
poke_game = pd.read_csv('pokemoncomplete.csv', index_col=0, encoding= 'unicode_escape')
poke_go = pd.read_csv('pokemongo.csv', index_col=0, encoding= 'unicode_escape')

# EDA
poke_go=poke_go.reset_index()

poke_go["Total"]=poke_go['Att']+poke_go['Def']+poke_go['Sta']
poke_game.rename(columns={'Total':'Total_game'},inplace=True)
#poke_go.info()

poke_go['Name']=poke_go['Name'].str.lower()
poke_game['Name']=poke_game['Name'].str.lower()

# Combine Dataframes

poke_go_compare=poke_go.merge(poke_game,left_on='Name',right_on='Name', how='inner')

# Find Missing Pokemons during Merge
missing_2=poke_go.loc[poke_go['Dex #'].isin(poke_go_compare['Dex #'])==False,'Name']

# Eventually try to add missing pokemons!

poke_go_complete=poke_go.merge(poke_game,left_on='Name',right_on='Name', how='inner')
ax=sns.pairplot(poke_go_compare)
#poke_go_complete.drop(columns=['Type1','Type2'], inplace=True)

    

# =============================================================================
# calclist=['Att','Def','Sta','Total']
# 
# for stat in calclist:
#     slope_stat, intercept_stat, rvalue_stat, pvalue_stat, std_stat = scipy.stats.linregress(poke_go[stat],poke_go['MaxCP'])
#     r_squared_stat=rvalue_stat**2
#     ax=sns.lmplot(x=stat, y='MaxCP', data=poke_go)
#     ax.fig.suptitle(stat+" vs Max CP")
#     y_pred=slope_stat*poke_go[stat]+intercept_stat
#     
#     print("Statistical Values for "+stat)
#     print("Slope: "+str(round(slope_stat,2)))
#     print("Intercept: "+str(round(intercept_stat,2)))
#     print("R^2: "+str(round(r_squared_stat,2)))
#     print('Mean Absolute Error:', round(metrics.mean_absolute_error(poke_go['MaxCP'], y_pred),2)) 
#     print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(poke_go['MaxCP'], y_pred)),2))
#     print('\n')
# 
# # Visualization
# 
# plt.figure(figsize=(15,5))
# sns.distplot(poke_go.MaxCP).set_title('Distribution of Max CP')
# 
# plt.figure(figsize=(15,5))
# sns.distplot(poke_go.Total).set_title('Distribution of Total')
# 
# plt.figure(figsize=(15,5))
# sns.boxplot(x='Type1',y='MaxCP',data=poke_go)
# plt.figure(figsize=(15,5))
# sns.boxplot(x='Type2',y='MaxCP',data=poke_go)
# =============================================================================
