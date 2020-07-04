import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Read in data
df = pd.read_csv('C:/Users/bench/Documents/Python Scripts/Scripts/Pokemon/pokemoncomplete.csv', index_col=0, encoding= 'unicode_escape')
df2 = pd.read_csv('pokemongo.csv', index_col=0, encoding= 'unicode_escape')

df2["Total"]=df2['Att']+df2['Def']+df2['Sta']
df2.describe()

df.sort_values(by="Total",ascending=False)
df['totatt']=df['Attack']+df['Sp. Atk']
df['totdef']=df['Defense']+df['Sp. Def']
df.describe()

pokemongolist=df2["Name"].tolist()
pokemonlist=[]

for pokemon in df['Name']:
    if pokemon.upper()==df2.loc[:,'Name']:
        pokemonlist=df['Name']


pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

# Set theme
sns.set_style('whitegrid')
plt.figure(figsize=(15,6))
# Pre-format DataFrame
stats_df = df.drop(['Legendary', 'Generation'], axis=1)
# New boxplot using stats_df
sns.boxplot(data=stats_df, palette=pkmn_type_colors)

plt.figure(figsize=(15,6))
# Create plot
sns.violinplot(x='Type 1',
               y='Total', 
               data=df, 
               inner=None, # Remove the bars inside the violins
               palette=pkmn_type_colors)
 
sns.swarmplot(x='Type 1', 
              y='Total', 
              data=df, 
              color='k', # Make points black
              alpha=0.7) # and slightly transparent
 
# Set title with matplotlib
plt.title('Total by Type')
