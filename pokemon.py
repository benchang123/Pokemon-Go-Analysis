import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('pokemoncomplete.csv', index_col=0, encoding= 'unicode_escape')
print(df.head())

total_stats=df.sort_values("Total",ascending=False)

df.to_numpy
df.describe()
df.sort_values(by="Speed")

df[df['Total'] > 599]





sns.lmplot(x='Attack', y='Defense', data=df,
           fit_reg=False)   # Color by evolution stage

# Pre-format DataFrame
stats_df = df.drop(['Legendary', 'Generation'], axis=1)
 
# New boxplot using stats_df
sns.boxplot(data=stats_df)

# Set theme
sns.set_style('whitegrid')

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
plt.figure(figsize=(10,6))
# Violin plot with Pokemon color palette
sns.violinplot(x='Type 1', y='Total', data=df, 
               palette=pkmn_type_colors) # Set color palette

sns.swarmplot(x='Type 1', y='Total', data=df, 
              palette=pkmn_type_colors)

# Set figure size with matplotlib
plt.figure(figsize=(10,6))
 
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