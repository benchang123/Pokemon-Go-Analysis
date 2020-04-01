import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Read in data
df = pd.read_csv('pokemoncomplete.csv', index_col=0, encoding= 'unicode_escape')

total_stats=df.sort_values("Total",ascending=False)

df.to_numpy
df.describe()
df.sort_values(by="Speed",ascending=False)

# df[df['Total'] > 599]

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
plt.figure(0,figsize=(10,6))
# Pre-format DataFrame
stats_df = df.drop(['Legendary', 'Generation'], axis=1)
# New boxplot using stats_df
sns.boxplot(data=stats_df, palette=pkmn_type_colors)

# Violin plot with Pokemon color palette
plt.figure(1,figsize=(10,6))
sns.violinplot(x='Type 1', y='Total', data=df,palette=pkmn_type_colors) # Set color palette

# Set figure size with matplotlib
plt.figure(2,figsize=(10,6))
sns.swarmplot(x='Type 1', y='Total', data=df,palette=pkmn_type_colors)

plt.figure(3,figsize=(10,6))
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