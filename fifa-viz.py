# Library import

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.makedirs('plots', exist_ok=True)

# Load file
fifa = pd.read_csv('players_20.csv', header=0)

# Explore file
for col in fifa.columns:
    print(col)

print(fifa.shape)

# Select features and focus on players
fifa_light = fifa[['short_name', 'age', 'dob', 'height_cm', 'weight_kg', 'nationality', 'club',
                   'overall', 'potential', 'value_eur', 'wage_eur', 'player_positions', 'preferred_foot',
                   'international_reputation', 'weak_foot', 'skill_moves', 'release_clause_eur', 'team_position',
                   'team_jersey_number', 'joined', 'contract_valid_until', 'pace',
                   'shooting', 'passing', 'dribbling', 'defending', 'physic']]

print(fifa_light.head())
print(fifa_light.describe())

players = fifa_light[fifa_light['player_positions'] != 'GK']
print(players.head())
print(players.shape)
print(players.describe())
print(players.corr())

# Generate a custom diverging colormap and draw heatmap
corr = players.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
cmap = sns.diverging_palette(220, 5, as_cmap=True)
h = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
h.set_title('Correlation Heatmap')
plt.savefig('plots/corr_heatmap.png')
plt.clf()

# Multiple plots in 1 figure
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Pie for preferred foot
axes[0][0].pie(players['preferred_foot'].value_counts(), labels=players['preferred_foot']
               .value_counts().index.tolist(), autopct='%1.1f%%')
axes[0][0].set_title('Preferred foot share')
axes[0][0].legend()

# Histogram of height
axes[1][0].hist(players['height_cm'], bins=10, color='green')
axes[1][0].set_title('Height distribution')
axes[1][0].set_xlabel('Bins')
axes[1][0].set_ylabel('Height in cm')

# Histogram of age
axes[0][1].hist(players['age'], bins=10, color='blue')
axes[0][1].set_title('Age distribution')
axes[0][1].set_xlabel('Bins')
axes[0][1].set_ylabel('age')

# Scatterplot of dribbling and passing
axes[1][1].scatter(players['dribbling'], players['passing'], s=players['international_reputation'] ** 3, alpha=0.5,
                   c=players['overall'], cmap='RdPu', marker='D', label='International reputation')
axes[1][1].set_title('Dribbling vs. passing')
axes[1][1].set_xlabel('Dribbling rating')
axes[1][1].set_ylabel('Passing rating')
plt.legend()

plt.savefig('plots/multiple_plots.png')
plt.clf()

# Scatter of value and overall rating
sns.set(style="white")
sc = sns.relplot(x="overall", y="value_eur", hue="preferred_foot", size="age",
                 sizes=(20, 100), alpha=.5, palette="muted",
                 height=6, data=players)
sc.set_axis_labels("FIFA rating", "Value in euros")
plt.title('Scatter plot of FIFA rating and Value of player')
plt.savefig(f'plots/value_rating_scatter.png', dpi=300)
plt.clf()

# Correlations in skills and overall rating
skills = ['overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
sns.pairplot(players[skills], diag_kind='hist')
plt.title('Pairplot of rating and skills')
plt.savefig('plots/seaborn_pairplot.png')
plt.clf()

# Reputation, FIFA rating and foot
f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x='international_reputation', y='overall', hue='preferred_foot',
              data=players, jitter=0.2, palette="Set2", dodge=True)
plt.title('Reputation, FIFA rating and preferred foot')
plt.xlabel('International reputation')
plt.ylabel('FIFA rating')
plt.savefig('plots/seaborn_foot.png')
plt.clf()

# Joint plot
sns.jointplot("weight_kg", "height_cm", data=players, kind="reg", truncate=False,
              color="m", height=7)
plt.title('Weight and height distribution')
plt.xlabel('Weight in kg')
plt.ylabel('Height in cm')
plt.savefig('plots/weight_height.png')
plt.clf()

# Plots on same axes
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.scatter(players['overall'], players['value_eur'], alpha=0.2, label='Player value')
axes.scatter(players['overall'], players['release_clause_eur'], alpha=0.2, label='Release clause value')
axes.set_xlabel('FIFA rating')
axes.set_ylabel('Value / Release clause')
axes.set_title(f'Player rating vs. worth')
axes.legend()
plt.savefig('plots/same_axes.png')
plt.clf()

# 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(players['value_eur'], players['dribbling'], players['passing'], c='skyblue', s=60)
ax.view_init(30, 185)
ax.set_xlabel('Dribbling rating')
ax.set_ylabel('Value in eur')
ax.set_zlabel('Passing rating')
ax.set_title(f'Player value vs. dribbling and passing')
plt.savefig('plots/3d_plot.png')
plt.clf()
