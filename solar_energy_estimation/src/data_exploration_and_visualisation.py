# import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set() # Revert to matplotlib defaults
plt.rcParams['figure.figsize'] = (16, 12)

# load dataset
cleaned_df = pd.read_csv(r'C:\Users\Dell\Documents\my_linux\repos_\large-scale-solar-energy-generation\solar_energy_estimation\data\processed\cleaned_solar_data.csv')
cleaned_df.head()

# Descriptive statistics summary
cleaned_df.describe()

# check the missing values
cleaned_df.isnull().sum()

# summary of dataframe
cleaned_df.info()

# A plot pf total irriadtion for each
rad_vs_month= cleaned_df.loc[:, ['Radiation', 'CloudOpacity', 'Month']].groupby('Month').sum()
rad_vs_month.plot()
plt.xticks(cleaned_df['Month'], rotation = 30)
plt.title('Monthly total solar irradiation in eastern Nigeria')
plt.show()

# A plot pf total irriadtion for each
rad_vs_month = cleaned_df.loc[:, ['CloudOpacity', 'Month']].groupby('Month').sum()
rad_vs_month.plot()
plt.xticks(cleaned_df['Month'], rotation = 30)
plt.title('Monthly total Cloud Opacity in eastern Nigeria')
plt.show()

# Daily solar irradiation plot
f, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(data = cleaned_df, x = "Date", y = "Radiation")

rad_avg= cleaned_df['Radiation'].mean()
ax.axhline(rad_avg, color='green', linestyle='--', label=f"Mean - {rad_avg:.2f}")

ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth=0.5, color='white',)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='white', alpha=0.7)

plt.show()
ax.legend()

# Histogram distribution of weather features and solar irradiation
f, ax = plt.subplots(1, 4, figsize = (20,4))
sns.histplot(cleaned_df['Radiation'], stat = "density", bins = 30, ax = ax[0])
sns.histplot(cleaned_df['Pressure'], stat = "density", bins = 30, ax = ax[1])
sns.histplot(cleaned_df['Humidity'], stat = "density", bins = 30, ax = ax[2])
sns.histplot(cleaned_df['Temperature'], stat = "density", bins = 30, ax = ax[3])
plt.show()

# Histogram plot of daily mean radiation
fig, ax = plt.subplots(figsize = (15, 6))
sns.barplot(x = "Month", y = 'Radiation', data = cleaned_df, palette = "BuPu")
ax.set_title('Daily Mean Radiation')
plt.show()

# feature correlation
corr = cleaned_df.corr(method = "pearson")

sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), cmap = "coolwarm", square = True, annot = True, fmt = ".2f")
plt.show()