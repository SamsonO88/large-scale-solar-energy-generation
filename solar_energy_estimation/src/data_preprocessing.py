# import libraries
import numpy as np
import pandas as pd

# read the dataframe
df = pd.read_csv(r'C:\Users\Dell\Documents\my_linux\repos_\large-scale-solar-energy-generation\solar_energy_estimation\data\raw\5.866302_7.509482_Solcast_PT60M.csv')

# printout the column names
df.columns

# select some features
df = df[['PeriodEnd', 'PeriodStart', 'AirTemp', 'CloudOpacity', 'Dhi', 'Dni', 'Ghi',
       'PrecipitableWater', 'RelativeHumidity', 'SurfacePressure',
       'WindDirection10m', 'WindSpeed10m']]

# randomly display 10 rows
df.sample(10)

old_names = ['PeriodEnd', 'PeriodStart', 'AirTemp', 'CloudOpacity', 'Dhi', 'Dni', 'Ghi','PrecipitableWater', 'RelativeHumidity',
             'SurfacePressure', 'WindDirection10m', 'WindSpeed10m']
new_names = ['PeriodEnd','PeriodStart','Temperature', 'CloudOpacity', 'DHI', 'DNI', 'Radiation', 'Precipitation', 'Humidity',
             'Pressure', 'WindDirection', 'WindSpeed']
df.rename(columns = dict(zip(old_names, new_names)), inplace = True)

# display dataframe
df.head()

# show a summary of the dataframe
df.info()

# interpret columns as appropriate data types to ensure compatibility
df['Radiation']     = df['Radiation'].astype(float)
df['Temperature']   = df['Temperature'].astype(float) # or int
df['Pressure']      = df['Pressure'].astype(float)
df['CloudOpacity']      = df['CloudOpacity'].astype(float)
df['Humidity']      = df['Humidity'].astype(int) # or int
df['WindDirection'] = df['WindDirection'].astype(float)
df['WindSpeed']     = df['WindSpeed'].astype(float)
df['PeriodStart'] = pd.to_datetime(df['PeriodStart']).dt.to_period('T').dt.to_timestamp()
df['PeriodEnd'] = pd.to_datetime(df['PeriodEnd']).dt.to_period('T').dt.to_timestamp()

# display dataframe summary
df.info()

# display top 6 rows
df.head()

# Create new features(year, month, day) from start and end time of data collection
df['YearPS'] = pd.to_datetime(df['PeriodStart']).dt.year
df['YearPE'] = pd.to_datetime(df['PeriodEnd']).dt.year

df['MonthPS'] = pd.to_datetime(df['PeriodStart']).dt.month
df['MonthPE'] = pd.to_datetime(df['PeriodEnd']).dt.month

df['HourPS'] = pd.to_datetime(df['PeriodStart']).dt.hour
df['HourPE'] = pd.to_datetime(df['PeriodEnd']).dt.hour

df.head(5)

# create a new feature for sunrise using the function below; fill time for sunrise
def time_to_sunrise(row):
    """
    A function for identifying time of sunrise.
    """
    value= 0
    if row['HourPS'] == 4 and row['HourPE'] == 5 and row['Radiation'] > 0:
        value= '05:00:00'
    elif row['HourPS'] == 5 and row['HourPE'] == 6 and row['Radiation'] > 0:
        value= '06:00:00'
    return value

#Create a new column using the function above
df['Sunrise'] = df.apply(time_to_sunrise, axis = 1)
df.head(5)

df.Sunrise.unique()

# create a new feature for sunset using the function below; fill time for sunset
def time_to_sunset(row):
    """
    A function for identifying time of sunset.
    """
    value= 0
    if row['HourPS'] == 17 and row['HourPE'] == 18 and row['Radiation']> 0:
        value= '18:00:00'
    elif row['HourPS'] == 18 and row['HourPE']== 19 and row['Radiation']> 0:
        value= '19:00:00'
    return value

#Create a new column using the function above
df['Sunset'] = df.apply(time_to_sunset, axis = 1)

df.Sunset.unique()

# show summary of our dataframe
df.info()

# changing integer values to datetime format
df_new = df.loc[~((df['Radiation'] == 0) & (df['DNI'] == 0) & (df['DHI'] == 0)),:]

# show summary of our dataframe
df_new.info()

# display 10 random samples of the dataframe
df_new.sample(10)

# convert datetime column of endperiod to just date
df_new['Date'] = pd.to_datetime(df_new['PeriodEnd']).dt.normalize()

# convert the column (it's a string) to datetime type
dt_series = pd.to_datetime(df_new['Date'])

# create datetime index passing the datetime series
dt_index = pd.DatetimeIndex(dt_series.values)

#
df_new1 = df_new.set_index(dt_index)

# display first 5 rows
df_new1.head()

#
df_new1.info()

df_new1['Date'] = pd.to_datetime(df_new1['PeriodEnd']).dt.date

for col in ['Sunrise', 'Sunset']:
    df_new1[col] = df_new1[col].astype(str)

agg_dict = {
    'Temperature':'mean',
    'CloudOpacity':'mean',
    'DHI':'mean',
    'DNI':'mean',
    'Radiation':'mean',
    'Precipitation':'mean',
    'Humidity':'mean',
    'Pressure':'mean',
    'WindDirection':'mean',
    'WindSpeed':'mean',
    'Sunrise':'max',
    'Sunset':'max',
}

df_grouped = df_new1.groupby(['Date']).agg(agg_dict)
df_grouped.reset_index(inplace = True)
df_grouped

df_grouped['Month'] = pd.to_datetime(df_grouped['Date']).dt.month

# display a summary of our dataframe
df_grouped.info()

# drop irrelevant features
new_df = df_grouped.drop(['Sunrise', 'Sunset'], axis = 1)

# display first 10 rows
new_df.head(10)

# show a summary of the dataframe
new_df.info()

# dropping ALL duplicate values exceept the last value
new_df = new_df[~new_df.Date.duplicated(keep = 'last')]

# show the number of rows and columns available
new_df.shape

# list out the column names
new_df.columns

# save cleaned dataset
new_df.to_csv(r'C:\Users\Dell\Documents\my_linux\repos_\large-scale-solar-energy-generation\solar_energy_estimation\data\processed\processed_solar_data.csv', index = False)
new_df.to_csv(r'C:\Users\Dell\Documents\my_linux\repos_\large-scale-solar-energy-generation\solar_energy_estimation\data\processed\cleaned_solar_data.csv', index = False)