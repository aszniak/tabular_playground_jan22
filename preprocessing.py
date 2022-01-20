import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime
pd.set_option('display.max_columns', None)


def convert_datetime(row):
    date_ = date.fromisoformat(row['date'])
    day = date_.day
    month = date_.month
    weekday = date_.weekday()
    month_day_tuple = (month, day)
    if row['country'] == 'Finland':
        bank_holidays = [(1, 1), (1, 6), (5, 1), (12, 6), (12, 25), (12, 26)]
        if month_day_tuple in bank_holidays:
            row['bank_holiday'] = 1
    if row['country'] == 'Sweden':
        bank_holidays = [(1, 1), (1, 6), (5, 1), (6, 6), (12, 25), (12, 26)]
        if month_day_tuple in bank_holidays:
            row['bank_holiday'] = 1
    if row['country'] == 'Norway':
        bank_holidays = [(1, 1), (5, 1), (5, 17), (12, 25), (12, 26)]
        if month_day_tuple in bank_holidays:
            row['bank_holiday'] = 1
    if weekday == 5:
        row['saturday'] = 1
    elif weekday == 6:
        row['sunday'] = 1
    if date_.month in [12, 1, 2]:
        row['winter'] = 1
    elif date_.month in [3, 4, 5]:
        row['spring'] = 1
    elif date_.month in [6, 7, 8]:
        row['summer'] = 1
    elif date_.month in [9, 10, 11]:
        row['autumn'] = 1
    timetuple = date_.timetuple()
    doy = timetuple.tm_yday
    row['sin_doy'] = np.sin(2 * np.pi * (doy / 365))
    row['cos_doy'] = np.cos(2 * np.pi * (doy / 365))
    row['sin_dom'] = np.sin(2 * np.pi * (day / 31))
    row['cos_dom'] = np.cos(2 * np.pi * (day / 31))
    row['sin_dow'] = np.sin(2 * np.pi * (weekday / 6))
    row['cos_dow'] = np.cos(2 * np.pi * (weekday / 6))
    return row


def append_weather(row, weather_df):
    selected = weather_df.loc[(weather_df['country'] == row['country']) & (weather_df['date'] == row['date'])]
    row['precipitation'] = selected['precipitation']
    row['snow_depth'] = selected['snow_depth']
    row['tavg'] = selected['tavg']
    return row


def append_gdp(row, gdp_df):
    row_year = date.fromisoformat(row['date']).year
    row_country = row['country']
    gdp_df = gdp_df[gdp_df.year == row_year]
    row['gdp'] = gdp_df[row_country].item()
    return row


def convert_date_format(row):
    row['date'] = datetime.strptime(row['date'], '%m/%d/%Y').date().isoformat()
    row['year'] = date.fromisoformat(row['date']).year
    return row


def convert_country(row):
    if row['country'] == 'Finland':
        row['finland'] = 1
    elif row['country'] == 'Norway':
        row['norway'] = 1
    elif row['country'] == 'Sweden':
        row['sweden'] = 1
    return row


def convert_product(row):
    if row['product'] == 'Kaggle Mug':
        row['mug'] = 1
    elif row['product'] == 'Kaggle Hat':
        row['hat'] = 1
    elif row['product'] == 'Kaggle Sticker':
        row['sticker'] = 1
    return row


def convert_store(row):
    if row['store'] == 'KaggleMart':
        row['store'] = 1
    else:
        row['store'] = 0
    return row


def preprocess():
    print(f"Preprocessing files, please stand by...")
    for set_type in ['train', 'test']:
        weather_df = pd.read_csv('nordics_weather.csv')
        weather_df = weather_df.apply(convert_date_format, axis=1)
        gdp_df = pd.read_csv('gdp.csv')

        df = pd.read_csv(f'{set_type}.csv')
        df = df.drop(columns=['row_id'])
        if set_type == "train":
            weather_df = weather_df[weather_df.year != 2019]
        elif set_type == "test":
            weather_df = weather_df[weather_df.year == 2019]
        print(f"Appending weather data for {set_type} set...")
        df = df.apply(append_weather, args=(weather_df,), axis=1)
        print(f"Appending GDP data for {set_type} set...")
        df = df.apply(append_gdp, args=(gdp_df,), axis=1)
        print(f"Converting datetime data for {set_type} set...")
        df = df.apply(convert_datetime, axis=1)
        df = df.apply(convert_country, axis=1)
        df = df.apply(convert_product, axis=1)
        df = df.apply(convert_store, axis=1)
        df = df.drop(columns=['date', 'country', 'product'])
        if set_type == "train":
            df = df.sample(frac=1).reset_index(drop=True)
            data_array = df.drop(columns=['num_sold']).to_numpy(dtype='float32')
        else:
            data_array = df.to_numpy(dtype="float32")
        np.save(f'{set_type}_data_array.npy', data_array)
        if set_type == "train":
            targets_array = df['num_sold'].to_numpy(dtype='float32')
            np.save(f'{set_type}_targets_array.npy', targets_array)
        print(f"Saved {set_type} arrays.")

