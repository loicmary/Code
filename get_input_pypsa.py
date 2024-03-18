import pandas as pd
from entsoe import EntsoePandasClient
import os
from datetime import datetime
from requests_html import HTMLSession
import warnings
import time
import argparse
from functools import reduce
import numpy as np

def preprocessing_generation_dataframe(df):
    df_generation = df.drop(['Actual Consumption'], axis=1, level=1, inplace=False)
    df_generation.columns = ['_'.join(col) for col in df_generation.columns.values]
    df_generation.columns = df_generation.columns.str.replace('_Actual Aggregated', '')
    df_generation.fillna(value=0.0, inplace=True)
    return df_generation

def fillna_moving_average(df):
    if df.isna().any().sum()>0:
        return df.fillna((df.ffill() + df.bfill())/2, inplace=False)
    else:
        return df
    
def preprocessing(df):
    nb_lines=df.shape[0]
    if nb_lines<8784 and nb_lines!=8760:
        return fillna_moving_average(df.resample('H').mean())
    elif nb_lines>8784:
        return fillna_moving_average(df.resample('H').asfreq())
    else:
        return fillna_moving_average(df)
    
def clean_power_datasets(df):
    df_power=df.copy()
    for energy in ['gas', 'coal', 'hydro', 'marine', 'wind', 'solar', 'nuclear', 'biomass', 'waste', 'other', 'geothermal', 'oil', 'peat']:
        if df_power.columns.str.lower().str.contains(energy).sum()>1:
            columns_to_drop=list(df_power.columns[df_power.columns.str.lower().str.contains(energy)])
            df_power[energy]=df_power[df_power.columns[df_power.columns.str.lower().str.contains(energy)]].sum(axis=1)
            df_power.drop(columns=columns_to_drop, inplace=True)

        elif df_power.columns.str.lower().str.contains(energy).sum()==1:
            column=list(df_power.columns[df_power.columns.str.lower().str.contains(energy)])
            df_power.rename(columns={column[0]:energy}, inplace=True)

        else:
            pass
    return df_power
    


if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('-start', required=True, type=str)
    parser.add_argument('-end', required=True, type=str)
    parser.add_argument('-studied_country', required=True, type=str)
    parser.add_argument('-folder_name', required=True, type=str)
    parser.add_argument('-tz', required=True, type=str)
    args = parser.parse_args()

    begin=time.time()
    warnings.filterwarnings('ignore')

    session = HTMLSession(verify=False)
    client=EntsoePandasClient(api_key='70fec3b8-9274-4fa2-b777-93d8a0390cdb')

    start = pd.Timestamp(args.start, tz=args.tz)
    end = pd.Timestamp(args.end, tz=args.tz)
    country_code = args.studied_country
    folder_name = args.folder_name
    year=args.start[:4]

    # print('Starting the query Actual Power data')
    # df_actual_power=client.query_generation(country_code=country_code, start=start, end=end, psr_type=None)
    # if isinstance(df_actual_power.columns, pd.MultiIndex):
    #     df_actual_power=preprocessing_generation_dataframe(df_actual_power)

    # df_actual_power=preprocessing(clean_power_datasets(df_actual_power))
    # df_actual_power.to_csv(os.path.join(folder_name, f"actual_power_{country_code}_{year}.csv"), sep=';')
    # print(np.round((time.time()-begin)/60,2))
    # print('DF for Actual Power created')
    # time.sleep(10)

    # print('Nominal Power data')
    # df_nominal_power=clean_power_datasets(client.query_installed_generation_capacity(country_code=country_code, start=start, end=end, psr_type=None))
    # df_nominal_power.to_csv(os.path.join(folder_name, f"nominal_power_{country_code}_{year}.csv"), sep=';')
    # print(np.round((time.time()-begin)/60,2))
    # print('DF for nominal power')
    # time.sleep(10)

    # print('Forecast Load data')
    # df_forecast_load=preprocessing(client.query_load_forecast(country_code=country_code, start=start, end=end))
    # df_forecast_load.to_csv(os.path.join(folder_name, f"forecast_load_{country_code}_{year}.csv"), sep=';')
    # print(np.round((time.time()-begin)/60,2))
    # print('DF for Forecast load data')

    # df_pmax_pu = pd.DataFrame(index=df_actual_power.index)
    # for column in [x for x in df_nominal_power.columns if x in df_actual_power.columns]:
    #     df_pmax_pu[column] = df_actual_power[column]/df_nominal_power[column].iloc[0]

    # df_pmax_pu.to_csv(os.path.join(folder_name, f"pmax_pu_{country_code}_{year}.csv"), sep=';')

    df_forecast_production = client.query_generation_forecast(country_code=country_code, start=start, end=end)
    if isinstance(df_forecast_production, pd.DataFrame):
        df_forecast_production = preprocessing(df_forecast_production.drop(columns='Actual Consumption', inplace=False))
    else:
        df_forecast_production= preprocessing(df_forecast_production.to_frame())

    df_forecast_production.to_csv(os.path.join(folder_name, f"forecast_production_{country_code}_{year}.csv"), sep=';')



    #client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None)

