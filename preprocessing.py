import pandas as pd
import numpy as np
import datetime
import calendar


import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def preprocessing_TECSV2(data, list_conditions, timezone):
    
    df_copy = data.copy()
    print(f'Number of sessions at the begining (including NANs) : {df_copy.shape[0]}')
    df_copy = df_copy.drop(['retiredDate','closeDate'], axis=1)
    df_copy = df_copy.dropna(subset=['maxPower'])
    
    time_columns = ['startDate', 'stopDate']
    for t in time_columns :
        #df_copy[t] = pd.to_datetime(df_copy[t], utc=False).dt.tz_localize(timezone)
        df_copy[t] = pd.to_datetime(df_copy[t], utc=False)
        
    n_session_start = df_copy.shape[0]    
    print(f'Number of sessions at the begining : {n_session_start}')
    

    
  
    
    df_copy['maxPower'] = df_copy['maxPower'].apply(lambda x: 11 if (x > 11 and x <=50) else x)
    
    
    df_copy['charging_time'] = np.round((df_copy['consumption']/(df_copy['maxPower']*1000))*3600)
    
    df_copy['doneCharging_Date'] = df_copy['startDate'] + df_copy['charging_time'].apply(lambda x:datetime.timedelta(seconds=x))
    
    df_copy['sojourn_time'] =  np.round((df_copy['stopDate']- df_copy['startDate']).dt.total_seconds())

    df_copy['idle_time'] = df_copy['sojourn_time'] - df_copy['charging_time']
    
    session_ids_to_keep = df_copy.query(''.join(list_conditions))['_id'].unique()
    df_copy = df_copy[(df_copy['_id'].isin(session_ids_to_keep))]
    
    
    
    df_copy = df_copy[(df_copy['sojourn_time'] <= 48*3600) & (df_copy['charging_time'] <= 48*3600) & (df_copy['idle_time'] <= 48*3600)] 
    
    df_copy["day"] = df_copy['startDate'].apply(lambda x: x.date())
    df_copy['weekday'] = df_copy['day'].apply(lambda x:calendar.day_name[x.weekday()])
    
    n_session_end = df_copy.shape[0]
    print(f'Number of sessions at the end : {n_session_end}')
    print(f'Percentage of exploitable sessions : {np.round((n_session_end/n_session_start)*100)} %')
    
    df_copy.rename(columns = {'maxPower':'Power'}, inplace = True)
    df_copy['Power'] = df_copy['Power'].astype('float64')
    df_copy['consumption'] = (df_copy['consumption']/1000)
    
    print("Consumption in kWh & maxPower in kW")
    return df_copy


def preprocessing_TECSV3(data, list_conditions, timezone):
    
    df_copy = data.copy()
    print(f'Number of sessions at the begining (including NANs) : {df_copy.shape[0]}')
    
    df_copy = df_copy.drop(['retiredDate','closeDate'], axis=1)
    
    df_copy = df_copy[df_copy['businessActivity'].isin(['B2BATHOME', 'PUBLIC', 'TERTIARY'])]
    time_columns = ['startDate', 'stopDate']
    for t in time_columns :
        # df_copy[t] = pd.to_datetime(df_copy[t], utc=False).dt.tz_localize(timezone)
        df_copy[t] = pd.to_datetime(df_copy[t], utc=False)
        
    n_session_start = df_copy.shape[0]    
    print(f'Number of sessions at the begining : {n_session_start}')
    

    df_copy.rename(columns = {'sessionMaxPower (wh)':'sessionMaxPower'}, inplace = True)

    def preprocessing_evse_max_power(x):
        if x <= 11 :
            return float(x*1000)
        if (x > 11) & (x<= 50) :
            return 11000.0
        else :
            return 0
    
    df_copy['sessionMaxPower'] = df_copy['sessionMaxPower'].fillna(df_copy['evseMaxPower'].apply(lambda x:preprocessing_evse_max_power(x)))
    df_copy.loc[df_copy['sessionMaxPower'] > df_copy['evseMaxPower']*1000, 'sessionMaxPower'] = df_copy['evseMaxPower'].apply(lambda x:preprocessing_evse_max_power(x))

    # df_copy['sessionMaxPower'] = df_copy['sessionMaxPower'].fillna(value=11000.0)
    # df_copy.loc[df_copy['sessionMaxPower'] > df_copy['evseMaxPower']*1000, 'sessionMaxPower'] = 11000.0
    df_copy = df_copy[(df_copy['sessionMaxPower'] > 0) & (df_copy['consumption']>0)]
    
    #keep sessionMaxPower <= 50kW
    df_copy = df_copy[df_copy['sessionMaxPower']<= 50000]



    
    df_copy['charging_time'] = np.round((df_copy['consumption']/(df_copy['sessionMaxPower']))*3600)
    
    df_copy['doneCharging_Date'] = df_copy['startDate'] + df_copy['charging_time'].apply(lambda x:datetime.timedelta(seconds=x))
    
    
    df_copy['sojourn_time'] =  np.round((df_copy['stopDate']- df_copy['startDate']).dt.total_seconds())

    df_copy['idle_time'] = df_copy['sojourn_time'] - df_copy['charging_time']
    
    df_copy['duration'] = (df_copy['stopDate'] - df_copy['startDate']).dt.total_seconds()/60
    
    
    session_ids_to_keep = df_copy.query(''.join(list_conditions))['_id'].unique()

  
    df_copy = df_copy[(df_copy['_id'].isin(session_ids_to_keep))]
    
    
    
    
    df_copy = df_copy[(df_copy['sojourn_time'] <= 48*3600) & (df_copy['charging_time'] <= 48*3600) & (df_copy['idle_time'] <= 48*3600)] 
    
    df_copy["day"] = df_copy['startDate'].apply(lambda x: x.date())
   
    df_copy['weekday'] = df_copy['day'].apply(lambda x:calendar.day_name[x.weekday()])
    
    n_session_end = df_copy.shape[0]
    print(f'Number of sessions at the end : {n_session_end}')
    print(f'Percentage of exploitable sessions : {np.round((n_session_end/n_session_start)*100)} %')
    
    #df_copy.rename(columns = {'sessionMaxPower':'Power'}, inplace = True)
    df_copy['sessionMaxPower'] = (df_copy['sessionMaxPower']/1000) #sessionMaxPower in kW
    df_copy['consumption'] = (df_copy['consumption']/1000) #consumption in kWh

    return df_copy