
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from plotly.subplots import make_subplots
import time
import datetime

# Shiftable energy



def shiftable_energy(data_preprocessed, power, energy_delivered):
    """returns a copy of data_preprocessed with the new column containing the shiftable energy for each sessions
    Args:
    data_preprocessed (pandas.Dataframe) : dataset preprocessed containing all the available sessions
    power (str) : name of the column related to power (must be in kW)
    energy_delivered (str) : name of the column related to the energy delivered during the session (must be in kWh)
    """
    
    final_df = data_preprocessed.copy()
    E_shiftable = []
    for row in tqdm(data_preprocessed.to_dict('records')):
        condition = row['idle_time'] < row['charging_time']
        if condition :
            E_shiftable.append(row[power]*(row['idle_time']/3600))
        else :
            E_shiftable.append(row[energy_delivered])

    final_df['E_shiftable'] = E_shiftable

    return final_df






def range_datetime(start , end , row , freq):
    """returns the list of the correct subdivision of the interval between start and end 
        into time slots every  freq minutes
    
    Args:
    start (str) : name of the column for the starting date of the session
    end (str) : name of the column for the ending date of the session 
    row () : a line of the dataset containing all the sessions
    freq (int) : 
    """
    
    range_datetime = list(pd.date_range(start=row[start],                    
                                        end=row[end], 
                                        freq=str(freq)+'T'))
    

    
    if range_datetime[-1] != row[end] :
        
        #case where the last element of range_datetime and the end of charging are in the same slot
        if range_datetime[-1].ceil(str(freq)+'min') == row[end].ceil(str(freq)+'min') :
            range_datetime[-1] = row[end]
            
            return range_datetime
        
        else :
            return range_datetime+[row[end]]
            
   
    return range_datetime




def create_Pflex_df(final_df, freq):
    """given a dataset with the flexible power for each time slots for each session, 
       returns a resampling of this dataset with in addition the features time, month and day
       
    Args:
    final_df (Pandas.Dataframe) : the dataset containing the flexible power for each session 
    freq (int) : 
    """
    
    P_flex_df  = final_df.resample(str(freq)+'T',on="date",convention="start").sum()
    P_flex_df.rename(columns={'Ps':'P_flex'}, inplace=True)
    
    
    P_flex_df = P_flex_df.reset_index()
    P_flex_df['time'] = P_flex_df['date'].apply(lambda x:x.time())
    P_flex_df['month'] = P_flex_df['date'].apply(lambda x:x.strftime("%B"))
    P_flex_df['day'] = P_flex_df['date'].apply(lambda x:x.strftime("%A"))
    
    return P_flex_df


#  Flexibility v1



def P_flex(df, start_session, end_charging, end_session, power, freq):
    """(implementation of the method flexible power for each time slots V1)
        returns a dataset with the following features for different moments
        -P_flex
        -Total_Power
    Args:
    df (Pandas.Dataframe) : output dataset of the function prepare_data
    start_session (str) : name of the column for the starting date of the session
    end_charging (str) : name of the column for the ending date of the charging part
    end_session (str) : name of the column for the ending date of the session
    power (str) ; name of the column related to power
    freq (int) : time interval IN minute for the flexibility
    """
    
    def get_minutes(datetime_var):
        """Get minutes from datetime (with seconds in decimal)"""
        minutes_datetime = datetime_var.minute+datetime_var.second/60
        return minutes_datetime

    final_df = pd.DataFrame()
    
    #
    for row in tqdm(df.to_dict('records')):
        
        #subdivision of the charging time interval in freq slots
        range_datetime_charging = range_datetime(start=start_session, 
                                                 end=end_charging, 
                                                 row=row, 
                                                 freq=freq)
        

        if (row['charging_time'] >= freq*60) & (row['idle_time'] >= freq*60) :
            
            Ps = row[power]
            
            row_start = range_datetime_charging[0]
            ratio_start = (freq-(get_minutes(row_start)%freq))/freq
            dic = {'date' : row_start,
                   'Ps': Ps*ratio_start,
                   'Total_Power':row[power]*ratio_start
                       
                       }
            final_df = final_df.append(dic, ignore_index=True)
            
            
            
            # for step in range_datetime_charging[1:-1] :
            #     dic = {'date' : step,
            #            'Ps': Ps,
            #            'Total_Power':row[power]
            #            }
            #     final_df = final_df.append(dic, ignore_index=True)


            final_df = final_df.append(list(({'date' : step,'Ps': Ps, 'Total_Power':row[power]} for step in range_datetime_charging[1:-1])),
                                       ignore_index=True)
                
            row_end = range_datetime_charging[-1]
            ratio_end = ((get_minutes(row_end)%freq)/freq)
            
            dic = {'date' : row_end,
                   'Ps': Ps*ratio_end,
                   'Total_Power':row[power]*ratio_end
                       }
            final_df = final_df.append(dic, ignore_index=True)
            
            
        elif (row['idle_time'] < freq*60) & (row['idle_time'] >= 30*60) :
            
            Ps = row[power]*(row['idle_time']/(freq*60))
            
            row_start = range_datetime_charging[0]
            ratio_start = (freq-(get_minutes(row_start)%freq))/freq
            dic = {'date' : row_start,
                   'Ps': Ps*ratio_start,
                   'Total_Power':row[power]*ratio_start
                       
                       }
            final_df = final_df.append(dic, ignore_index=True)
            
            
            
            # for step in range_datetime_charging[1:-1] :
            #     dic = {'date' : step,
            #            'Ps': Ps,
            #            'Total_Power':row[power]
            #            }
            #     final_df = final_df.append(dic, ignore_index=True)

            final_df = final_df.append(list(({'date' : step,'Ps': Ps, 'Total_Power':row[power]} for step in range_datetime_charging[1:-1])),
                                       ignore_index=True)

                
            row_end = range_datetime_charging[-1]
            ratio_end =  ((get_minutes(row_end)%freq)/freq)
            dic = {'date' : row_end,
                   'Ps': Ps*ratio_end,
                   'Total_Power':row[power]*ratio_end
                       }
            final_df = final_df.append(dic, ignore_index=True)
            
            
            
        else :
            
            Ps = 0
            
            row_start = range_datetime_charging[0]
            ratio_start = (freq-(get_minutes(row_start)%freq))/freq
            dic = {'date' : row_start,
                   'Ps': Ps,
                   'Total_Power':row[power]*ratio_start
                       }
            final_df = final_df.append(dic, ignore_index=True)
            
            
            
            # for step in range_datetime_charging[1:-1] :
            #     dic = {'date' : step,
            #            'Ps': Ps,
            #            'Total_Power':row[power]
            #            }
            #     final_df = final_df.append(dic, ignore_index=True)


            final_df = final_df.append(list(({'date' : step,'Ps': Ps, 'Total_Power':row[power]} for step in range_datetime_charging[1:-1])),
                                       ignore_index=True)
                
            row_end = range_datetime_charging[-1]
            ratio_end =  ((get_minutes(row_end)%freq)/freq)
            dic = {'date' : row_end,
                   'Ps': Ps,
                   'Total_Power':row[power]*ratio_end
                       }
            final_df = final_df.append(dic, ignore_index=True)
                
        
        range_datetime_idle = range_datetime(start=end_charging, 
                                             end=end_session, 
                                             row=row, 
                                             freq=freq)
        

        # for step in range_datetime_idle :
        #     dic = {'date' : step,
        #            'Ps': 0,
        #            'Total_Power':0
        #                }
            
        #     final_df = final_df.append(dic, ignore_index=True)

        final_df = final_df.append(list(({'date' : step,'Ps': 0, 'Total_Power':0} for step in range_datetime_idle)),
                                       ignore_index=True)
                

    

    
    return create_Pflex_df(final_df, freq)


#  Flexibility v2


def P_flex_2(df, start_session, end_charging, end_session, power, freq):
    """(implementation of the method flexible power for each time slots V2)
        returns a dataset with the following features for different moments
        -P_flex
        -Total_Power
    Args:
    df (Pandas.Dataframe) : output dataset of the function prepare_data
    start_session (str) : name of the column for the starting date of the session
    end_charging (str) : name of the column for the ending date of the charging part
    end_session (str) : name of the column for the ending date of the session
    power (str) ; name of the column related to power
    freq (int) : time interval IN minute for the flexibility
    """
    
    def get_minutes(datetime_var):
        """Get minutes from datetime (with seconds in decimal)"""
        minutes_datetime = datetime_var.minute+datetime_var.second/60
        return minutes_datetime

    final_df = pd.DataFrame()
    for row in tqdm(df.to_dict('records')):
        
        #subdivision of the charging time interval in freq slots
        range_datetime_charging = range_datetime(start=start_session, 
                                                 end=end_charging, 
                                                 row=row, 
                                                 freq=freq)
        

        
    
        
        row_start = range_datetime_charging[0]
        ratio_start = (freq-(get_minutes(row_start)%freq))/freq
        
        if (row['charging_time'] >= freq*60) & (row['idle_time'] >= freq*60) :
            
            Ps = row[power]
        
            dic = {'date' : row_start,
                   'Ps': Ps*ratio_start,
                   'Total_Power':row[power]*ratio_start

                       }
            final_df = final_df.append(dic, ignore_index=True)
            
        elif (row['idle_time'] < freq*60) & (row['idle_time']>=30*60):
            
            Ps = row[power]*(row['idle_time']/(freq*60))
        
            dic = {'date' : row_start,
                   'Ps': Ps*ratio_start,
                   'Total_Power':row[power]*ratio_start

                       }
            final_df = final_df.append(dic, ignore_index=True)
            
        
        else :
            
            Ps = 0
            dic = {'date' : row_start,
               'Ps': Ps,
               'Total_Power':row[power]*ratio_start

                   }
            final_df = final_df.append(dic, ignore_index=True)
            



        for step in range_datetime_charging[1:-1] :
            time_since_arrival = (step.floor(str(freq)+'min') - row[start_session]).total_seconds()
            
            if (row['charging_time'] >= time_since_arrival + freq*60) & (row['idle_time'] >= time_since_arrival + freq*60) :
                
                Ps = row[power]
            
                dic = {'date' : step,
                       'Ps': Ps,
                       'Total_Power':row[power]
                       }
                final_df = final_df.append(dic, ignore_index=True)
                
            elif (row['idle_time']-time_since_arrival < freq*60) & (row['idle_time']-time_since_arrival >= 30*60):
                
                Ps = row[power]*(row['idle_time']/(freq*60))
            
                dic = {'date' : step,
                       'Ps': Ps,
                       'Total_Power':row[power]
                       }
                final_df = final_df.append(dic, ignore_index=True)
                
                
            else :
                
                Ps = 0
                dic = {'date' : step,
                       'Ps': Ps,
                       'Total_Power':row[power]
                       }
                    
                final_df = final_df.append(dic, ignore_index=True)
                

        row_end = range_datetime_charging[-1]
        ratio_end =  ((get_minutes(row_end)%freq)/freq)
        
        time_since_arrival = (row_end.floor(str(freq)+'min') - row[start_session]).total_seconds()
        
        if (row['charging_time'] >= time_since_arrival + freq*60) & (row['idle_time'] >= time_since_arrival + freq*60) :
            
            Ps = row[power]*ratio_end
        
            dic = {'date' : row_end,
                   'Ps': Ps,
                   'Total_Power':row[power]*ratio_end
                       }
            final_df = final_df.append(dic, ignore_index=True)
            
        elif (row['idle_time']-time_since_arrival < freq*60) & (row['idle_time']-time_since_arrival >= 30*60):
            
                
            Ps = row[power]*(row['idle_time']/(freq*60))

            dic = {'date' : step,
                   'Ps': Ps*ratio_end,
                   'Total_Power':row[power]*ratio_end
                   }
            final_df = final_df.append(dic, ignore_index=True) 
            
        else :
            dic = {'date' : row_end,
                   'Ps': 0,
                   'Total_Power':row[power]*ratio_end
                       }
            final_df = final_df.append(dic, ignore_index=True)
            
           
            
        
        range_datetime_idle = range_datetime(start=end_charging, 
                                             end=end_session, 
                                             row=row, 
                                             freq=freq)[1:]
        
        for step in range_datetime_idle :
            dic = {'date' : step,
                   'Ps': 0,
                   'Total_Power':0
                       }
            final_df = final_df.append(dic, ignore_index=True)
                

    
    return create_Pflex_df(final_df, freq)
    #return final_df


# Faster way to compute flexibility
#WARNING : this new model does not take in consideration flexibility < delta



def rangeDateDf_bis(data, start, stop , evse, power, freq= 30, verbose=False):
    """Generate a daframe with a datetimeindex per 30 minutes over the entire month for rows and evse used in the months for columns

    Args:
    data (pandas.Dataframe) : dataset with all the available sessions
    start (str): name of the column for the beginning of the session
    stop (str): name of the column for the end
    power (str): name of the column for the power delivered during the session
    evse (str) : name of the column related to the evse used
    freq(int) : length of time slots (in min)
    verbose (bool) : True if the user wants to print  the duration of the creation of the dataframe
    """

    
    start_time = time.time()


    #create a list of all days of the months with a 30min frequency
    df = data.copy()
    months = df[start].dt.to_period('M').unique()
    
    date_range = []
    
    for Month in months :
        month = pd.to_datetime(str(Month))
        date_range = date_range + pd.date_range(month,
                               periods=month.days_in_month * 24 * (60/freq),
                               freq=str(freq)+"T").to_list()

    date_range = pd.DatetimeIndex(date_range)
    #print(date_range)
    
    df.loc[df[start].notnull(), "status_start"] = df[power]   
    df.loc[df[stop].notnull(), 'status_stop'] = 0
    df_start = df.filter([evse, start, "status_start"])
    df_stop = df.filter([evse, stop, "status_stop"])

    
    df_stop[stop] = df_stop[stop].dt.floor(str(freq)+"min")
    df_start[start] = df_start[start].dt.floor(str(freq)+"min")
    
    df_start = df_start.rename(columns={start: "date", "status_start": 'status'})
    df_stop = df_stop.rename(columns={stop: "date", "status_stop": 'status'})
    

    #concatenate df_start and df_stop to have an unique column status
    df_concat = pd.concat([df_start, df_stop])

    #in the case of a session where start = stop, remove the value of status_stop
    df_concat.drop_duplicates(subset=[evse,'date'],keep='last', inplace=True)

    
    if verbose :
        print("--- %s seconds ---  Création du DF" % (time.time() - start_time))
    


    #pivot of df_concat to have date as columns and evse as rows with the corresponding values
    pivoted = pd.pivot_table(df_concat, index=[evse], columns=['date'], values=["status"], aggfunc=np.sum)
    
    if verbose :
        print("--- %s seconds ---  Pivot du DF" % (time.time() - start_time))

    # create an empty dataframe with a DateTimeIndex per minute over the month   
    df_range = pd.DataFrame({'date': date_range}).set_index('date')
    #print(df_range)
    
    pivoted.columns = pivoted.columns.droplevel(0) #remove amount
    pivoted.columns.name = None               #remove categories
    pivoted.index.names = ["date"]
    
    #concatenation to be sure to have a datetimeindex per minute over the entire month for rows
    df_full = pd.concat([pivoted.T, df_range], axis=1)

    #using ffill method to fill NA in order to have the value power between start and end and 0 for sessions without indexes
    df_full = df_full.fillna(method='ffill')
    
    #truncate indexes from 1st 00:00 to 31st 23:59
    #mask = (df_full.index >= date_range[0]) & (df_full.index <= date_range[-1])
    
    df_full = df_full.fillna(0.0)
    
    if verbose:
        print("--- %s seconds --- Fin d'exécution" % (time.time() - start_time))
    
    return df_full


def Pflex_fast(data, start_session, end_charging, evse, power, freq):
    """Generate a daframe with a the flexible power in freq min slots for the first approach in a faster way

    Args:
    data (pandas.Dataframe) : dataset with all the available sessions
    start_session (str): name of the column for the beginning of the session
    end_charging (str): name of the column for the end of charging
    power (str): name of the column for the power delivered during the session
    evse (str) : name of the column related to the evse used
    freq(int) : length of time slots (in min)
    """

    df = data.copy()
    
    data_good = df[(df['idle_time']>=freq*60) & (df['charging_time']>=freq*60)]
    
    flex_df = rangeDateDf_bis(data=data_good, 
                        start=start_session,
                        stop=end_charging,
                        evse=evse,
                        power=power,
                        freq=freq)
    


    Pflex = flex_df.sum(axis=1)
    Pflex = Pflex.reset_index()
    Pflex.rename(columns={0:'Pflex','index':'date'}, inplace=True)
    return Pflex

def Pflex2_fast(data, start_session, end_charging, evse, power, freq):
    """Generate a daframe with a the flexible power in freq min slots for the second approach in a faster way

    Args:
    data (pandas.Dataframe) : dataset with all the available sessions
    start_session (str): name of the column for the beginning of the session
    end_charging (str): name of the column for the end of charging
    power (str): name of the column for the power delivered during the session
    evse (str) : name of the column related to the evse used
    freq (int): length of time slots (in min)
    """


    def categorise(row):  
        if row['idle_time'] > row['charging_time'] :
            return row[end_charging]
        else :
            return (row[start_session] + datetime.timedelta(seconds=row['idle_time'])).round('T')

    df = data.copy()
    
    df['done_Date'] = df.apply(lambda row: categorise(row), axis=1)

    
            
    flex_df = rangeDateDf_bis(data=df, 
                        start=start_session,
                        stop='done_Date',
                        evse=evse,
                        power=power,
                        freq=freq)
            
           
    


    Pflex = flex_df.sum(axis=1)
    Pflex = Pflex.reset_index()
    Pflex.rename(columns={0:'Pflex','index':'date'}, inplace=True)
    return Pflex



# Plots for flexibility v1 & v2



def distribution_on_typeofdays(P_flex_df, freq):
    """returns boxplot of P_flex  at each time slot depending on weekday / weekend
    
    Args:
    P_flex_df (pandas.Dataframe) : dataframe containing the flexible power 
    """
    fig = make_subplots(rows=2, cols=1, 
                         
                        subplot_titles=[f"Distribution of P_flex with delta = {str(freq)} min (weekday)",
                                        f"Distribution of P_flex with delta = {str(freq)} min (weekend)"])
                        
                                        

    
    weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','weekday']
    weekend = ['Saturday','Sunday','weekend']
    P_flex_df = P_flex_df.sort_values('time')
    
    max_P = 0
    
    for day, row, col in zip([weekday, weekend],[1,2],[1,1]) :
    
    
        
        data_ = P_flex_df[P_flex_df['day'].isin(day)]
        
        max_data_P = data_['P_flex'].max()
        
  
        max_P = max_data_P if max_P <= max_data_P else max_P
 
        
        fig.append_trace(go.Box(x=data_['time'], y=data_['P_flex']), row = row , col =col)
        
    
    
    fig.update_layout(height=1000, 
                      width=1200,
                     font=dict(size=20),
                     showlegend=False)
    
   
    fig.update_annotations(font_size=20)
    fig.update_yaxes(dict(range=[-50,max_P*1.1]), title_text = 'P_flex (kW)')
    
                     
    fig.show()



def distribution_by_days(P_flex_df):
    """returns boxplot of P_flex at each time slot for each day
    
    Args:
    P_flex_df (pandas.Dataframe) : dataframe containing the flexible power 
    """
    fig = make_subplots(rows=4, cols=2, 
                        subplot_titles=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                        )
    max_P = 0
    P_flex_df = P_flex_df.sort_values('time')
    for day,row,col in zip(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], [1,1,2,2,3,3,4] , [1,2,1,2,1,2,1]) :
        
        data_ = P_flex_df[P_flex_df['day']==day]
        
        
        max_data_P = data_['P_flex'].max()
        max_P = max_data_P if max_P <= max_data_P else max_P
        
    
        fig.append_trace(go.Box(x=data_['time'], y=data_['P_flex']), row = row , col =col)
                    
                    
    fig.update_layout(height=1000, 
                      width=1200,
                      showlegend=False)
    
    fig.update_yaxes(dict(range=[-50,max_P*1.1]), title_text = 'P_flex (kW)')
    fig.show()

