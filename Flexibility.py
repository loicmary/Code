
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from plotly.subplots import make_subplots


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


def plot_shiftable_energy_by_a_column(final_df, column_to_groupby):
    """plots the distribution (boxplot) of the shiftable energy through a feature
    Args:
    final_df (Pandas.Dataframe) : dataframe of all the available sessions containing the column E_shiftable
    """
    fig = go.Figure()
    final_df_agg = final_df.groupby(column_to_groupby).sum().reset_index()
    
    for element in final_df[column_to_groupby].unique():
        data_ = final_df_agg[(final_df_agg[column_to_groupby]==element)]
        #print(f"{element} : {data_.shape}")
        fig.add_trace(go.Bar(x = data_[column_to_groupby], 
                             y=data_['E_shiftable'], 
                             name=f"{element}",
                             text=np.round(data_['E_shiftable'],1),
                             textposition='auto'))

    fig.update_traces(textposition='outside')
    fig.update_layout(
        title=f"Total sessions' shiftable energy by {column_to_groupby}",
        yaxis_title="Total E_shiftable (kWh)",
        font=dict(
            size=18
        ),
        #yaxis_range=[0,100000]

    )

    fig.show()



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
                

    
    return final_df

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

