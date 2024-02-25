

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Distribution Arrivals/Departures of EVs



def histogram_distribution_arrivals_departures_typeofday(data, start_session, end_session, category=''):
    """
    Plot the histogramm of the distribution of arrivals and departures of the cars with the distinction weekday/weekend

    Args:
    data (Pandas.Dataframe) : dataframe with all the exploitable sessions
    start_session (str) : name of the column for the beginning of the session
    end_session (str) : name of the column for the end of the session
    category (str) : if not '', it represents a certain category for the sessions (ex : a BusinessActivity)
    """

    fig = make_subplots(rows=2, cols=1, 
                         
                        subplot_titles=[f"Distribution Arrivals/Departures (weekday) [{category}]",
                                        f"Distribution Arrivals/Departures (weekend) [{category}]"])
    
    weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','weekday']
    weekend = ['Saturday','Sunday','weekend']
    
    

    for day, row, col  in zip([weekday, weekend], [1,2], [1,1]):
        df_d = data[data['weekday'].isin(day)]
        
        hours_arrival = df_d[start_session].apply(lambda x:x.hour)
        hours_departure = df_d[end_session].apply(lambda x:x.hour)

        fig.append_trace(go.Histogram(
        x=hours_departure,

        name='departure',
        xbins=dict(
        start=0,
        end=24,
        size=1
        ),
        marker_color='orange',
        ), row=row, col=col)
        
        
        fig.append_trace(go.Histogram(
        x=hours_arrival,
        name='arrivals',
        xbins=dict(
            start=0,
            end=24,
            size=1
        ),
        marker_color='blue'

    ), row=row, col=col)
        
        
    fig.update_layout(
    xaxis_title="Hours",
    font=dict(
    size=20),
    height=700)
    
    fig.update_xaxes(dict(
        tickmode = 'array',
        tickvals = [i for i in range(24)]))
    
    #fig.update_yaxes(dict(range=[0,1700]))
    fig.show()


# Distributions sojourn, charging & idle times




def distribution_durations(data, category = None):
    """
    return the boxplots for sojourn, charging & idle times

    Args:
    data (Pandas.Dataframe) : dataset with all the exploitable sessions
    category (str) : name of a column of data (ex: BusinessActivity)
    """
    fig = go.Figure()

    weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','weekday']
    weekend = ['Saturday','Sunday','weekend']

    if category is None :
    
        for col in ['sojourn_time', 'charging_time', 'idle_time']:
            
            for day in [weekday, weekend]:

                #durations in hour
                data_ = data[(data['weekday'].isin(day))]
                                     
                fig.add_trace(go.Box(y=data_[col]/3600, name=col+'('+day[-1]+')'))

        fig.update_layout(
            title="Boxplot of sojourn, charging & idle times  | weekday vs weekend",
            yaxis_title="Hours",
            showlegend=False
        )

        fig.show()

    else :

        for col in ['sojourn_time', 'charging_time', 'idle_time']:

            for cat in data[category].unique():
            
                for day in [weekday, weekend]:

                    #durations in hour

                    data_ = data[(data['weekday'].isin(day)) & (data[category]==cat)]

                    fig.add_trace(go.Box(y= data_[col]/3600, 
                                         name=f"{col} [{cat}] ({day[-1]})"))

        fig.update_layout(
            title="Boxplot of sojourn, charging & idle times  | weekday vs weekend",
            yaxis_title="Hours",
            showlegend=False
        )
        #fig.update_traces(boxpoints=False)
        fig.show()



def distribution_idle_time(data, category=None):
    """returns the boxplots of the distribution of idle time with the distinction weekday/weekend

    Args :
    data_ (Pandas.Dataframe) : dataset with all the exploitable sessions
    """

    if category is None :
        fig = go.Figure()

        i = 0
        for col in  ['idle_time']:

            data_ =  data
                                    

            fig.add_trace(go.Box(y=data_[col]/3600, name=f"{col} (all days)"))
            for x in (data_[col]/3600).quantile([0.5,0.75]).values:
                    fig.add_annotation(
                    x=i+0.4,
                    y=np.round(x,2),
                    text=str(np.round(x,2)),
                    showarrow=False
                    )

            i+=1
            weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','weekday']
            weekend = ['Saturday','Sunday','weekend']

            for day in [weekday, weekend]:

                data_ = data[(data['weekday'].isin(day))]

                fig.add_trace(go.Box(y=data_[col]/3600, name=f"{col} ({day[-1]})"))

                for x in (data_[col]/3600).quantile([0.5,0.75]).values:
                    fig.add_annotation(
                    x=i+0.4,
                    y=np.round(x,2),
                    text=str(np.round(x,2)),
                    showarrow=False
                    )
                i+=1

        fig.update_layout(
            title=f"Boxplot idle times | weekday vs weekend",
            yaxis_title="Hours",
            font=dict(
            size=18
        ),
            showlegend=False

        )

        fig.show()
    
    else :

        fig = go.Figure()

        i = 0
        for cat in data[category].unique():
            for col in  ['idle_time']:

                data_ =  data[data[category]==cat]
                                        

                fig.add_trace(go.Box(y=data_[col]/3600, name=f"{cat} (all days)"))
                for x in (data_[col]/3600).quantile([0.5,0.75]).values:
                        fig.add_annotation(
                        x=i+0.4,
                        y=np.round(x,2),
                        text=str(np.round(x,2)),
                        showarrow=False
                        )

                i+=1
                weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','weekday']
                weekend = ['Saturday','Sunday','weekend']

                for day in [weekday, weekend]:

                    data_ = data[(data['weekday'].isin(day)) & (data[category]==cat)]

                    fig.add_trace(go.Box(y=data_[col]/3600, name=f"{cat} ({day[-1]})"))

                    for x in (data_[col]/3600).quantile([0.5,0.75]).values:
                        fig.add_annotation(
                        x=i+0.4,
                        y=np.round(x,2),
                        text=str(np.round(x,2)),
                        showarrow=False
                        )
                    i+=1

        fig.update_layout(
            title=f"Boxplot idle times | weekday vs weekend",
            yaxis_title="Hours",
            font=dict(
            size=18
        ),
            showlegend=False

        )

        fig.show()



def distribution_idle_time_by_arrival_hour(df, start_session):
    """returns for each arrival hour, the distribution of the idle time with the distinction weekday/weekend

    Args:
    df (Pandas.Dataframe) : dataset with the exploitable sessions
    start_session (str) : name of the column for the beginning of the session
    """


    df['hour_arrival'] = df[start_session].dt.hour
    fig = make_subplots(rows=2, cols=1, 
                         
                        subplot_titles=[f"Distribution of idle time by arrivals hour (weekday)",
                                        f"Distribution of idle time by arrivals hour (weekend)"])
                        
                                        

    
    weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','weekday']
    weekend = ['Saturday','Sunday','weekend']
    
    for day, row, col in zip([weekday, weekend],[1,2],[1,1]) :
    
                
        
        data_ = df[df['weekday'].isin(day)]
        
        fig.append_trace(go.Box(x=data_['hour_arrival'], y=data_['idle_time']/3600), row = row , col =col)
    
    
    fig.update_layout(height=1000, 
                      width=1200,
                     font=dict(size=20), showlegend=False)
    
    fig.update_annotations(font_size=20)
    fig.update_xaxes(dict(
        tickmode = 'array',
        tickvals = [i for i in range(24)]))
    fig.update_yaxes(dict(range=[-5,50]))
                     
    fig.show()


def plot_num_feature_by_category(data, column_for_groupBy, column_to_study, operation, title, yaxis_title):
    data_agg = data.groupby([column_for_groupBy])[[column_to_study]].agg(operation).reset_index()

    max_value = data_agg[column_to_study].max()
    fig = go.Figure()
    fig.add_trace(go.Bar(x = data_agg[column_for_groupBy], 
                     y=data_agg[column_to_study], 
                     text= np.round(data_agg[column_to_study],2),
                     textposition='auto'),
             )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        font=dict(
            size=18
        ),
        yaxis_range=[0,max_value*1.1]
    )

    fig.show()

# Plots on map

def plots_on_map(data, column_for_groupBy, column_to_study, operation, geojson_file, featureidkey, title):
    """the function makes first a count-aggregation of column_to_study by column_for_groupBy 
    Then, we use this aggregation to plot a map colored by column_to_study

    Args:
    data (pandas.Dataframe) : dataset with the exploitable sessions
    column_for_groupBy (str) : name of the column for the groupBy for the count-aggregation (can be a zipcode, region, department ...)
    column_to_study (str) : name of the column we want to stydy (can be evse, sessions ...)
    geojson_file (.geojson) : file containing the polygon to plot the map
    featureidkey (str) : feature id key in the geojson file, must represent the same thing than column_to_study (can be the id key for a city, region, department ...)
    title (str) : title of the plot
    
    Exemple of the use of this function : plot the number of sessions by department in France
    """
    data_agg = data.groupby([column_for_groupBy])[[column_to_study]].agg(operation).reset_index()
    
    fig = px.choropleth(
    data_agg ,
    locations=column_for_groupBy,
    geojson=geojson_file,
    color=column_to_study,
    featureidkey=featureidkey,
    title=title,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.add_scattergeo(
      geojson=geojson_file,
      locations = data_agg[column_for_groupBy],
      text = data_agg[column_for_groupBy],
      featureidkey=featureidkey,
      mode = 'text',
      textfont=dict(
            color="white"
        ))
    fig.show()


