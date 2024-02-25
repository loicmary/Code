import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_pflex(Pflex, title=''):
    """Returns a plot with the time series and a red line which represents a limit date that separates the dataset as follow :
       data <= limit date --> 80% of the global time series

    Args:
    Pflex (pandas.Dataframe) : dataframe with the time series to plot
    title (str) :  title of the plot
    """
    
    fig = px.line(Pflex, x="date", y="Pflex", title=title)

    n_days = (Pflex['date'].iloc[-1].date() - Pflex['date'].iloc[0].date()).days
    limit_date = pd.Timestamp(Pflex['date'].iloc[0].date() + datetime.timedelta(days=int(np.round(n_days*0.8))))
    fig.add_vline(x=limit_date, line_width=3, line_dash="dash", line_color="red")

    fig.show()

def PflexforHolidays(row, holidays):
    """Preprocessing for holidays :
     - if J-7 is holiday then Pflex_J7 = Pflex_J14
     - if J = holiday then Pflex_J7 = Pflex of the closest sunday of last week
    This function is used in the function create_features
    Args:
    row : row of the dataframe created by the function create_features
    holidays : list of the holidays of a country
    """
    if row['is_holiday']:
        closed_sunday = 7 +  ((row['date'].weekday() + 1) % 7)
        return row['Pflex_J'+str(closed_sunday)]
        
    else :
        if (row['date'] - datetime.timedelta(days=7)) in holidays:
            return row['Pflex_J14']
        else :
            return row['Pflex_J7']


def create_features(Pflex, last_day, holidays):
    """Based on a dataframe containing information about the flexibility per time-slots, 
       this function creates a new dataframe with different features including the flexible power history of a week

    Args:
    last_day (int) : the maximum number of day you want the historical flexibility | EX: if last_day = 14, you can the historic until Pflex_J14 
    Pflex (pandas.Dataframe) : dataframe containing flexible power per time-slots
    holidays : list of the holidays of a country
    """

    
    pflex = Pflex.copy()
    
    pflex = pflex.set_index('date')
    pflex = pd.merge(pflex.reset_index(), pflex["Pflex"].shift(1, freq="D").reset_index(), on='date',how='left')
    
    for i in range(2,last_day+1):
        pflex = pflex.set_index('date')
        pflex = pd.merge(pflex.reset_index(), pflex["Pflex_"+'x_'*(i-2)+'x'].shift(i, freq="D").reset_index(), on='date',how='left')
        
    
    #Pflex_Jn represents the Pflex n days before
    pflex.rename(columns = {'Pflex'+'_x'*(last_day):'Pflex'}, inplace = True)
    for i in range(1, last_day+1):
        pflex.rename(columns={"Pflex_"+"x_"*(i-1)+"y":"Pflex_J"+str(i)}, inplace=True)
        
    pflex.dropna(inplace=True)
    

    pflex['month'] = pflex['date'].dt.month
    pflex['weekday'] = pflex['date'].dt.weekday
    pflex['hour'] = pflex['date'].dt.hour
    pflex['minute'] = pflex['date'].dt.minute
    pflex['timestep'] = pflex['date'].dt.time.apply(lambda x : int(((x.hour*60) + x.minute)/30 +1))
    pflex['is_holiday'] = pflex['date'].dt.date.apply(lambda x:x in holidays)
    
    pflex['Pflex_J7'] = pflex.apply(lambda row: PflexforHolidays(row, holidays), axis=1)

    def mean_PflexJ7(row):
        return np.mean([row['Pflex_J1'], row['Pflex_J2'],row['Pflex_J3'], row['Pflex_J4'], row['Pflex_J5'], row['Pflex_J6'], row['Pflex_J7']])
    
    pflex['mean_Pflex7'] = pflex.apply(lambda row: mean_PflexJ7(row), axis=1)

    
    return pflex


def create_train_test_set(pflex, 
                        features_to_keep):
    """Based on the dataframe created by the function create_feature, the function returns x_train, y_train, x_test, y_test, limit_date
       where limit_date is the date which separates the dataset in 2 with 80% for the train and 20% for the test

    Args:
    pflex (pandas.Dataframe) : dataframe with flexible power and different features
    features_to_keep (list of str) : name of the columns to keep to be the co-variables of the model
    """
    

    #compute the number between the end and the start of Date in pflex and we consider 80% of this number 
    # to consider the limit_date to separate the train and test set
    #Thus, the train set represents 80% of the global dataset
    n_days = (pflex['date'].iloc[-1].date() - pflex['date'].iloc[0].date()).days
    limit_date = pd.Timestamp(pflex['date'].iloc[0].date() + datetime.timedelta(days=int(np.round(n_days*0.8))))
    
    
    train = pflex[pflex['date']<limit_date]
    test = pflex[pflex['date']>=limit_date]
    
    

 
    x_train = train[features_to_keep].set_index(train['date'])
    y_train = train['Pflex']

    x_test = test[features_to_keep].set_index(test['date'])
    y_test = test['Pflex']
    
    return x_train, y_train, x_test, y_test, limit_date


def predictions(x_train, y_train, x_test, model):
    """fit a model and gives predictions

    Args:
    x_train : training features
    y_train : training target variable
    x_test : testing features
    model : model used for predictions
    """
    model.fit(x_train, y_train)
    return model.predict(x_test)

def forecast_perPeriod(pflex, limit_date, features_to_keep, model):
    """Returns predictions through the method of  the sliding training and test set

    Args:
    pflex (pandas.Dataframe) : dataframe with the flexible power by half hour slot and others features (result of the function create_features)
    limite_date (date) : the limit date in order to have 80% of the dataset for the train and 20% for the test
    features_to_keep (list of str) : list of features to keep in the model
    model : model used to make predictions
    """

    def train_test(pflex, limit_dates_train, limit_dates_test, features_to_keep):
        """Returns the training and test set time-bounded

    Args:
    pflex (pandas.Dataframe) : dataframe with the flexible power by half hour slot and others features (result of the function create_features)
    limit_dates_train (list of 2 elements) : the time interval to delimit the  training set
    limit_dates_test (list of 2 elements) : the time interval to delimit the  test set
    features_to_keep (list of str) : list of features to keep in the model
    """
        train = pflex[(pflex['date']>= limit_dates_train[0]) & ((pflex['date']<= limit_dates_train[1]))]
        test = pflex[(pflex['date']>= limit_dates_test[0]) & ((pflex['date']<= limit_dates_test[1]))]

        
        x_train = train[features_to_keep].set_index(train['date'])
        y_train = train['Pflex']
        
        x_test = test[features_to_keep].set_index(test['date'])
    
        
        return x_train, y_train, x_test


    #We define the first training and test set of the sliding method
    #The first training set starts one month before the limit date and ends the day before the limit date
    #The first test set starts at the limit date and end 1 week after the limit date
    initial_limit_dates_train = [limit_date-datetime.timedelta(weeks=4), limit_date-datetime.timedelta(days=1, minutes=30)]
    initial_limit_dates_test = [limit_date, limit_date+ datetime.timedelta(weeks=1, hours=23, minutes=30)]
    
    y_pred = []
    x_train, y_train, x_test = train_test(pflex, initial_limit_dates_train, initial_limit_dates_test, features_to_keep)

    model.fit(x_train, y_train)
    y_pred = y_pred + list(model.predict(x_test))
    
    while 1 :

        # One week shift of the train and test set
        limit_dates_train = [initial_limit_dates_train[i] + datetime.timedelta(weeks=1) for i in [0,1]]
        limit_dates_test = [initial_limit_dates_test[i] + datetime.timedelta(days=8) for i in [0,1]]
        
        initial_limit_dates_train = limit_dates_train
        initial_limit_dates_test = limit_dates_test
        
        x_train, y_train, x_test = train_test(pflex, limit_dates_train, limit_dates_test, features_to_keep)

        
        if x_test.shape[0] == 0 :
            # When we exceeds the dataset deadline, get out of the while because there is no data
            break

        model.fit(x_train, y_train)
        y_pred = y_pred + list(model.predict(x_test))
    

        
    return y_pred


    
def plot_predictionsVSvalues(y_pred, y_test, index):
    """This functions plots y_pred & y_test 
    Args:
    y_pred (numpy.Arrays) : array of the prediction
    y_test (pandas.Series) : Series of the true values 
    index (pandas.index) : index date of x_test (index = x_test.index)
    """
    
    Y_test = y_test.reset_index()
    Y_pred = pd.DataFrame(y_pred, columns=['Pflex'])
    
    
    mae = np.round(mean_absolute_error(y_true=Y_test['Pflex'], y_pred=Y_pred['Pflex']),1) 
    rmse = np.round(mean_squared_error(y_true=Y_test['Pflex'], y_pred=Y_pred['Pflex'], squared=False ),1)
    percentage_overestimation = np.round(((y_pred>y_test).sum()/y_test.shape[0])*100,1)
    
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=index, y=Y_test['Pflex'], mode='lines', name='true values'))
    fig.add_trace(go.Scatter(x=index, y=Y_pred['Pflex'], mode='lines', name='predicted values'))

    fig.update_layout(
    title=f"MAE : {mae} | RMSE : {rmse} | % of overestimation : {percentage_overestimation}",
    yaxis_title="Pflex (kW)"
)
    fig.show()

    fig = px.line(x=index, y=Y_test['Pflex']- Y_pred['Pflex'] , title="y_true - y_pred")
    fig.show()


    
    plt.figure(figsize=(30,20))
    plt.scatter(Y_test['Pflex'], Y_pred['Pflex'])
    max_Pflex = max(Y_test['Pflex'].max(), Y_pred['Pflex'].max())
    plt.plot([0,max_Pflex*1.1], [0,max_Pflex*1.1], color='red')
    plt.xlabel('true values', fontsize=18)
    plt.ylabel('predicted values', fontsize=18)
    plt.title('Predictions VS True Values', fontsize=18)
    plt.show()

def error_by_month(y_pred, x_test, y_test):
    """Returns the error od prediction by month*
    Args:
    y_pred : predictions
    x_test : test co-variables
    y_test : true values for the target variable
    """
    
    
    performances = pd.concat([x_test.reset_index(), pd.Series(y_pred, name='y_pred').to_frame(), pd.Series(y_test, name='y_test').to_frame().reset_index(drop=True)], axis=1)
    
    if 'month' not in performances.columns:
        performances['month'] = performances['date'].dt.month
        
    performances = performances[['month', 'y_pred', 'y_test']]
    performances['error'] = performances['y_test'] - performances['y_pred']

    
    performances_by_month = performances.groupby('month').mean()['error']
    
    return performances_by_month.to_frame()


def performances_2_approaches(list_models, list_model_names, pflex, limit_date, features_to_keep,  x_train_init, y_train_init, x_test_init, y_test_init) :
    """Returs 2 dataframe (for each approach) with some metrics (RMSE, sum of negative and positive errors and the percentage of overestimation) given by different models for prediction
    Args:
    list_models : a list of models
    list_model_names (list of str) : list of the name of the model used
    pflex (pandas.Dataframe) : dataframe with the flexible power by half hour slot and others features (result of the function create_features)
    limit_date (date) : the limit date in order to have 80% of the dataset for the train and 20% for the test (result of the function create_train_test_set)
    features_to_keep (list of str) : list of features to keep in the model
    x_train_init : co-variables of the training set given by the function create_train_test_set
    y_train_init : target variable of the training set given by the function create_train_test_set
    x_test_init : co-variables of the test set given by the function create_train_test_set
    y_test_init : target variable of the test set given by the function create_train_test_set
     """
    
    #Results for the first Approach (80% train et 20% test)
    Approach1 = {"RMSE": [],
                 "Sum of positive errors":[],
                 "Sum of negative errors":[],
                 "% of overestimation": []}
    
    #Results for the second Approach (sliding training & test sets)
    Approach2 = {"RMSE": [],
                 "Sum of positive errors":[],
                 "Sum of negative errors":[],
                 "% of overestimation": []}
    
    
    
    for model in list_models :
 
        model.fit(x_train_init, y_train_init)
        y_pred = model.predict(x_test_init)
        
        y_pred2 = forecast_perPeriod(pflex, 
                              limit_date, 
                              features_to_keep, 
                              model=model)
        rmse1 = np.round(mean_squared_error(y_true=y_test_init, y_pred=y_pred, squared=False ),1)
        
        ERROR = y_test_init-y_pred
        error_positive = ERROR[ERROR>= 0]
        error_negative = ERROR[ERROR<0]

        #percentage of predictions > true values
        percentage_overestimation1 = np.round(((y_pred>y_test_init).sum()/y_test_init.shape[0])*100,1)
        
        Approach1['RMSE'].append(rmse1)
        Approach1['Sum of positive errors'].append(np.round(error_positive.sum(),1))
        Approach1['Sum of negative errors'].append(np.round(error_negative.sum(),1))
        Approach1['% of overestimation'].append(percentage_overestimation1)
        
        
        
        rmse2 = np.round(mean_squared_error(y_true=y_test_init, y_pred=y_pred2, squared=False ),1)
        
        ERROR2 = y_test_init-y_pred2
        error2_positive = ERROR2[ERROR2>= 0]
        error2_negative = ERROR2[ERROR2<0]
        percentage_overestimation2 = np.round(((y_pred2>y_test_init).sum()/y_test_init.shape[0])*100,1)
        
        Approach2['RMSE'].append(rmse2)
        Approach2['Sum of positive errors'].append(np.round(error2_positive.sum(),1))
        Approach2['Sum of negative errors'].append(np.round(error2_negative.sum(),1))
        Approach2['% of overestimation'].append(percentage_overestimation2)
        

    return pd.DataFrame(Approach1, columns=['RMSE', 'Sum of positive errors', 'Sum of negative errors', '% of overestimation'], index=list_model_names), pd.DataFrame(Approach2, 
                 columns=['RMSE', 'Sum of positive errors', 'Sum of negative errors', '% of overestimation'],
                 index=list_model_names)