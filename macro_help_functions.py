# To work with data
import pandas as pd
import numpy as np
# preprocessing
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import hankel
# Visualisation
import matplotlib.pyplot as plt
# Quality metrics
from sklearn.metrics import r2_score, mean_absolute_percentage_error


def client_groups_nth_week(df_2models,df_real_id, bool_clients, nth_week=0, window_size=21):
    '''
    Calculating spent amounts for each client class on a given week.
    
    Params:
        df_2models -- dataset with real and predicted values
        df_real_id -- dataset with real users' ids
        bool_clients -- array with clients' binary predictability for every week 
        nth_week -- week number, on which to find client classes
        window_size -- window size for moving median
    Output:
        Ids of predictable clients, spent amounts for each client class
    '''
    good_clients_idx = np.where(bool_clients[:,nth_week]==1)
    # There're no client ids in the array; putting the dataset's indexes instead
    good_clients_id_upto8k = pd.Series(df_2models.index).iloc[good_clients_idx].values
    # Finding corresponding real clients' ids
    good_clients_id = pd.Series(df_real_id.index).iloc[good_clients_id_upto8k].values
    # Spent amounts of predictable clients
    gclients_amount = df_real_id.loc[good_clients_id]['money_trans']

    # Spent amounts of unpredictable clients
    # their real ids
    bad_clients_idx = np.where(bool_clients[:,nth_week]==0)
    bad_clients_id_upto8k = pd.Series(df_2models.index).iloc[bad_clients_idx].values
    bad_clients_id = pd.Series(df_real_id.index).iloc[bad_clients_id_upto8k].values
    # Spent amounts
    bclients_amount = df_real_id.loc[bad_clients_id]['money_trans']

    # Spent amounts of all clients
    aclients_amount = df_real_id.loc[pd.Series(df_real_id.index).iloc[df_2models.index].values]['money_trans']
        
    gclients = gclients_amount.squeeze().to_list()
    bclients = bclients_amount.squeeze().to_list()
    aclients = aclients_amount.squeeze().to_list()
    
    # Moving median to get rid of dips at the end of each quarter and extremely rich guys' influence
    good_rw = pd.Series(np.array(gclients).sum(0)[2,:]).rolling(window=window_size).median()
    bad_rw = pd.Series(np.array(bclients).sum(0)[2,:]).rolling(window=window_size).median()
    all_rw = pd.Series(np.array(aclients).sum(0)[2,:]).rolling(window=window_size).median()
    
    return good_clients_id, good_rw, bad_rw, all_rw


def prepare_data(clients_amount, split, dim=28, mem=5, scaler=0):
    '''
    Train/test split with preprocessing.
    
    Params:
        clients_amount -- pd.Series with spent amounts
        split -- number of days for test data
        dim -- predictions are made based on <dim> days
        mem -- memory depth for LSTM
        scaler -- sklearn scaler for normalization, or 0
    Output:
        Fitted scaler, features, targets, normalized data.
    '''
    clients_amount = pd.DataFrame(clients_amount)
    
    # Normalizing train and test set separately!
    # Using sklearn scaler, if it's given
    if not scaler:
        scaler = MinMaxScaler(clip=False, feature_range=(-1, 1))
        dat_train = scaler.fit_transform(clients_amount[:-split])
    else:
        dat_train = scaler.transform(clients_amount[:-split])      
    dat_test = scaler.transform(clients_amount[-split:])
    dat = np.concatenate((dat_train,dat_test))
    
    # Creating time series shifts
    ser=dat_train.reshape(-1)[:]
    x, y = MakeSet(ser, dim, mem)
    
    return scaler, x, y, dat


def MakeSet(ser, dim, mem):
    '''
    Creating time series shifts.
    
    Params:
        ser -- time series
        dim -- predictions are made based on <dim> days
        mem -- memory depth for LSTM
    Output:
        Features, targets.
    '''
    H=hankel(ser)
    X0=H[:-dim, :dim]
    X=[]
    for i in range(X0.shape[0]-mem-1):
        X.append(X0[i:i+mem, :])  
    X=np.array(X)
    y=H[mem+1:-dim, dim:dim+1]
    
    return X, y


def plot_clients_pred(clients_amount, full_fwd_pred_tr,full_fwd_true,y_true,y_pred_tr,
        dim, mem, split, fwd, days, tick_freq=13,shift_days=20):
    '''
    Plot a graph showing, how the model has been fitted to the train set and its forecasts on test
    
    Params:
        clients_amount -- spents amounts
        full_fwd_pred_tr -- forecasts on test data
        full_fwd_true -- real values on test data
        y_true -- real values on train data
        y_pred_tr -- forecasts on test data
        dim -- predictions are made based on <dim> days
        mem -- memory depth for LSTM
        split -- number of days for test data
        fwd -- how many days are predicted one by one
        days -- dates
        tick_freq -- show every <tick_freq> tick on x axis
        shift_days -- first valid index after moving median
    '''
    yt= clients_amount.iloc[mem+dim:-split].values.reshape(-1)
    ticks = days[shift_days+mem+dim:] 
    xt=np.arange(0, len(ticks), tick_freq)
    
    fig = plt.figure()
    # Plot results on train data
    plt.plot(yt, c='darkblue', alpha=.5, label='Initial series')
    plt.plot(y_pred_tr, lw=2, label=f'Model on history. R2={r2_score(y_true[1:], y_pred_tr):.2f}')
    # Plot results on test data
    t=np.arange(len(yt), len(yt)+fwd)
    plt.plot(t, full_fwd_true[-1], c='darkblue', alpha=.1, label=None)
    plt.plot(t, full_fwd_pred_tr[-1], lw=2,c='green')
    plt.plot(np.arange(len(yt), len(yt)+split-2), clients_amount.iloc[-split:-2].values.reshape(-1),c='darkblue', alpha=.5)
    # line between train and test data
    plt.axvline(len(yt), ls=':', c='k')
    plt.text((len(yt)), plt.gca().get_ylim()[1]*0.95, 
             f'{days.iloc[-split]}', rotation=0)
    plt.xticks(xt, ticks.iloc[xt], rotation=45, ha='right')

    plt.title('Predictions for clients')
    plt.xlabel('Date')
    plt.ylabel('Spent amount')

    plt.legend()
    plt.grid()
    
    return fig


def plot_clients_pred_base(chosen_client_group,a1_dm,df_2models,df_real_id,bool_clients,
                           full_fwd_pred_tr,full_fwd_true,split,
                            dim, mem, fwd, days, tick_freq, n_weeks):
    '''
    Plot a graph showing what forecasts does the model make.
    
    Params:
        chosen_client_group -- good/bad/all
        a1_dm -- test weeks
        df_2models -- dataset with real and predicted values
        df_real_id -- dataset with real users' ids
        bool_clients -- array with clients' binary predictability for every week 
        full_fwd_pred_tr -- forecasts on test data
        full_fwd_true -- real values on test data
        split -- number of days for test data
        dim -- predictions are made based on <dim> days
        mem -- memory depth for LSTM
        fwd -- how many days are predicted one by one
        days -- dates
        tick_freq -- show every <tick_freq> tick on x axis
        n_weeks -- number of test weeks
    '''
    fig = plt.figure()
        
    for nth_week in range(n_weeks-1):
        good_clients_id, next_good_rw, next_bad_rw, \
            next_all_rw = client_groups_nth_week(df_2models,df_real_id,bool_clients,nth_week)
        
        if chosen_client_group == 'good':
            client_group = next_good_rw
        elif chosen_client_group == 'bad':
            client_group = next_bad_rw
        elif chosen_client_group == 'all':
            client_group = next_all_rw
        
        last_day = days[days==f'{a1_dm[nth_week]}'].index[0]
        next_split= days.shape[0]-last_day-1
        yt= client_group[13+mem+dim:-next_split].values.reshape(-1)
        
        # Real values on train data
        plt.plot(client_group[13+mem+dim:-split].values.reshape(-1), c='darkblue', alpha=.02)
        
        ticks=days[13+mem+dim:]
        xt=np.arange(0, len(ticks), tick_freq)
        
        # Forecasts and real values on test data
        t=np.arange(len(yt), len(yt)+fwd)
        plt.plot(t, full_fwd_true[nth_week], c='darkblue', alpha=.5, label=None)
        plt.plot(t, full_fwd_pred_tr[nth_week], lw=2, c='green')

        # Line between train and test data
        #plt.axvline(len(yt), ls=':', c='k')
        plt.xticks(xt, ticks.iloc[xt], rotation=45, ha='right')
        
    plt.title('Predictions for clients')
    plt.xlabel('Date')
    plt.ylabel('Spent amount')
    plt.grid()
    
    return fig