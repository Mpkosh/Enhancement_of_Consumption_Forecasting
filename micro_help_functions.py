import pandas as pd
import numpy as np


def get_df(df, client_id, k=1, n_cats=3, n_train=28, n_pred=7):
    '''
    Time series' shifts for a given client
    
    Params:
        df -- dataset with users and transactions
        client_id
        k -- take each <k> shift (k==1: all data is taken)
        n_cats -- number of basic values
        n_train -- predicting based on <n_train> days
        n_pred -- predicting <n_pred> days at a time
    Output:
        Dataset with shifts.
    '''
    a = pd.DataFrame(df.loc[client_id], 
                 columns=[*[f'event_{i}' for i in range(n_cats)]
                          ,'d_sin','d_cos','m_sin','m_cos'])
    # changing type 
    cols = ['d_sin','d_cos','m_sin','m_cos']
    a[cols] = a[cols].apply(lambda x: x.astype('float16'))
    cols = [f'event_{i}' for i in range(n_cats)]
    a[cols] = a[cols].apply(lambda x: x.astype('int8'))

    orig_cols = a.columns
    
    # taking shifts
    to_add = []
    for i in range(1,n_train):
          for col in orig_cols[:]:
                to_add.append(a[col].shift(-i).iloc[::k].rename(f'{col}-{i}'))
    # shifts of a target (predicted <n_pred> days)
    for i in range(n_train, n_train+n_pred):
          for col in orig_cols[:n_cats]: # whether any transactions were made
                to_add.append(a[col].shift(-i).iloc[::k].rename(f'{col}+{i-n_train+1}'))
    
    a = pd.concat([a.iloc[::k]]+to_add, axis=1) 
    a.dropna(inplace=True)
    
    return a


def get_splits_by_client(client_id, train_df, test_df, n_cats=3, n_train=28, n_pred=7, n_features=7):
    '''
    Train/test split of a given clients.
    
    Params:
        client_id
        train_df
        test_df
        n_cats -- number of basic values
        n_train -- predictions are made based on <n_train> days
        n_pred -- <n_pred> days are predicted at a time
        n_features -- number of transaction features
    Output:
        Train/test split.
    '''
    train_client = get_df(train_df, client_id, k=1, n_train=n_train, n_pred=n_pred)
    test_client = get_df(test_df, client_id, k=1, n_train=n_train, n_pred=n_pred)
    
    all_x = train_client.iloc[:,:-n_pred*n_cats].values.reshape(-1,n_train,n_features)
    all_y = train_client.iloc[:,-n_pred*n_cats:].values.reshape(-1,n_pred,n_cats)

    all_test_x = test_client.iloc[:,:-n_pred*n_cats].values.reshape(-1,n_train,n_features)
    all_test_y = test_client.iloc[:,-n_pred*n_cats:].values.reshape(-1,n_pred,n_cats)
    
    return all_x,all_y,all_test_x,all_test_y


def get_df_full_y_new(df, min_date='2020-07-01',max_date='2021-06-29', n_cats=3,
                     n_train=28,n_pred=7):
    '''
    Time series' shifts for a given client with added date to the target.
    Params:
        df -- dataset with users and transactions
        min_date -- first day (YYYY-MM-DD)
        max_date -- last day (YYYY-MM-DD)
        n_cats -- number of basic values
        n_train -- predictions are made based on <n_train> days
        n_pred -- <n_pred> are predicted days at a time
    Output:
        Dataset with shifts.
    '''
    a = pd.DataFrame(df, columns=[*[f'event_{i}' for i in range(n_cats)]
                          ,'d_sin','d_cos','m_sin','m_cos'])

    cols = ['d_sin','d_cos','m_sin','m_cos']
    a[cols] = a[cols].apply(lambda x: x.astype('float16'))
    cols = [f'event_{i}' for i in range(n_cats)]
    a[cols] = a[cols].apply(lambda x: x.astype('int32'))
    a['date'] = pd.date_range(min_date,max_date)

    orig_cols = a.columns
    # taking shifts
    to_add = []
    for i in range(1,n_train):
          for col in orig_cols[:]:
                to_add.append(a[col].shift(-i).rename(f'{col}-{i}'))
    # shifts of a target (predicted <n_pred> days)
    for i in range(n_train, n_train+n_pred):
          for col in orig_cols[:]:
                to_add.append(a[col].shift(-i).rename(f'{col}+{i-n_train+1}'))
       
    a = pd.concat([a]+to_add, axis=1) 
    a.dropna(inplace=True)
    
    return a


def F1metr(x_real, x_pred):
    '''
    Calculating F-score by hand to ensure that F1-score([0,0,0],[0,0,0]) is 1.
    '''
    x_pred, x_real= x_pred.astype(int), x_real.astype(int) 
    
    tp=len(np.where(x_pred[np.where(x_real==1)]==1)[0])
    tn=len(np.where(x_pred[np.where(x_real==0)]==0)[0])
    fp=len(np.where(x_pred[np.where(x_real==0)]==1)[0])
    fn=len(np.where(x_pred[np.where(x_real==1)]==0)[0])
    
    if (tp+fp)*(tp+fn)*tp:
        precision, recall = tp/(tp+fp), tp/(tp+fn)
        f1=2*precision*recall/(precision+recall) 
    else:
        f1=0.
        
    if (tp+tn+fp+fn):
        accuracy=(tp+tn)/(tp+tn+fp+fn)*100
    else:
        accuracy=0.
        
    if accuracy>99.: f1=1
    
    return f1


def apply_metric(metric, array):
    '''
    Measure the given metric for real and predicted values for every week for every basic value.
    '''
    return np.array([[[metric(ys[0],(ys[1]>0).astype('int')) for ys in weeks] for weeks in cats] for cats in array])
    
    
def get_base_inc_arrays(df_2models, metric, col_base='base', col_inc='inc',
                       n_cats=3, n_test_weeks=25, n_pred=7):
    '''
    Measure the given quality metric for real and predicted values.
    
    Params:
        df_2models -- dataset with real and predicted values
        metric -- quality metric
        col_base -- column name with a base model's outputs
        col_inc -- column name with an incremental model's outputs
        n_cats -- number of basic values
        n_test_weeks -- number of week in test
        n_pred -- <n_pred> days are predicted at a time
    Output:
        Numpy arrays with calculated quality metric and the euclidean norm of the three-
component vector divided by a square root of three to keep the value in [0, 1].
    '''
    # Arrays for the base and the incremental models
    base_array = np.array([np.vstack(x) for x in df_2models[col_base].to_numpy().reshape(-1)],dtype=object
                         ).reshape(-1,n_cats,n_test_weeks,2,n_pred) # 2 -- real values and predictions

    inc_array = np.array([np.vstack(x) for x in df_2models[col_inc].to_numpy().reshape(-1)],dtype=object
                        ).reshape(-1,n_cats,n_test_weeks,2)
    inc_array = np.array([i.flatten() for i in inc_array.reshape(-1)],dtype=object
                        ).reshape(-1,n_cats,n_test_weeks,2,n_pred)
    
    # Measuring the given quality metric
    inc_with_metric = apply_metric(metric, inc_array)
    base_with_metric = apply_metric(metric, base_array)
    
    # the euclidean norm
    inc_with_metric_3in1 = np.array([np.sqrt(np.sum(user**2,0))/np.sqrt(3) for user in inc_with_metric])
    base_with_metric_3in1 = np.array([np.sqrt(np.sum(user**2,0))/np.sqrt(3) for user in base_with_metric])
    
    return inc_with_metric, inc_with_metric_3in1, base_with_metric, base_with_metric_3in1