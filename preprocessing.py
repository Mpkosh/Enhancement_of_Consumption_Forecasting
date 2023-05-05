import pandas as pd
import numpy as np


def cats_to_14(df, col='mcc', new_col='group_14'):
    '''
    Classifying MCCs into 14 groups of users' interests
    
    Params:
        df -- dataset with MCC
        col -- column name with MCC
        new_col -- output column name
    Output:
        Dataset <df> with groups of users' interests in <new_col>.
    '''
    
    # Defining which MCCs belong to which group of users' interests
    cat_food = [5411,5814,5499,5812,5462,5441,5422,5451,5309]
    cat_outfit = [5691,5651,5661,5621,5699,5949]
    cat_dwelling = [5211,4900,5722,5712,5261,5719,5251,5714,5039]
    cat_health = [5912,8099,8011,8021,8043,8062,8071]
    cat_beauty = [5977,7230,7298,5631]
    cat_money = [6011,6536,6012,6538,6010,9311,9222,6051,6300,6540]
    cat_travel = [4111,5541,4121,4131,7512,4784,4112,5533,7011,7523,
                  7542,5511,4511,5542,4789,7538,3011,5521] # added 5521
    cat_kids = [5641,5945,5943,8299,8220]
    cat_nonfood = [5331,5999,5311,5200,5399,5931,5948]
    cat_remote = [5968,5964]
    cat_telecom = [4814,4816,5732,4812,9402,4215,4899]
    cat_fun = [5921,5813,5993,5995,5192,5816,5992,5735,5942,7832,5941,
               7995,5947,7922,5944,7997,7999,5193,7941,742,7991,7994,
               7221,7841,5815,4722]
    cat_charity = [8398]
    cat_misc = [8999,7299,7311,7399,9399,7216,7699] #added 7216,7699
    
    # Creating a dictionary "MCC":"group"
    cats_14_dict = dict(zip(cat_food+cat_outfit+cat_dwelling+cat_health+cat_beauty+cat_money+cat_travel+
                            cat_kids+cat_nonfood+cat_remote+cat_telecom+cat_fun+cat_charity+cat_misc, 
                                    ['food' for i in cat_food]+['outfit' for i in cat_outfit]+
                                    ['dwelling' for i in cat_dwelling]+['health' for i in cat_health]+
                                    ['beauty' for i in cat_beauty]+['money' for i in cat_money]+
                                    ['travel' for i in cat_travel]+['kids' for i in cat_kids]+
                                    ['nonfood' for i in cat_nonfood]+['remote' for i in cat_remote]+
                                    ['telecom' for i in cat_telecom]+['fun' for i in cat_fun]+
                                    ['charity' for i in cat_charity]+['misc' for i in cat_misc]))
    
    # In case MCC doesn't belong anywhere, put it to "mics." group
    df[new_col] = df[col].apply(lambda x: cats_14_dict.get(x, 'misc'))
    
    return df


def create_trans_data(df, col_clientid='REGNUM', col_date='date', 
                        col_group='group_3', col_amount='PRC_AMT',
                        add_count=False):
    '''
    Creating dataframe with information regarding spent amount and whether there were any transactions made 
    on every day for every client for 3 basic values.
    
    Params:
        df -- dataframe with users and basic values
        col_clientid -- column name with users' ids
        col_date -- column name with transaction dates
        col_group -- column name with basic values
        col_amount -- column name with spent amounts
        add_count -- whether to add the number of transactions made on every day
    Output:
        Dataframe of shape (number of clients, 2) with columns money_trans and bin_trans.
    '''
    
    # Total spent amount for every day for every client
    
    # Figuring out total spent money
    pivot_money_tr = pd.pivot_table(df, index=col_clientid, columns=[col_date,col_group],
                                    values=col_amount, aggfunc='sum', fill_value=0)
    # Taking into account all days, even if they have zero transactions
    pivot_money_tr = pivot_money_tr.stack()
    pivot_money_tr = pivot_money_tr.reindex(pd.MultiIndex.from_product(pivot_money_tr.index.levels, 
                                                                       names=pivot_money_tr.index.names), fill_value=0)
    # To numpy array
    pivot_money_tr_arr = pivot_money_tr.values.reshape(
            df[col_clientid].nunique(),3,-1) # number of clients, number of basic values, number of days
    # Filling in NaNs
    pivot_money_tr_arr[np.isnan(pivot_money_tr_arr)] = 0 
    # Adding users ids
    money_compressed = pd.DataFrame(dict(money_trans = list(pivot_money_tr_arr.astype(int))),
                                    index=np.sort(df[col_clientid].unique()))
    
    # Whether there were any transactions made for every day for every client
    
    # a day with less than 10 spent money units is considered the day with zero transactions
    pivot_bin_tr_arr = pivot_money_tr_arr.copy()
    pivot_bin_tr_arr[np.where(pivot_bin_tr_arr<10)] = 0 
    pivot_bin_tr_arr[np.where(pivot_bin_tr_arr>=10)] = 1
    # Adding users ids
    bin_compressed = pd.DataFrame(dict(bin_trans = list(pivot_bin_tr_arr.astype(int))),
                                    index=np.sort(df[col_clientid].unique()))
    # Adding everything together
    bin_compressed['money_trans'] = money_compressed['money_trans']
    
    # Number of transactions made for every day for every client
    if add_count:
        pivot_cnt_tr =pd.pivot_table(df, index=col_clientid, columns=[col_date,col_group],
                                    values=col_amount, aggfunc='count', fill_value=0)
        pivot_cnt_tr = pivot_cnt_tr.stack()
        pivot_cnt_tr = pivot_cnt_tr.reindex(pd.MultiIndex.from_product(pivot_cnt_tr.index.levels, 
                                                                       names=pivot_cnt_tr.index.names), fill_value=0)
        pivot_cnt_tr_arr = pivot_cnt_tr.values.reshape(df[col_clientid].nunique(),3,-1)
        pivot_cnt_tr_arr[np.isnan(pivot_cnt_tr_arr)] = 0
        count_compressed = pd.DataFrame(dict(count_trans= list(pivot_cnt_tr_arr)),
                                        index=np.sort(df[col_clientid].unique()))
        bin_compressed['count_trans'] = count_compressed['count_trans']
    
    return bin_compressed
    
    
def add_date_features(main_feat_trans, start_date='2020-01-01', end_date='2021-06-29'):
    '''
    Adding sine and cosine of a day of the week number and of a month number
    
    Params:
        main_feat_trans -- pd.Series with main transaction features (spent amount and binary)
        start_date -- first day (YYYY-MM-DD)
        end_date -- last day (YYYY-MM-DD)
    Output:
        pd.Series of shape (number of clients) with new features for every client for every day.
    '''
    
    dates = pd.date_range(start_date, end_date)
    n_clients = main_feat_trans.shape[0]
    n_cats = main_feat_trans.iloc[0].shape[0]
    main_feat_trans = np.concatenate(main_feat_trans.explode().values).reshape(n_clients, n_cats, -1)
    # Days of the week
    
    # Days of the week to numbers 0-6
    f = lambda x: x.weekday
    squares = f(dates)
    # sine and cosine of a day of the week number
    f = lambda x: [np.sin(x*(2.*np.pi/7)), np.cos(x*(2.*np.pi/7))]
    sin_l, cos_l = f(squares.to_numpy())
    # getting desired shape
    sin = np.tile(sin_l,(n_cats,1))
    sin = np.tile(sin,(n_clients,1,1))
    cos = np.tile(cos_l,(n_cats,1))
    cos = np.tile(cos,(n_clients,1,1))
    
    # Months to numbers 0-11
    f_m = lambda x: x.month
    squares_m = f_m(dates) - 1 # to start from 0
    # sine and cosine of a month number
    f_m = lambda x: [np.sin(x*(2.*np.pi/12)), np.cos(x*(2.*np.pi/12))]
    sin_l_m, cos_l_m = f_m((squares_m).to_numpy())
    # getting desired shape
    sin_m = np.tile(sin_l_m,(n_cats,1))
    sin_m = np.tile(sin_m,(n_clients,1,1))
    cos_m = np.tile(cos_l_m,(n_cats,1))
    cos_m = np.tile(cos_m,(n_clients,1,1))

    # to one dataframe
    # (n_clients,1) -> (n_clients, n_cats, n_days)
    feat_df = pd.DataFrame.from_records([main_feat_trans, sin, cos, sin_m, cos_m]).T
    feat_df.columns = ['feat_trans', 'dow_sin', 'dow_cos', 'm_sin', 'm_cos']
    feat_df = feat_df.apply(lambda x: np.column_stack((*[x['feat_trans'][cat,:] for cat in range(n_cats)],
                                                          x['dow_sin'][0,:], x['dow_cos'][0,:],
                                                          x['m_sin'][0,:], x['m_cos'][0,:])),
                           axis=1)

    
    return feat_df