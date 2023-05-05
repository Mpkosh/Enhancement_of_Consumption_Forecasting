# To work with data
import numpy as np
import macro_help_functions
# Track progress
from tqdm.notebook import tqdm,trange
from tqdm.keras import TqdmCallback
# Tensorflow
import tensorflow
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense, concatenate, Dropout, LSTM
from keras import backend as K
# preprocessing
from sklearn.preprocessing import MinMaxScaler
# Quality metrics
from sklearn.metrics import mean_absolute_percentage_error


def fit_lstm(X, y, n_epoch, n_batch, n_neurons,learning_rate=0.00005):
    '''
    Training LSTM model
    
    Params: 
        X -- features
        y -- targets
        n_epoch -- number of epochs
        n_batch -- batch size
        n_neurons -- number of neurons in LSTM
        learning_rate
    Output:
        Trained model.
    '''
    tensorflow.keras.utils.set_random_seed(42)
    # Architecture
    in1 = Input(batch_shape = (n_batch,X.shape[1], X.shape[2]))
    out = LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), 
                stateful=True, return_sequences=False, activation='relu')(in1)
    out = Dropout(0.1)(out)
    out = Dense(32, activation='relu')(out)
    out = Dropout(0.1)(out)
    x = Dense(y.shape[1], activation='linear')(out)
    model = Model(inputs=[in1], outputs=x)
    
    # Training
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error',metrics=['mape'], optimizer=optimizer,  run_eagerly=True)
    model.fit([X], y, validation_split=0.2,
                     epochs=n_epoch, batch_size=n_batch, verbose=0, callbacks=[TqdmCallback(verbose=0)], shuffle=False)

    return model


def make_forecast(model, dat,dim, mem, split, fwd):
    '''
    Forecasting.
    
    Params:
        model -- trained model
        dat -- normalized data
        dim -- predictions are made based on <dim> days
        mem -- memory depth for LSTM
        split -- number of days for test data
        fwd -- how many days are predicted one by one
    Output:
        Forecasts
    '''
    zfwd=np.array([])
    trg=dat[-(dim+mem+split+1):-split+1]
    
    # Foreasting <fwd> days one by one
    for i in range(fwd):
        X, y = macro_help_functions.MakeSet(trg, dim, mem)
        inp=[X[:1]]
        
        z=model.predict(inp, verbose=0)[0]
        # Adding the forecast to the array
        # to predict the next day based on it as well as on previous days
        zfwd=np.concatenate((zfwd, z))
        trg=np.concatenate((trg[1:], z))
    return zfwd


def make_model(scaler, X, y, dat, clients_amount,
               dim, mem, split, fwd,
               n_epochs_each_model, n_batch_each_model,n_neurons_each_model,lr_each_model):
    '''
    Creating a model and forecasting.
    
    Params:
        scaler -- sklearn scaler for normalization
        X -- features
        y -- targets
        dat -- normalized data
        clients_amount -- spents amounts
        dim -- predictions are made based on <dim> days
        mem -- memory depth for LSTM
        split -- number of days for test data
        fwd -- how many days are predicted one by one
        n_epochs_each_model -- number epochs for each model
        n_batch_each_model -- batch size for each model
        n_neurons_each_model -- number of neurons in LSTM for each model
        lr_each_model -- ;earning rate for each model
    Output:
        Trained model, MAPEs, forecasts and real data on test data, forecasts and real data on train data.
    '''
    # If the model was used before -- deleting it
    try:
        del model
    except:
        pass
    
    model = fit_lstm(X[:-1], y[:], n_epochs_each_model, n_batch_each_model, 
                                            n_neurons_each_model,learning_rate=lr_each_model)

    # Due to the use of a stateful LSTM, the batch size for training and validation
    # has to match, therefore its size was decided to be one. 
    in1 = Input(batch_shape = (1, mem, dim))
    out = LSTM(n_neurons_each_model, batch_input_shape=(1, mem, dim), 
                        stateful=True, return_sequences=False, activation='relu')(in1)
    out = Dropout(0.1)(out)
    out = Dense(32, activation='relu')(out)
    out = Dropout(0.1)(out)
    x = Dense(1, activation='linear')(out)
    model_1 = Model(inputs=[in1], outputs=x)
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr_each_model)
    model_1.compile(loss='mean_squared_error',metrics=['mape'], optimizer=optimizer,  run_eagerly=True)
    # Setting weights of the trained model to the new one 
    # just in case we'd like to use batch sizes other than 1)
    model_1.set_weights(model.get_weights())
    
    # Predicting on train data
    y_pred = []
    for i in trange(len(X), desc='Predict on train data'):
        y_pred.append(model_1.predict([X[i:i+1]], verbose=0))
    # Transform predictions back to original scale
    y_pred_tr = scaler.inverse_transform(np.array(y_pred).reshape(-1,1)).reshape(-1)
    y_true = clients_amount.iloc[mem+dim:-split].values.reshape(-1)
    
    # Forecast <fwd> days
    fwd_mapes = []
    full_fwd_pred_tr = []
    full_fwd_true = []

    fwd_pred = make_forecast(model_1, dat.reshape(-1), dim, mem, split,fwd)
    # Transform predictions back to original scale
    fwd_pred_tr = scaler.inverse_transform(fwd_pred.reshape(-1,1)).reshape(-1)
    # Real values for the same time period
    fwd_true = clients_amount.iloc[-split:-split+fwd].values.reshape(-1)

    full_fwd_pred_tr.append(fwd_pred_tr)
    full_fwd_true.append(fwd_true)
    # calculating MAPE
    fwd_mapes.append(mean_absolute_percentage_error(fwd_true, fwd_pred_tr))
    
    return model_1, fwd_mapes, full_fwd_pred_tr, full_fwd_true, y_pred_tr, y_true


def make_pred_base_model(model, scaler, dat, clients_amount,
                        dim, mem, split, fwd,
                        fwd_mapes, full_fwd_pred_tr, full_fwd_true):
    '''
    Forecasting (adding to week 0).
    
    Params:
        model -- trained model
        scaler -- sklearn scaler for normalization
        dat -- normalized data
        clients_amount -- spents amounts
        dim -- predictions are made based on <dim> days
        mem -- memory depth for LSTM
        split -- number of days for test data
        fwd -- how many days are predicted one by one
        fwd_mapes -- MAPEs for week 0
        full_fwd_pred_tr --predictions for week 0
        full_fwd_true -- real values for week 0
    Output:
        MAPEs, predictions and real values on test set.
    '''
    fwd_pred = make_forecast(model, dat.reshape(-1), dim, mem, split,fwd)
    # Transform predictions back to original scale
    fwd_pred_tr = scaler.inverse_transform(fwd_pred.reshape(-1,1)).reshape(-1)
    # Real values for the same time period
    fwd_true = clients_amount.iloc[-split:-split+fwd].values.reshape(-1)

    full_fwd_pred_tr.append(fwd_pred_tr)
    full_fwd_true.append(fwd_true)
    # calculating MAPE
    fwd_mapes.append(mean_absolute_percentage_error(fwd_true, fwd_pred_tr))
    
    return fwd_mapes, full_fwd_pred_tr, full_fwd_true


def make_forecast_inc(model, dat, dim, mem, split, fwd, ep):
    '''
    Forecasting with incremental learning.
    
    Params:
        model -- trained model
        dat -- normalized data
        dim -- predictions are made based on <dim> days
        mem -- memory depth for LSTM
        split -- number of days for test data
        fwd -- how many days are predicted one by one
        ep -- number of epochs for incremental learning
    Output:
        Predictions.
    '''
    zfwd=np.array([])
    trg=dat[-(dim+mem+split+1):-split+1]
    
    x_for_train = []
    y_for_train = []
    
    # Forecast <fwd> days
    for i in range(fwd):
        X, y = macro_help_functions.MakeSet(trg, dim, mem)
        inp=[X[:1]]
        x_for_train.append(inp[0])
        y_for_train.append(y)

        z=model.predict(inp, verbose=0)[0]
        zfwd=np.concatenate((zfwd, z))
        trg=np.concatenate((trg[1:], z))
        
    # Incremental learning on <fwd> days
    model.fit(np.stack(x_for_train,axis=1)[0], np.stack(y_for_train,axis=1)[0], validation_split=0,
               epochs=ep, batch_size=1, verbose=0, shuffle=False)
    
    return zfwd


def use_inc_model(model, scaler, dat, clients_amount,
               dim, mem, split, fwd,
                  fwd_mapes, full_fwd_pred_tr, full_fwd_true,ep):
    '''
    Forecasting (adding to week 0) with incremental learning.
    
    Params:
        model -- trained model
        scaler -- sklearn scaler for normalization
        dat -- normalized data
        clients_amount -- spents amounts
        dim -- predictions are made based on <dim> days
        mem -- memory depth for LSTM
        split -- number of days for test data
        fwd -- how many days are predicted one by one
        fwd_mapes -- MAPEs for week 0
        full_fwd_pred_tr --predictions for week 0
        full_fwd_true -- real values for week 0
        ep -- number of epochs for incremental learning
    Output:
        MAPEs, predictions and real values on test set.
    '''
    # Forecasting with incremental learning
    fwd_pred = make_forecast_inc(model, dat.reshape(-1), dim, mem, split,fwd,ep)
    # Transform predictions back to original scale
    fwd_pred_tr = scaler.inverse_transform(fwd_pred.reshape(-1,1)).reshape(-1)
    # Real values for the same time period
    fwd_true = clients_amount.iloc[-split:-split+fwd].values.reshape(-1)

    full_fwd_pred_tr.append(fwd_pred_tr)
    full_fwd_true.append(fwd_true)
    # calculating MAPE
    fwd_mapes.append(mean_absolute_percentage_error(fwd_true, fwd_pred_tr))
    
    return fwd_mapes, full_fwd_pred_tr, full_fwd_true