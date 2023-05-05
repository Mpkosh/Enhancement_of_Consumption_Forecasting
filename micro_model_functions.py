# To work with data
import pandas as pd
import numpy as np
# Track progress
from tqdm.notebook import tqdm,trange
# PyTorch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class LSTM_cat_model(nn.Module):

    def __init__(self, input_size=32, hidden_size=128, to_pred=7,
                 dropout_inside=0.1,dropout_outside=0.1):
        '''
        Params:
            input_size -- number of features in input data
            hidden_size -- LSTM hidden size
            to_pred -- how many timesteps(days) are predicted at a time
            dropout_inside -- dropout between LSTM layers
            dropout_outside -- dropout after fully connected layers
        '''
        super(LSTM_cat_model, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=self.hidden_size,
                            num_layers=2,
                            dropout=dropout_inside,
                            batch_first=True,
                            bidirectional=False
                          )
        self.linear = nn.Linear(in_features=self.hidden_size, 
                                out_features=16)
        self.act = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_outside)
        self.linear1 = nn.Linear(in_features=16, 
                                 out_features=to_pred)

    def forward(self, input):
        '''
        Params:
            input -- input data, where batch_size is on 0 place!
        '''
        # h and c are full of zero by default
        out, (h, c) = self.lstm(input)
        lstm_output = h[-1,:,:].view(-1, self.hidden_size)
        
        linear_out = self.linear(lstm_output)
        linear_out = self.act(linear_out)
        linear_out = self.dropout2(linear_out)
        
        linear_out = self.linear1(linear_out)
        
        return linear_out
    
    
def adjust_learning_rate(optimizer, shrink_factor):
    #print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    #print(f"The new learning rate is {(optimizer.param_groups[0]['lr'],)}")
    

def main_s(train_loader, test_loader, the_model, loss_function, optimizer, epoch_n):
    '''
    Main function with model training and validation.
    
    Params:
        train_loader -- dataloader for train data
        test_loader -- dataloader for test data
        the_model -- PyTorch model
        loss_function
        optimizer
        epoch_n -- number of epochs
    Output:
        losses on train and test sets, the best model
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses, test_losses = [], []
    best_loss = 999
    epochs_since_improvement = 0
    checkpoint= {}
    
    # For each epoch
    for epoch in trange(epoch_n, desc='epoch'):
        batch_losses = []
        
        the_model.train() # "Turn on" training (dropout layer works)
        
        # lowering the learning rate every 10 epochs if necessary
        if epochs_since_improvement == 100:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
            adjust_learning_rate(optimizer, 0.8)
                
        # Training for each batch
        for i, (input, y) in enumerate(train_loader):
            # tensors to GPU if applicable
            input = input.to(device)
            y = y.to(device)
            
            # Getting predictions
            preds = the_model(input)
            # Zeroing gradients for them not to accumulate
            optimizer.zero_grad() 
            # Measuring loss
            loss = loss_function(preds, y) 
            loss.backward()
            # Updating weights
            optimizer.step() 

            batch_losses.append(loss.item())

        batch_losses = np.array(batch_losses)
        train_losses.append(np.mean(batch_losses))
        #print(f'TRAIN: {epoch} epoch loss: {train_losses[-1]:.4f}', end="")
        

        the_model.eval() # "Turn on" validation (dropout layer is disabled)
        
        # just in case: turn off gradient calculation
        with torch.no_grad(): 
            batch_losses = []
            # Validation for each batch
            for i, (input, y) in enumerate(test_loader):
                # tensors to GPU if applicable
                input = input.to(device)
                y = y.to(device)
                preds = the_model(input)
                # Measuring loss
                loss = loss_function(preds, y) 
                batch_losses.append(loss.item())

        batch_losses = np.array(batch_losses)
        test_losses.append(np.mean(batch_losses))
        recent_loss = test_losses[-1]
        #print(f'___TEST: {epoch} epoch loss: {recent_loss:.4f}', end="")
        
        # Saving the model with the best loss on test data
        best_loss = min(recent_loss, best_loss)
        if recent_loss > best_loss:
            epochs_since_improvement += 1
            #print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")
        else:
            epochs_since_improvement = 0
            checkpoint = {'model': the_model.state_dict(),
                          'optimizer' : optimizer.state_dict()}
            #print("Saving")
            
    #print(f'Train: {train_losses[-1]}, Test: {test_losses[-1]}')
    return train_losses, test_losses, checkpoint


class TransactionsDataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x, dtype=torch.float32)#float32
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.x.shape[0]


def pred(input, the_model):
    '''
    Oututting the model's predictions
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    the_model.eval()
    preds = the_model(input)

    # Returning logits!
    return preds
    

    
def train_model(all_x, all_y, all_test_x, all_test_y, 
                e_n=100, lr=0.0005, cat=0, n_cats=3, n_train=28):
    '''
    Training a base model (a non-incremental one)
    
    Params:
        all_x -- features on train data 
        all_y -- targets on train data
        all_test_x -- features on test data
        all_test_y -- targets on test data
        e_n -- number of epochs
        lr -- learning rate
        cat -- the number of a basic value in range [0, n_cats-1]
        n_cats -- number of basic values
        n_train -- predictions are made based on <n_train> days
    Output: a trained model, its optimizer, train losses, validation losses, checkpoint, weights for the loss function.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    x_for_train = np.concatenate((all_x[:,:,n_cats:],all_x[:,:,cat].reshape(-1,n_train,1)),axis=2)
    x_for_test = np.concatenate((all_test_x[:,:,n_cats:],all_test_x[:,:,cat].reshape(-1,n_train,1)),axis=2)
    
    # calculating weights for the loss function
    if x_for_train[:,:,-1].sum() == 0:
        weights = 1
    else:
        weights = (x_for_train[:,:,-1]==0).sum()/x_for_train[:,:,-1].sum()
    
    
    train_loader = torch.utils.data.DataLoader(TransactionsDataset(x_for_train[:], 
                                                                   all_y[:,:,cat].astype('float32')),
                                               batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)  
    test_loader = torch.utils.data.DataLoader(TransactionsDataset(x_for_test, 
                                                                  all_test_y[:,:,cat].astype('float32')),
                                               batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    the_model = LSTM_cat_model(input_size=5, hidden_size=64, to_pred=7,
                            dropout_inside=0.2,dropout_outside=0.1).to(device)
    
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([weights])).to(device)

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, the_model.parameters()),lr=lr)

    # training
    train_losses, test_losses, checkpoint = main_s(train_loader, test_loader, the_model, 
                                                                 loss_function, optimizer, epoch_n=e_n)
    
    return the_model, optimizer, train_losses, test_losses, checkpoint, weights


def base_model_data(q, model_after,days,cat=0, n_train=28, n_pred=7, n_features=7):
    '''
    The base model's predictions for test data.
    
    Params:
        q -- Time series' shifts for a given client with added date to the target.
        model_after -- the trained base model
        days -- weeks of test data
        cat -- the number of a basic value
        n_train -- predictions are made based on <n_train> days
        n_pred -- <n_pred> are predicted days at a time
        n_features -- number of features
    Outputs: real values and predictions for each test week
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # dropping the column with date information
    all_test_x = q.iloc[:,:-n_pred*(n_features+1)].drop(q.iloc[:,n_features:-n_pred*(n_features+1):n_features+1], axis=1, inplace=False
                                                       ).values.reshape(-1,n_train,n_features)
    all_test_y = q.iloc[:,-n_pred*(n_features+1):].values.reshape(-1,n_pred,(n_features+1))

    x_for_test = np.concatenate((all_test_x[:,:,3:],all_test_x[:,:,cat].reshape(-1,n_train,1)),axis=2)

    test_loader_n = torch.utils.data.DataLoader(TransactionsDataset(x_for_test, 
                                                                    all_test_y[:,:,cat].astype('float32')),
                                               batch_size=x_for_test.shape[0], shuffle=False, pin_memory=True)
    tests, y_t = next(iter(test_loader_n))
    
    # forecasting
    model_after.eval()
    preds = pred(tests, model_after)
    y = y_t.to(device)
    
    # for each test week geeting [real value, prediction]
    r=[]
    y_days = all_test_y[:,:,-1].astype('datetime64[D]')
    for week_n, week in enumerate(days):
        pred_wday = preds[week_n]
        y_wday = y[week_n]

        r.append([y_wday.detach().cpu().numpy(),
                  pred_wday.detach().cpu().numpy()])
    
    return r


def incr_model_data(q, model_after,days,optimizer_after,with_weights=True,cat=0,
                   n_train=28, n_pred=7, n_features=7, n_cats=3):
    '''
    The incremental model's predictions for test data.
    
    Params:
        q -- Time series' shifts for a given client with added date to the target.
        model_after -- the trained base model
        days -- weeks of test data
        optimizer_after -- the trained base model's optimizer
        with_weights -- whether to use loss function weights during incremental learning
        cat -- the number of a basic value
        n_pred -- <n_pred> are predicted days at a time
        n_features -- number of features
        n_train -- predictions are made based on <n_train> days
        n_cats -- number of basic values
    Output: real values and predictions for each test week
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # for each test week geeting [real value, prediction]
    i_r = []
    for week in days:
        # finding the client's days, which are in a given range
        qq = q[(week[0]==q['date+1'])&(q['date+7']==week[-1])]
        # creating features and targets
        all_test_x = qq.iloc[:,:-n_pred*(n_features+1)].drop(qq.iloc[:,n_features:-n_pred*(n_features+1):n_features+1], 
                                                             axis=1, inplace=False).values.reshape(-1,n_train,n_features)
        all_test_y = qq.iloc[:,-n_pred*(n_features+1):].values.reshape(-1,n_pred,(n_features+1))
        # choosing one basic value's transaction features
        x_for_test = np.concatenate((all_test_x[:,:,n_cats:],all_test_x[:,:,cat].reshape(-1,n_train,1)),axis=2)

        loader = torch.utils.data.DataLoader(TransactionsDataset(x_for_test, 
                                                                 all_test_y[:,:,cat].astype('float32')),
                                                   batch_size=7, shuffle=False, pin_memory=True)
        
        # calculating weights for the loss function 
        if with_weights:
            if x_for_test[:,:,-1].sum() == 0:
                weights = 1
            else:
                weights = (x_for_test[:,:,-1]==0).sum()/x_for_test[:,:,-1].sum()
            loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([weights])).to(device)
            
        else: loss_function = nn.BCEWithLogitsLoss().to(device)
        
        # forecasting and incremental learning 
        x_vals,y,preds = pred_and_train(loader,model_after,loss_function,optimizer_after)
        i_r.append([y.reshape(-1).detach().cpu().numpy(),
                    preds.detach().cpu().numpy()])
        
    return i_r


def pred_and_train(loader, the_model, loss_function, optimizer):
    '''
    Forecasting and incremental learning for the incremental model.
    
    Params:
        loader -- dataloader for test data
        the_model -- the trained base model
        loss_function
        optimizer -- the trained base model's optimizer
    Output: 
        Features, real values, predictions
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i, (x, y) in enumerate(loader):
        the_model.eval() # "Включить" режим валидации
        
        x = x.to(device)
        y = y.to(device)
        preds_before = pred(x, the_model)
        
        
        the_model.train() # "Включить" режим обучения (dropout слой будет работать)
        
        preds = the_model(x)
        optimizer.zero_grad() # обнуляем градиенты, чтобы не накапливались с предыдущих
        loss = loss_function(preds, y) # считаем ошибку
        loss.backward()
        optimizer.step() # обновляем веса
        #print(f'\nTRAIN loss: {loss.item():.4f}', end="")
        
    return x,y,preds_before
