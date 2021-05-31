"""
@author: Wei Jiang

quickGCNs是基于torch_geometric的一个包
quickGCNs 用于快速调用训练现存的经典GCN模型，并且执行网格搜索保存最优参数。

quickGCNs uses to quickly train the exist typical GCN models, 
and execute grid search to find the best hyperparameters group.

quickGCNs is based on the package torch_geometric

version 1.2
In this version only can process node-regression task.

Update on 2021/05/27
"""
from torch_geometric.data import DataLoader

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import torch.nn as nn
import torch.optim as optim
import torch

import pickle

import matplotlib.pyplot as plt
import numpy as np
import csv

import models

import os 
import random

import pandas as pd
import time

# 模型全局初始化
ChebNet = models.ChebNet
GGNN =  models.GGNN
GCN = models.GCN
GAT = models.GAT

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径

def train(train_dataset, test_dataset, train_arg_list, model_arg_list, Model, use_cuda):
    bs, wd, lr, ep = train_arg_list

    loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 建立 model 和 参数设置
    model = Model(model_arg_list)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    loss_list=[]

    # check cuda
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        device = torch.device('cuda')

    # train
    for epoch in range(ep):
        
        # 使用 cuda
        if (use_gpu) and (use_cuda):
            model = model.to(device)

        model.train()

        total_loss = 0
        i=0
        for batch in loader:
            opt.zero_grad()
            
            if (use_gpu) and (use_cuda):
                batch = batch.to(device)

            pred = model(batch)
            label = batch.y

            if (use_gpu) and (use_cuda):
                label = label.to(device)


            loss = criterion(pred, label)

            if (use_gpu) and (use_cuda):
                loss = loss.to(device)
            
            loss.backward()

            opt.step()
            i+=1
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)

        loss_list.append(total_loss)
        
    # 清理cuda缓存
    if (use_gpu) and (use_cuda):
        torch.cuda.empty_cache()
    
    # test model
    if (use_gpu) and (use_cuda):
        model.cpu()

    test_preds, labels = test(test_loader, model)

    return model, test_preds, labels, loss_list


def test(loader, model):
    model.eval()
    i=0
    for test_data in loader:
        with torch.no_grad():
            test_pred = model(test_data)
            label = test_data.y

            if i==0:
                test_preds = test_pred
                labels = label
            else:
                test_preds = torch.cat((test_preds, test_pred), 1) 
                labels = torch.cat((labels, label), 1) 
            i+=1

    return test_preds, labels


def eval(y_true, y_predict):
    
    mse =  mean_squared_error(y_true, y_predict)
    rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    mae = mean_absolute_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)
    
    return  mse, rmse, mae, r2

def plot_result(test_preds, labels, save_path, model_name):
    for i in range(len(labels)):
        plt.plot(range(len(test_preds[i])), test_preds[i], label='prediction')
        plt.plot(range(len(labels[i])), labels[i], 'r--', label='y_true')
        plt.legend()
        mkdir(save_path + '/figure/node_'+ str(i) + '/')
        plt.savefig(save_path + '/figure/node_'+ str(i) + '/' + 'best_' + model_name + '_pred_line_plot_node_' + str(i) + '.pdf', dpi=1000, bbox_inches='tight')
        plt.close()

def plot_loss(loss_list, save_path, model_name):
    
    plt.plot(range(len(loss_list)), loss_list, label='prediction')
    mkdir(save_path + '/figure/loss/')
    plt.savefig(save_path + '/figure/loss/' + model_name + '.pdf', dpi=1000, bbox_inches='tight')
    plt.close()

def save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list):
    for i in range(len(labels)):
        rows = []
        for t in range(len(pred_list)):
            test_preds = pred_list[t]
            mse, rmse, mae, r2 = eval(labels[i], test_preds[i])
            tmp_row = train_arg_list + model_arg_list + [mse, rmse, mae, r2]
            rows.append(tmp_row)

        #保存一个最佳模型的多次测试结果
        mkdir(save_path + '/records/node_'+ str(i) + '/')
        with open(save_path + '/records/node_'+ str(i) + '/' + 'best_' + model_name + '_node_' + str(i) + '_record.csv','w',newline="") as f:
            headers = ['batch_size', 'weight_decay', 'learning_rate', 'EPOCH', 'feat_dims', 'hidden_dims', 'actfunc_type', 'layer_num', 'K', 'MSE', 'RMSE', 'MAE', 'R2']
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(rows)

def save(save_path, model_name, mean_preds, labels, model, mean_rows):
    # 画出预测结果
    mkdir(save_path + '/figure/')
    plot_result(mean_preds, labels, save_path, model_name)

    # 保存模型
    mkdir(save_path + '/model/')
    torch.save(model.state_dict(), save_path + '/model/' + 'best_' + model_name + '.pt')

    # 保存预测结果变量
    mkdir(save_path + '/pred_result/')
    pickle.dump(mean_preds, open(save_path + '/pred_result/'+ 'best_' + model_name + '_pred_results' + '.pkl', 'wb'))

    # 保存最佳模型的参数和结果
    
    with open(save_path + '/records/'+ 'best_' + model_name + '_mean_record.csv','w',newline="") as f:
        headers = ['node_id', 'batch_size', 'weight_decay', 'learning_rate', 'EPOCH', 'train_times', 'feat_dims', 'hidden_dims', 'actfunc_type', 'layer_num', 'K', 'MSE', 'RMSE', 'MAE', 'R2']
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(mean_rows)
    

def grid_search(train_dataset, test_dataset, Model, eval_score, train_times, save_path, model_name, use_cuda, id):

    # records
    best_score = 0

    # training 相关超参数
    train_arg = {
    'batch_size': [8, 16, 32], 
    'weight_decay': [0, 1e-1, 1e-2, 1e-3, 1e-4],
    'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
    'EPOCH': [20],
    'train_times' : train_times

    # testing args
    # 'weight_decay': [0],
    # 'learning_rate': [1e-1],
    # 'EPOCH': [11],
    # 'train_times' : train_times
    }

    # model 相关超参数
    model_arg = {
    'feat_dims': train_dataset[0].num_node_features,
    'hidden_dims': [8, 16, 32, 64, 128],
    'actfunc_type': ['relu'],
    'layer_num': [1, 2, 3, 4],
    'K':[1, 2, 3, 4]

    # # testing args
    # 'feat_dims': dataset[0].num_node_features, 
    # 'hidden_dims': [16],
    # 'actfunc_type': ['relu'],
    # 'layer_num': [1,2],
    # 'K':[1]
    }

    # 超参数初始化
    batch_size = train_arg['batch_size']
    weight_decay = train_arg['weight_decay']
    learning_rate = train_arg['learning_rate']
    EPOCH = train_arg['EPOCH']
    train_times = train_arg['train_times']

    feat_dims = model_arg['feat_dims']
    hidden_dims = model_arg['hidden_dims']
    actfunc_type = model_arg['actfunc_type']
    layer_num = model_arg['layer_num']
    K = model_arg['K']
    
    if(os.path.exists(save_path + '/model_run_check_point_log.csv')):
        data = pd.read_csv(save_path + '/model_run_check_point_log.csv')
        if(data.shape[0]>1):
            id_log = int(data.values[-1,0])
        else:
            id_log = 0
    else:
            id_log = 0
            
    # 网格搜索
    for bs in batch_size:
        for wd in weight_decay:
            for lr in learning_rate:
                for ep in EPOCH:
                    #  GGNN 特殊情况 in_channel=outchannel
                    if model_name == 'GGNN':
                        hidden_dims = [feat_dims]
                    for hd in hidden_dims:
                        for at in actfunc_type:
                            for ln in layer_num:
                                # ChebNet特殊参数k
                                if model_name != 'ChebNet':
                                    K=[0]
                                for k in K:
                                    # 断点check
                                    if(id_log>=id):
                                        id+=1
                                    else:
                                        train_arg_list = [bs, wd, lr, ep]
                                        model_arg_list = [feat_dims, hd, at, ln, k]

                                        # 多次训练测试稳定
                                        pred_list = [] # 用于记录
                                        for t in range(train_times):
                                            time1 = time.time()

                                            model, test_preds, labels, loss_list = train(train_dataset, test_dataset, train_arg_list, model_arg_list, Model, use_cuda)

                                            time2 = time.time()

                                            print('Start to train', model_name, 'with args:', train_arg_list + model_arg_list, str(t+1), 'times, used', "%.2f" % (time2-time1), 'seconds')
                                            
                                            pred_list.append(test_preds)

                                            if t==0:
                                                sum_preds = test_preds
                                            else:
                                                sum_preds = sum_preds + test_preds
                                        
                                        mean_preds = sum_preds / train_times

                                        mean_rows = []
                                        for i in range(len(labels)):

                                            mean_mse, mean_rmse, mean_mae, mean_r2 = eval(labels[i], mean_preds[i])
                                            mean_row = [i] + train_arg_list + [train_times] + model_arg_list + [mean_mse, mean_rmse, mean_mae, mean_r2]
                                            mean_rows.append(mean_row)
                                            if i==0:
                                                sum_mean_mse = mean_mse
                                                sum_mean_rmse = mean_rmse
                                                sum_mean_mae = mean_mae
                                                sum_mean_r2 = mean_r2
                                            else:
                                                sum_mean_mse = mean_mse + sum_mean_mse
                                                sum_mean_rmse = mean_rmse + sum_mean_rmse
                                                sum_mean_mae = mean_mae + sum_mean_mae
                                                sum_mean_r2 = mean_r2 + sum_mean_r2
                                        
                                        total_mean_mse = sum_mean_mse/len(labels)
                                        total_mean_rmse = sum_mean_rmse/len(labels)
                                        total_mean_mae = sum_mean_mae/len(labels)
                                        total_mean_r2 = sum_mean_r2/len(labels)

                                        # 保存每个模型第一次运行组合结果，如果后门有更好的则被覆盖
                                        mkdir(save_path + '/records/')
                                        # 保存每个模型第一次运行组合结果，如果后门有更好的则被覆盖
                                        if(os.path.exists(save_path + '/records/'+ 'best_' + model_name + '_mean_record.csv')) is False:  
                                            plot_loss(loss_list, save_path, model_name)      
                                            # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                                            save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图

                                        # 更新 best score，选择不同eval方法
                                        if eval_score == 'mse':
                                            score = total_mean_mse
                                            if score < best_score:
                                                best_score = score
                                                plot_loss(loss_list, save_path, model_name)
                                                # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                                                save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图

                                        elif eval_score == 'rmse':
                                            score = total_mean_rmse
                                            if score < best_score:
                                                best_score = score
                                                plot_loss(loss_list, save_path, model_name)
                                                # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                                                save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图

                                        elif eval_score == 'mae':
                                            score = total_mean_mae
                                            if score < best_score:
                                                best_score = score
                                                plot_loss(loss_list, save_path, model_name)
                                                # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                                                save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图

                                        elif eval_score == 'r2':
                                            score = total_mean_r2
                                            if score > best_score:
                                                best_score = score
                                                plot_loss(loss_list, save_path, model_name)
                                                # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                                                save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图 
                                        # 记录断点log文件
                                        mkdir(save_path)
                                        with open(save_path + '/model_run_check_point_log.csv','a',newline="") as f:
                                            row = [id, model_name] + train_arg_list + model_arg_list
                                            f_csv = csv.writer(f)
                                            f_csv.writerow(row)
                                            
                                        id+=1
    
    return id

def random_search(train_dataset, test_dataset, Model, eval_score, train_times, save_path, model_name, use_cuda, id):

    # training 相关超参数
    train_arg = {
    'batch_size': [8, 16, 32], 
    'weight_decay': [0, 1e-1, 1e-2, 1e-3, 1e-4],
    'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
    'EPOCH': [15],
    'train_times' : [train_times]

    # 'batch_size': [16], 
    # 'weight_decay': [0],
    # 'learning_rate': [1e-1],
    # 'EPOCH': [10],
    # 'train_times' : [train_times]
    }

    # model 相关超参数
    model_arg = {
    'feat_dims': [train_dataset[0].num_node_features],
    'hidden_dims': [8, 16, 32, 64, 128],
    'actfunc_type': ['relu'],
    'layer_num': [1, 2, 3, 4],
    'K':[1, 2, 3, 4]

    # 'feat_dims': [train_dataset[0].num_node_features],
    # 'hidden_dims': [8],
    # 'actfunc_type': ['relu'],
    # 'layer_num': [1],
    # 'K':[1]
    }
    
    if(os.path.exists(save_path + '/model_run_check_point_log.csv')):
        data = pd.read_csv(save_path + '/model_run_check_point_log.csv')
        if(data.shape[0]>1):
            id_log = int(data.values[-1,0])
        else:
            id_log = 0
    else:
            id_log = 0
        

    MAX_EVALS = 300 # 设置忍耐度

    # 记录用
    best_score = 0

    tmp_dict = dict(train_arg, **model_arg)
    
    # 随机搜索
    for i in range(MAX_EVALS):
        random.seed(i)	# 设置随机种子，每次搜索设置不同的种子，若种子固定，那每次选取的超参都是一样的
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in tmp_dict.items()}
        
        # 超参数提取
        batch_size = hyperparameters['batch_size']
        weight_decay = hyperparameters['weight_decay']
        learning_rate = hyperparameters['learning_rate']
        EPOCH = hyperparameters['EPOCH']
        train_times = hyperparameters['train_times']

        feat_dims = hyperparameters['feat_dims']
        hidden_dims = hyperparameters['hidden_dims']
        actfunc_type = hyperparameters['actfunc_type']
        layer_num = hyperparameters['layer_num']
        K = hyperparameters['K']
        
        # ChebNet特殊参数k
        if model_name != 'ChebNet':
            K=0

        if model_name == 'GGNN':
            hidden_dims = feat_dims

        # 断点check
        if(id_log>=id):
            id+=1
        else:
            train_arg_list = [batch_size, weight_decay, learning_rate, EPOCH]
            model_arg_list = [feat_dims, hidden_dims, actfunc_type, layer_num, K]

            # 多次训练测试稳定
            pred_list = [] # 用于记录
            for t in range(train_times):
                time1 = time.time()

                model, test_preds, labels, loss_list = train(train_dataset, test_dataset, train_arg_list, model_arg_list, Model, use_cuda)

                time2 = time.time()

                print('Start to train', model_name, 'with args:', train_arg_list + model_arg_list, str(t+1), 'times, used', "%.2f" % (time2-time1), 'seconds')
                                        
                
                pred_list.append(test_preds)

                if t==0:
                    sum_preds = test_preds
                else:
                    sum_preds = sum_preds + test_preds
            
            mean_preds = sum_preds / train_times

            mean_rows = []

            for i in range(len(labels)):
                mean_mse, mean_rmse, mean_mae, mean_r2 = eval(labels[i], mean_preds[i])
                mean_row = [i] + train_arg_list + [train_times] + model_arg_list + [mean_mse, mean_rmse, mean_mae, mean_r2]
                mean_rows.append(mean_row)
                if i==0:
                    sum_mean_mse = mean_mse
                    sum_mean_rmse = mean_rmse
                    sum_mean_mae = mean_mae
                    sum_mean_r2 = mean_r2
                else:
                    sum_mean_mse = mean_mse + sum_mean_mse
                    sum_mean_rmse = mean_rmse + sum_mean_rmse
                    sum_mean_mae = mean_mae + sum_mean_mae
                    sum_mean_r2 = mean_r2 + sum_mean_r2
            
            total_mean_mse = sum_mean_mse/len(labels)
            total_mean_rmse = sum_mean_rmse/len(labels)
            total_mean_mae = sum_mean_mae/len(labels)
            total_mean_r2 = sum_mean_r2/len(labels)
            

            mkdir(save_path + '/records/')
            # 保存每个模型第一次运行组合结果，如果后门有更好的则被覆盖
            if(os.path.exists(save_path + '/records/'+ 'best_' + model_name + '_mean_record.csv')) is False:  
                plot_loss(loss_list, save_path, model_name)      
                # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图

            # 更新 best score，选择不同eval方法
            if eval_score == 'mse':
                score = total_mean_mse
                if score < best_score:
                    best_score = score
                    # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                    plot_loss(loss_list, save_path, model_name)
                    save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图

            elif eval_score == 'rmse':
                score = total_mean_rmse
                if score < best_score:
                    best_score = score
                    # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                    plot_loss(loss_list, save_path, model_name)
                    save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图

            elif eval_score == 'mae':
                score = total_mean_mae
                if score < best_score:
                    best_score = score
                    # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                    plot_loss(loss_list, save_path, model_name)
                    save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图

            elif eval_score == 'r2':
                score = total_mean_r2
                if score > best_score:
                    best_score = score
                    # save_rows(save_path, model_name, labels, pred_list, train_arg_list, model_arg_list) # 保存最佳模型多次测试结果
                    plot_loss(loss_list, save_path, model_name)
                    save(save_path, model_name, mean_preds, labels, model, mean_rows) # 保存最佳模型平均测试结果、画图 

            # 记录断点log文件
            mkdir(save_path)
            with open(save_path + '/model_run_check_point_log.csv','a',newline="") as f:
                row = [id, model_name] + train_arg_list + model_arg_list
                f_csv = csv.writer(f)
                f_csv.writerow(row)
                
            id+=1
    
    return id
                                       
def fit(train_dataset, test_dataset, eval_score, train_times, save_path='.', search_method='random', use_cuda=False):

    mkdir(save_path)
    Models = [models.ChebNet, models.GGNN, models.GCN, models.GAT]
    model_name = ['ChebNet', 'GGNN', 'GCN', 'GAT']
    
    id = 1 # 用于记录断点 

    for idx, Model in enumerate(Models):
        if search_method=='grid':
            id = grid_search(train_dataset, test_dataset, Model, eval_score, train_times, save_path, model_name[idx], use_cuda, id)
        elif search_method=='random':
            id = random_search(train_dataset, test_dataset, Model, eval_score, train_times, save_path, model_name[idx], use_cuda, id)
