import torch
import numpy as np
import tqdm
import time

from statistics import mean
from utils.utils_time import convert_seconds
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred):
    
    r2 = r2_score(y_true=y_true, y_pred=y_pred, multioutput="raw_values") if (len(y_true) > 2) else [0.0]
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
    
    return r2, mae, mse, rmse, mape

def train_model(model, optimizer, loader, criterion, epoch_info, start_time, target_whb, device=torch.device('cuda')):
    model.train()
    
    y_pred_accumulator = []
    y_target_accumulator = []
    
    running_loss = []
    running_time = 0.0
    
    worst_loss = {"id_"+str(x):{"mse": 0.0, "path": "", "id": 0} for x in range(10)}
    worst_mae = {"id_"+str(x):{"mae": 0.0, "path": "", "id": 0} for x in range(10)}
    
    current_epoch = epoch_info[0] if isinstance(epoch_info[0], str) else epoch_info[0]
    
    loader_size = len(loader)
    bar_range = range(loader_size)
    with tqdm.tqdm(bar_range, unit="batchs", position=1, dynamic_ncols=True) as bar:
        for batch_idx, (data, info, target) in enumerate(loader):
            
            #* bar
            bar.set_description("Train Epoch-{}/{} Batch-{}/{}".format(current_epoch, epoch_info[1] - 1, batch_idx, loader_size -1))
            
            #* time
            batch_init_time = time.time()
            
            #* prediction
            # print("[DEVICE] Device: {}".format(device))
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            #* only weight or height
            if target_whb == "weight":
                target = target[:, 0]
            elif target_whb == "height":
                target = target[:, 1]
            elif target_whb == "bmi":
                target = target[:, 2]
            output = output.view_as(target)
            
            loss = criterion(output, target)
            running_loss.append(float(loss))
            
            #* running metrics
            y_pred = output.numpy(force=True).tolist()
            y_true = target.numpy(force=True).tolist()
            
            y_pred_accumulator += y_pred
            y_target_accumulator += y_true
            
            r_r2, r_mae, r_mse, r_rmse, r_mape = calculate_metrics(y_true=y_target_accumulator, y_pred=y_pred_accumulator)
            #* current metrics
            r2, mae, mse, rmse, mape = calculate_metrics(y_true=y_true, y_pred=y_pred)
            
            #* time
            batch_final_time = time.time()
            batch_elapsed_time = batch_final_time - batch_init_time
            running_time += batch_elapsed_time
            total_running_time = batch_final_time - start_time
            
            bar.update(1)
            bar.set_postfix({
                "loss":"{:.3f}".format(float(loss)),
                "mse":"{:.3f}".format(mse[0]),
                "rmse":"{:.3f}".format(rmse[0]),
                "mae":"{:.3f}".format(mae[0]),
                "r2":"{:.3f}".format(r2[0]),
                "r_rmse":"{:.3f}".format(r_rmse[0]),
                "r_mae":"{:.3f}".format(r_mae[0]),
                "r_r2":"{:.3f}".format(r_r2[0]),
                "r_time":convert_seconds(running_time),
                "total_time":convert_seconds(total_running_time)
                })
            
            #* save worst prediction
            w_mse = ((output - target) **2)
            w_mae = torch.abs(output - target)
            for idx, m in enumerate(w_mse):
                for k_l, v_l in worst_loss.items():
                    if v_l["mse"] < m:
                        v_l["mse"] = float(m)
                        v_l["path"] = info[0][idx]
                        v_l["id"] = int(info[1][idx])
                        worst_loss = dict(sorted(worst_loss.items(), key=lambda x: x[1]["mse"]))
                        break
            
                for k_h, v_h in worst_mae.items():
                    if v_h["mae"] < w_mae[idx]:
                        v_h["mae"] = float(w_mae[idx])
                        v_h["path"] = info[0][idx]
                        v_h["id"] = int(info[1][idx])
                        worst_mae = dict(sorted(worst_mae.items(), key=lambda x: x[1]["mae"]))
                        break
            
            loss.backward()
            optimizer.step()
    
    f_r2, f_mae, f_mse, f_rmse, f_mape = calculate_metrics(y_true=y_target_accumulator, y_pred=y_pred_accumulator)
    
    return_dict ={
        "train_mean_loss": mean(running_loss),
        "train_mean_r2": f_r2[0],
        "train_mean_mae": f_mae[0],
        "train_mean_mse": f_mse[0],
        "train_mean_rmse": f_rmse[0],
        "train_mean_mape": f_mape[0],
        "train_worst_loss": worst_loss,
        "train_worst_mae": worst_mae,
        "train_y_pred_accumulator": y_pred_accumulator,
        "train_y_target_accumulator": y_target_accumulator
    }
    
    return return_dict


def test_model(model, loader, criterion, epoch_info, start_time, target_whb, device=torch.device('cuda')):
    model.eval()
    
    running_loss = []
    running_time = 0.0
    
    worst_loss = {"id_"+str(x):{"mse": 0.0, "path": "", "id": 0} for x in range(10)}
    worst_mae = {"id_"+str(x):{"mae": 0.0, "path": "", "id": 0} for x in range(10)}
    
    targets_accumulator = []
    preds_accumulator = []
    ids_accumulator = []
    
    current_epoch = epoch_info[0] if isinstance(epoch_info[0], str) else epoch_info[0]
    
    loader_size = len(loader)
    bar_range = range(len(loader))
    with tqdm.tqdm(bar_range, unit="batchs", position=1, dynamic_ncols=True) as bar:
        with torch.no_grad():
            for batch_idx, (data, info, target) in enumerate(loader):
                
                #* bar
                bar.set_description("Test Epoch-{}/{} Batch-{}/{}".format(current_epoch, epoch_info[1] - 1, batch_idx, loader_size - 1))
                
                #* time
                batch_init_time = time.time()
                
                #* prediction
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                
                #* only weight or height
                if target_whb == "weight":
                    target = target[:, 0]
                elif target_whb == "height":
                    target = target[:, 1]
                elif target_whb == "bmi":
                    target = target[:, 2]
                output = output.view_as(target)
                
                loss = criterion(output, target)
                
                #* running metrics
                y_pred = output.numpy(force=True).tolist()
                y_true = target.numpy(force=True).tolist()
                
                running_loss.append(float(loss))
                
                #* current metrics
                r2, mae, mse, rmse, mape = calculate_metrics(y_true=y_true, y_pred=y_pred)
                
                #* store targets and preds
                targets_accumulator += y_true
                preds_accumulator += y_pred
                ids_accumulator += info[1].numpy(force=True).tolist()
                
                #* running metrics
                r_r2, r_mae, r_mse, r_rmse, r_mape = calculate_metrics(y_true=targets_accumulator, y_pred=preds_accumulator)
                
                #*time
                batch_final_time = time.time()
                batch_elapsed_time = batch_final_time - batch_init_time
                running_time += batch_elapsed_time
                total_running_time = batch_final_time - start_time
                
                bar.update(1)
                bar.set_postfix({
                    "mse":"{:.3f}".format(mse[0]),
                    "rmse":"{:.3f}".format(rmse[0]),
                    "mae":"{:.3f}".format(mae[0]),
                    "r2":"{:.3f}".format(r2[0]),
                    "r_rmse":"{:.3f}".format(r_rmse[0]),
                    "r_mae_w":"{:.3f}".format(r_mae[0]),
                    "r_r2":"{:.3f}".format(r_r2[0]),
                    "r_time":convert_seconds(running_time),
                    "total_time":convert_seconds(total_running_time)
                    })
                
                #* save worst prediction
                mse_list = ((output - target) **2)
                mae_list = torch.abs(output - target)
                
                for idx, mse in enumerate(mse_list):
                    for k_l, v_l in worst_loss.items():
                        if v_l["mse"] < mse:
                            v_l["mse"] = float(mse)
                            v_l["path"] = info[0][idx]
                            v_l["id"] = int(info[1][idx])
                            worst_loss = dict(sorted(worst_loss.items(), key=lambda x: x[1]["mse"]))
                            break
                
                    for k_m, v_m in worst_mae.items():
                        if v_m["mae"] < mae_list[idx]:
                            v_m["mae"] = float(mae_list[idx])
                            v_m["path"] = info[0][idx]
                            v_m["id"] = int(info[1][idx])
                            worst_mae = dict(sorted(worst_mae.items(), key=lambda x: x[1]["mae"]))
                            break
    
    f_r2, f_mae, f_mse, f_rmse, f_mape = calculate_metrics(y_true=targets_accumulator, y_pred=preds_accumulator)
    
    return_dict ={
        "test_mean_loss": mean(running_loss),
        "test_mean_r2": f_r2[0],
        "test_mean_mae": f_mae[0],
        "test_mean_mse": f_mse[0],
        "test_mean_rmse": f_rmse[0],
        "test_mean_mape": f_mape[0],
        "test_worst_loss": worst_loss,
        "test_worst_mae": worst_mae,
        "test_y_pred_accumulator": preds_accumulator,
        "test_y_target_accumulator": targets_accumulator,
        "test_ids_accumulator": ids_accumulator
    }
    
    return return_dict
