import torch
import numpy as np
import tqdm
import time

from statistics import mean
from utils.utils_time import convert_seconds
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred, num_targets):
    
    r2 = r2_score(y_true=y_true, y_pred=y_pred, multioutput="raw_values") if (len(y_true) > 2) else [0.0] * num_targets
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
    rmse = [np.sqrt(m) for m in mse]
    mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
    
    return r2, mae, mse, rmse, mape

def train_model(model, optimizer, loader, criterion_config, epoch_info, start_time, device=torch.device('cuda')):
    model.train()
    
    y_pred_accumulator = []
    y_target_accumulator = []
    
    num_targets = len(criterion_config.keys())
    
    final_dict = {}
    final_dict["running_loss"] = []
    final_dict["worst_losses"] = {"id_"+str(x):{"mse": 0.0, "path": "", "id": 0} for x in range(10)}
        
    for key in criterion_config.keys(): 
        criterion_config[key]["running_loss"] = []
        criterion_config[key]["worst_losses"] = {"id_"+str(x):{"mse": 0.0, "path": "", "id": 0} for x in range(10)}
    
    data_mean = loader.dataset.overall_mean
    data_std = loader.dataset.overall_std
    
    running_time = 0.0
    
    current_epoch = epoch_info[0] if isinstance(epoch_info[0], str) else epoch_info[0]
    
    loader_size = len(loader)
    bar_range = range(loader_size)
    with tqdm.tqdm(bar_range, unit="batchs", position=1, dynamic_ncols=True) as bar:
        for batch_idx, (data, info, target) in enumerate(loader):
            
            #* bar
            bar.set_description("Train Epoch-{}/{} Batch-{}/{}".format(current_epoch, epoch_info[1], batch_idx, loader_size))
            
            #* time
            batch_init_time = time.time()
            
            #* prediction
            # print("\n[TRAIN] target type: {}".format(type(target)))
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output_weight, output_height, output_bmi = model(data)
            
            target_weight = target[:, 0]
            target_height = target[:, 1]
            target_bmi = target[:, 2]
            
            target_to_cat = []
            if "weight" in criterion_config.keys():
                output_weight = output_weight.view_as(target_weight)
                criterion_config["weight"]["loss"] = criterion_config["weight"]["criterion"](output_weight, target_weight)
                criterion_config["weight"]["running_loss"].append(float(criterion_config["weight"]["loss"]))
                criterion_config["weight"]["output"] = output_weight
                criterion_config["weight"]["mse_list"] = ((output_weight - target_weight) **2)
                target_to_cat.append(target_weight.unsqueeze(1))
                
            if "height" in criterion_config.keys():
                output_height = output_height.view_as(target_height)
                criterion_config["height"]["loss"] = criterion_config["height"]["criterion"](output_height, target_height)
                criterion_config["height"]["running_loss"].append(float(criterion_config["height"]["loss"]))
                criterion_config["height"]["output"] = output_height
                criterion_config["height"]["mse_list"] = ((output_height - target_height) **2)
                target_to_cat.append(target_height.unsqueeze(1))
                
            if "bmi" in criterion_config.keys():
                output_bmi = output_bmi.view_as(target_bmi)
                criterion_config["bmi"]["loss"] = criterion_config["bmi"]["criterion"](output_bmi, target_bmi)
                criterion_config["bmi"]["running_loss"].append(float(criterion_config["bmi"]["loss"]))
                criterion_config["bmi"]["output"] = output_bmi
                criterion_config["bmi"]["mse_list"] = ((output_bmi - target_bmi) **2)
                target_to_cat.append(target_bmi.unsqueeze(1))
            
            target = torch.cat(target_to_cat, dim=1)
            
            list_loss = []
            for k in criterion_config.keys():
                list_loss.append(criterion_config[k]["loss"])
            final_loss = torch.stack(list_loss, dim=0)
            final_loss = torch.sum(final_loss)
            # print(final_loss)
            
            final_dict["running_loss"].append(float(final_loss))
            list_losses_tensor = [criterion_config[k]["mse_list"] for k in criterion_config.keys()]
            stacked_losses_tensor = torch.stack(list_losses_tensor, dim=0)
            final_dict["mse_list"] = torch.sum(stacked_losses_tensor, dim=0)
            
            #* running metrics
            output_list_of_tuples = (criterion_config[k]["output"].numpy(force=True).tolist() for k in criterion_config.keys())
            zipped = zip(*output_list_of_tuples)
            y_pred = [list(out) for out in zipped]
            y_true = target.numpy(force=True).tolist()
            
            # print("y_pred: {}".format(y_pred))
            # print("y_true: {}".format(y_true))
            
            if data_mean and data_std:
                y_pred_accumulator += [[(p[0] * data_std) + data_mean, (p[1] * data_std) + data_mean] for p in y_pred]
                y_target_accumulator += [[(t[0] * data_std) + data_mean, (t[1] * data_std) + data_mean] for t in y_true]
            else:
                y_pred_accumulator += y_pred
                y_target_accumulator += y_true
                
            r_r2, r_mae, r_mse, r_rmse, r_mape = calculate_metrics(y_true=y_target_accumulator, y_pred=y_pred_accumulator, num_targets=num_targets)
            #* current metrics
            r2, mae, mse, rmse, mape = calculate_metrics(y_true=y_true, y_pred=y_pred, num_targets=num_targets)
            
            #* time
            batch_final_time = time.time()
            batch_elapsed_time = batch_final_time - batch_init_time
            running_time += batch_elapsed_time
            total_running_time = batch_final_time - start_time
            
            bar_dict = {}
            for key in criterion_config.keys():
                bar_dict["rmse_"+key[0]] = "{:.3f}".format(rmse[criterion_config[key]["idx"]])
                bar_dict["mae_"+key[0]] = "{:.3f}".format(mae[criterion_config[key]["idx"]])
                bar_dict["r_rmse_"+key[0]] = "{:.3f}".format(r_rmse[criterion_config[key]["idx"]]) 
                bar_dict["r_mae_"+key[0]] = "{:.3f}".format(r_mae[criterion_config[key]["idx"]])
            bar_dict["r_time"] = convert_seconds(running_time)
            bar_dict["total_time"] = convert_seconds(total_running_time)
                
            bar.update(1)
            bar.set_postfix(bar_dict)
            
            #* save worst prediction
            for idx in range(len(data)):
                for key in criterion_config.keys():
                    for k, v in criterion_config[key]["worst_losses"].items():
                        if v["mse"] < criterion_config[key]["mse_list"][idx]:
                            v["mse"] = float(criterion_config[key]["mse_list"][idx])
                            v["path"] = info[0][idx]
                            v["id"] = int(info[1][idx])
                            criterion_config[key]["worst_losses"] = dict(sorted(criterion_config[key]["worst_losses"].items(), key=lambda x: x[1]["mse"]))
                            break
                for k, v in final_dict["worst_losses"].items():
                    if v["mse"] < final_dict["mse_list"][idx]:
                        v["mse"] = float(final_dict["mse_list"][idx])
                        v["path"] = info[0][idx]
                        v["id"] = int(info[1][idx])
                        final_dict["worst_losses"] = dict(sorted(final_dict["worst_losses"].items(), key=lambda x: x[1]["mse"]))
                        break
                    
            final_loss.backward()
            optimizer.step()
    
    f_r2, f_mae, f_mse, f_rmse, f_mape = calculate_metrics(y_true=y_target_accumulator, y_pred=y_pred_accumulator, num_targets=num_targets)
    
    return_dict = {}
    return_dict["train_mean_final_loss"] = mean(final_dict["running_loss"])
    return_dict["train_worst_final_loss"] = final_dict["worst_losses"]
    for key in criterion_config.keys():
        return_dict["train_mean_"+key+"_loss"] = mean(criterion_config[key]["running_loss"]) 
        return_dict["train_mean_"+key+"_r2"] = f_r2[criterion_config[key]["idx"]]
        return_dict["train_mean_"+key+"_mae"] = f_mae[criterion_config[key]["idx"]]
        return_dict["train_mean_"+key+"_mse"] = f_mse[criterion_config[key]["idx"]]
        return_dict["train_mean_"+key+"_rmse"] = f_rmse[criterion_config[key]["idx"]]
        return_dict["train_mean_"+key+"_mape"] = f_mape[criterion_config[key]["idx"]]
        return_dict["train_worst_"+key] = criterion_config[key]["worst_losses"]
    return_dict["train_y_pred_accumulator"] = y_pred_accumulator
    return_dict["train_y_target_accumulator"] = y_target_accumulator
    
    return return_dict


def test_model(model, loader, criterion_config, epoch_info, start_time, device=torch.device('cuda')):
    model.eval()
    
    running_time = 0.0
    
    num_targets = len(criterion_config.keys())
    
    final_dict = {}
    final_dict["running_loss"] = []
    final_dict["worst_losses"] = {"id_"+str(x):{"mse": 0.0, "path": "", "id": 0} for x in range(10)}
        
    for key in criterion_config.keys(): 
        criterion_config[key]["running_loss"] = []
        criterion_config[key]["worst_losses"] = {"id_"+str(x):{"mse": 0.0, "path": "", "id": 0} for x in range(10)}
    
    targets_accumulator = []
    preds_accumulator = []
    ids_accumulator = []
    
    data_mean = loader.dataset.overall_mean
    data_std = loader.dataset.overall_std
    
    current_epoch = epoch_info[0] if isinstance(epoch_info[0], str) else epoch_info[0]
    
    loader_size = len(loader)
    bar_range = range(len(loader))
    with tqdm.tqdm(bar_range, unit="batchs", position=1, dynamic_ncols=True) as bar:
        with torch.no_grad():
            for batch_idx, (data, info, target) in enumerate(loader):
                
                #* bar
                bar.set_description("Test Epoch-{}/{} Batch-{}/{}".format(current_epoch, epoch_info[1], batch_idx, loader_size))
                
                #* time
                batch_init_time = time.time()
                
                #* prediction
                data = data.to(device)
                target = target.to(device)
                output_weight, output_height, output_bmi = model(data)
                
                target_weight = target[:, 0]
                target_height = target[:, 1]
                target_bmi = target[:, 2]
                
                target_to_cat = []
                if "weight" in criterion_config.keys():
                    output_weight = output_weight.view_as(target_weight)
                    criterion_config["weight"]["loss"] = criterion_config["weight"]["criterion"](output_weight, target_weight)
                    criterion_config["weight"]["running_loss"].append(float(criterion_config["weight"]["loss"]))
                    criterion_config["weight"]["output"] = output_weight
                    criterion_config["weight"]["mse_list"] = ((output_weight - target_weight) **2)
                    target_to_cat.append(target_weight.unsqueeze(1))
                    
                if "height" in criterion_config.keys():
                    output_height = output_height.view_as(target_height)
                    criterion_config["height"]["loss"] = criterion_config["height"]["criterion"](output_height, target_height)
                    criterion_config["height"]["running_loss"].append(float(criterion_config["height"]["loss"]))
                    criterion_config["height"]["output"] = output_height
                    criterion_config["height"]["mse_list"] = ((output_height - target_height) **2)
                    target_to_cat.append(target_height.unsqueeze(1))
                    
                if "bmi" in criterion_config.keys():
                    output_bmi = output_bmi.view_as(target_bmi)
                    criterion_config["bmi"]["loss"] = criterion_config["bmi"]["criterion"](output_bmi, target_bmi)
                    criterion_config["bmi"]["running_loss"].append(float(criterion_config["bmi"]["loss"]))
                    criterion_config["bmi"]["output"] = output_bmi
                    criterion_config["bmi"]["mse_list"] = ((output_bmi - target_bmi) **2)
                    target_to_cat.append(target_bmi.unsqueeze(1))
                    
                target = torch.cat(target_to_cat, dim=1)
                
                list_loss = []
                for k in criterion_config.keys():
                    list_loss.append(criterion_config[k]["loss"])
                final_loss = torch.stack(list_loss, dim=0)
                final_loss = torch.sum(final_loss)
                # print(final_loss)
                final_dict["running_loss"].append(float(final_loss))
                list_losses_tensor = [criterion_config[k]["mse_list"] for k in criterion_config.keys()]
                stacked_losses_tensor = torch.stack(list_losses_tensor, dim=0)
                final_dict["mse_list"] = torch.sum(stacked_losses_tensor, dim=0)
                
                #* running metrics
                output_list_of_tuples = (criterion_config[k]["output"].numpy(force=True).tolist() for k in criterion_config.keys())
                zipped = zip(*output_list_of_tuples)
                y_pred = [list(out) for out in zipped]
                y_true = target.numpy(force=True).tolist()
                
                #* store targets and preds
                if data_mean and data_std:
                    preds_accumulator += [[(p[0] * data_std) + data_mean, (p[1] * data_std) + data_mean] for p in y_pred]
                    targets_accumulator += [[(t[0] * data_std) + data_mean, (t[1] * data_std) + data_mean] for t in y_true]
                    
                else:    
                    targets_accumulator += y_true
                    preds_accumulator += y_pred
                ids_accumulator += info[1].numpy(force=True).tolist()
                
                #* current metrics
                r2, mae, mse, rmse, mape = calculate_metrics(y_true=y_true, y_pred=y_pred, num_targets=num_targets)
                
                #* running metrics
                r_r2, r_mae, r_mse, r_rmse, r_mape = calculate_metrics(y_true=targets_accumulator, y_pred=preds_accumulator, num_targets=num_targets)
                
                #*time
                batch_final_time = time.time()
                batch_elapsed_time = batch_final_time - batch_init_time
                running_time += batch_elapsed_time
                total_running_time = batch_final_time - start_time
                
                bar_dict = {}
                for key in criterion_config.keys():
                    bar_dict["rmse_"+key[0]] = "{:.3f}".format(rmse[criterion_config[key]["idx"]])
                    bar_dict["mae_"+key[0]] = "{:.3f}".format(mae[criterion_config[key]["idx"]])
                    bar_dict["r_rmse_"+key[0]] = "{:.3f}".format(r_rmse[criterion_config[key]["idx"]]) 
                    bar_dict["r_mae_"+key[0]] = "{:.3f}".format(r_mae[criterion_config[key]["idx"]])
                bar_dict["r_time"] = convert_seconds(running_time)
                bar_dict["total_time"] = convert_seconds(total_running_time)
                
                bar.update(1)
                bar.set_postfix(bar_dict)
                
                #* save worst prediction
                for idx in range(len(data)):
                    for key in criterion_config.keys():
                        for k, v in criterion_config[key]["worst_losses"].items():
                            if v["mse"] < criterion_config[key]["mse_list"][idx]:
                                v["mse"] = float(criterion_config[key]["mse_list"][idx])
                                v["path"] = info[0][idx]
                                v["id"] = int(info[1][idx])
                                criterion_config[key]["worst_losses"] = dict(sorted(criterion_config[key]["worst_losses"].items(), key=lambda x: x[1]["mse"]))
                                break
                    for k, v in final_dict["worst_losses"].items():
                        if v["mse"] < final_dict["mse_list"][idx]:
                            v["mse"] = float(final_dict["mse_list"][idx])
                            v["path"] = info[0][idx]
                            v["id"] = int(info[1][idx])
                            final_dict["worst_losses"] = dict(sorted(final_dict["worst_losses"].items(), key=lambda x: x[1]["mse"]))
                            break
    
    f_r2, f_mae, f_mse, f_rmse, f_mape = calculate_metrics(y_true=targets_accumulator, y_pred=preds_accumulator, num_targets=num_targets)
    
    return_dict = {}
    return_dict["test_mean_final_loss"] = mean(final_dict["running_loss"])
    return_dict["test_worst_final_loss"] = final_dict["worst_losses"]
    for key in criterion_config.keys():
        return_dict["test_mean_"+key+"_loss"] = mean(criterion_config[key]["running_loss"]) 
        return_dict["test_mean_"+key+"_r2"] = f_r2[criterion_config[key]["idx"]]
        return_dict["test_mean_"+key+"_mae"] = f_mae[criterion_config[key]["idx"]]
        return_dict["test_mean_"+key+"_mse"] = f_mse[criterion_config[key]["idx"]]
        return_dict["test_mean_"+key+"_rmse"] = f_rmse[criterion_config[key]["idx"]]
        return_dict["test_mean_"+key+"_mape"] = f_mape[criterion_config[key]["idx"]]
        return_dict["test_worst_"+key] = criterion_config[key]["worst_losses"]
    return_dict["test_y_pred_accumulator"] = preds_accumulator
    return_dict["test_y_target_accumulator"] = targets_accumulator
    return_dict["test_ids_accumulator"] = ids_accumulator
    
    return return_dict
