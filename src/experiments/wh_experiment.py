import torch
import torch.backends.cudnn as cudnn
import torch.backends.cuda as cuda
import time
import numpy as np
import os
import json
import torch.optim as optim
import datetime as dt
import tqdm
import io
import contextlib

from beeprint import pp
from experiments.experiment_interface import ExperimentInterface
from experiments.wh_experiment_config import WH_oneout_ConfigExperiment, WH_twoout_ConfigExperiment, WH_multimodel_ConfigExperiment
from configs.config import Config
from about_models.models_utils.model_utils import get_num_parameters
from data.data_loader import silhouettes_loader, loader_repr
from about_models import models
from about_models.models_utils.train_utils_regression_2out import train_model as train_model_twoout
from about_models.models_utils.train_utils_regression_2out import test_model as test_model_twoout
from about_models.models_utils.train_utils_regression_1out import train_model as train_model_oneout
from about_models.models_utils.train_utils_regression_1out import test_model as test_model_oneout

from utils.utils_plots_regression import save_plot_prediction, save_plot_prediction_1out
from utils.utils_json import saveorupdate_json
from utils.utils_time import convert_seconds, get_time_string
from utils.utils_pdf_regression import ManagePDF_R

class WHPredExperiment(ExperimentInterface):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.line =  "_" * 100
        
    def define_data(self, config: Config):
        data_config = config.CONFIG["current_project_info"]["data_config"]
        
        train_loader, val_loader, test_loader, infe_loader = None, None, None, None
            
        #* train/test
        if data_config["split_data_type"]["use"] == data_config["split_data_type"]["choices"][1] or data_config["split_data_type"]["use"] == data_config["split_data_type"]["choices"][2]:    
            train_loader, test_loader = silhouettes_loader(config=config)
            print('Train:')
            print(loader_repr(train_loader))
            print('\nTest:')
            print(loader_repr(test_loader))
        
        return train_loader, val_loader, test_loader, infe_loader

    def define_model_and_train(self, config, train_loader, test_loader, val_loader = None):
        train_config = config.CONFIG["current_project_info"]["train_infe_config"]
        
        model_and_train = self.define_model_twoOut_and_train
            
        model_and_train(config=config, train_loader=train_loader, test_loader=test_loader)
    
    #* Use multimodel function
    def define_model_twoOut_and_train(self, config, train_loader, test_loader):
        
        #* initiate config
        wh_config = WH_twoout_ConfigExperiment(config)
        model = wh_config.model
        
        #* model parameters
        num_params = get_num_parameters(model)
        print('\n[MODEL] num_params:', num_params)
        
        #* print parameters per modules
        for name, module in model.named_children():
            print("Module Name: {}".format(name))
            # print(module)
            s_print = []
            for name, param in module.named_parameters():
                s_print.append("## Param {} :: trainable {} ## ".format(name, param.requires_grad))
            print("   {}".format(s_print))
        
        print('\n[DEVICE] Device: {}'.format(wh_config.device))
        num_gpus = None
        if wh_config.use_cuda:
            num_gpus = torch.cuda.device_count()
            cudnn.enabled = True
            cudnn.benchmark = True
            model.cuda()
            if num_gpus > 1:
                model = torch.nn.DataParallel(model, range(num_gpus))
            print('[DEVICE] model is using {} GPU(s)'.format(num_gpus))
        
        #* verifying in witch device the model is.
        print("[DEVICE] Model is in Device: {}".format(next(model.parameters()).device))
        
        #* optimizaer
        print("[OPTIMIZER]: ")
        print(wh_config.optimizer)
        print('Inicial Learning Rate: {}'.format(wh_config.optimizer.param_groups[0]['lr']))
        
        #* scheduler       
        print("[SCHEDULER]: ")
        pp(wh_config.scheduler_config_dict, sort_keys=False)
        print("Inicial state_dict: ")
        pp(wh_config.lr_scheduler.state_dict())
        
        #* training
        print('\n[PRINT] Training')
        print("[PRINT] Test epochs: {}".format(list(wh_config.test_epochs)))
        print("[TRAIN] Num Epochs: {}, Initial Epoch: {}, Num Epochs to execute: {}".format(wh_config.epochs, wh_config.init_epoch, wh_config.epochs - wh_config.init_epoch))
        
        #* where save models - best and final
        print("\nPaths:")
        print("Directory: {}".format(wh_config.save_model_dir))
        print("Best Loss Model Path: {}".format(wh_config.save_bestloss_model_path))
        print("Best Loss only Weight Path: {}".format(wh_config.save_bestloss_model_path_w))
        print("Best Loss only Height Path: {}".format(wh_config.save_bestloss_model_path_h))
        print("Best MAE only Weight Path: {}".format(wh_config.save_bestmae_model_path_w))
        print("Best MAE only Height Path: {}".format(wh_config.save_bestmae_model_path_h))
        print("Update Json path: {}".format(wh_config.upadate_json_path))
        print("Checkpoint Path: {}".format(wh_config.save_checkpoint_path))
        print("Final Model Path: {}".format(wh_config.save_final_path))
        print("Save PDF Path: {}".format(wh_config.save_pdf_path))
        
        #* loss function
        print("\nCriterion Weight:")
        print(wh_config.criterion_config["weight"]["name"])
        print("Criterion Height:")
        print(wh_config.criterion_config["height"]["name"])
        
        #* time and best metrics if checkpoint is considerate, the variable will be initiated accordingly
        start_time = time.time()
        print("\n[TRAIN] Inicial Best MSE Weight and MSE Height (loss): {}".format(wh_config.best_final_loss))
        print("[TRAIN] Inicial Best MSE Weight (loss): {}".format(wh_config.best_mse_weight))
        print("[TRAIN] Inicial Best MSE Height (loss): {}".format(wh_config.best_mse_height))
        print("[TRAIN] Inicial Best MSE Only-Weight (loss): {}".format(wh_config.best_mse_onlyw))
        print("[TRAIN] Inicial Best MSE Only-Height (loss): {}".format(wh_config.best_mse_onlyh))
        print("[TRAIN] Inicial Best MAE Only-Weight: {}".format(wh_config.best_mae_onlyw))
        print("[TRAIN] Inicial Best MAE Only-Height: {}".format(wh_config.best_mae_onlyh))
        print("[TRAIN] Inicial Best Epoch Weight and Height (loss): {}".format(wh_config.best_epoch_wh))
        print("[TRAIN] Inicial Best Epoch Only-Weight (loss): {}".format(wh_config.best_epoch_w))
        print("[TRAIN] Inicial Best Epoch Only-Height (loss): {}".format(wh_config.best_epoch_h))
        print("[TRAIN] Inicial Best MAE Epoch Only-Weight: {}".format(wh_config.best_mae_epoch_w))
        print("[TRAIN] Inicial Best MAE Epoch Only-Height: {}".format(wh_config.best_mae_epoch_h))
        print(config.LINE + config.LINE)
        
        verbose_output = ""
        history = {}
        results = {
            "final": None,
            "history": history
        }
        pdf_items = []
        total_running_time = 0.0
        
        #* If it is only a test, use a reduced number os epochs
        if wh_config.testing:
            print("\n[Testing] It's a test\n")
            print(config.LINE + config.LINE)
        print(self.line + self.line)
        bar_range = range(wh_config.epochs)
        with tqdm.tqdm(bar_range, unit="epochs", position=0, initial=wh_config.init_epoch, dynamic_ncols=True, leave=True) as pbar:
            
            for epoch in range(wh_config.init_epoch, wh_config.epochs):
                pbar.set_description("Epoch {}/{}".format(epoch, wh_config.epochs))
                
                #* time
                epoch_init_time = time.time()
                #* run train
                train_result_dict = train_model_twoout(
                    wh_config.model, 
                    wh_config.optimizer, 
                    train_loader, 
                    criterion_config=wh_config.criterion_config, 
                    epoch_info=[epoch, wh_config.epochs], 
                    start_time=start_time, 
                    device=wh_config.device
                )
                
                #* validation
                if epoch in wh_config.test_epochs:
                    
                    #* time
                    test_init_time = time.time()
                    
                    #* run validadtion/test
                    test_result_dict = test_model_twoout(
                        wh_config.model, 
                        test_loader, 
                        criterion_config=wh_config.criterion_config, 
                        epoch_info=[epoch, wh_config.epochs],
                        start_time=start_time, 
                        device=wh_config.device
                    )
                    
                    #* time
                    test_final_time = time.time()
                    test_elapsed_time = test_final_time - test_init_time
                    total_running_time += test_elapsed_time
                    
                    #* save better final loss (final loss is weight loss plus height loss)
                    if test_result_dict["test_mean_final_loss"] < wh_config.best_final_loss:
                        wh_config.best_mse_weight = test_result_dict["test_mean_weight_loss"]
                        wh_config.best_mse_height = test_result_dict["test_mean_height_loss"]
                        wh_config.best_final_loss = test_result_dict["test_mean_final_loss"]
                        wh_config.best_epoch_wh = epoch
                        
                        torch.save({
                            'current_epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': wh_config.optimizer.state_dict(),
                            "scheduler_state_dict": wh_config.lr_scheduler.state_dict(),
                            'best_final_loss': test_result_dict["test_mean_final_loss"],
                            'best_loss_weight': test_result_dict["test_mean_weight_loss"],
                            'best_loss_height': test_result_dict["test_mean_height_loss"],
                            "best_loss_onlyw": wh_config.best_mse_onlyw,
                            'best_loss_onlyh': wh_config.best_mse_onlyh,
                            "best_epoch_wh": epoch,
                            "best_epoch_w": wh_config.best_epoch_w,
                            "best_epoch_h": wh_config.best_epoch_h
                        }, wh_config.save_bestloss_model_path)
                    
                    #* save better weight only
                    if test_result_dict["test_mean_weight_loss"] < wh_config.best_mse_onlyw:
                        wh_config.best_mse_onlyw = test_result_dict["test_mean_weight_loss"]
                        wh_config.best_epoch_w = epoch
                        
                        torch.save({
                            'current_epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': wh_config.optimizer.state_dict(),
                            "scheduler_state_dict": wh_config.lr_scheduler.state_dict(),
                            'best_final_loss': wh_config.best_final_loss,
                            'best_loss_weight': wh_config.best_mse_weight,
                            'best_loss_height': wh_config.best_mse_height,
                            "best_loss_onlyw": test_result_dict["test_mean_weight_loss"],
                            'best_loss_onlyh': wh_config.best_mse_onlyh,
                            "best_epoch_wh": wh_config.best_epoch_wh,
                            "best_epoch_w": epoch,
                            "best_epoch_h": wh_config.best_epoch_h
                        }, wh_config.save_bestloss_model_path_w)
                        
                    #* save better height
                    if test_result_dict["test_mean_height_loss"] < wh_config.best_mse_onlyh:
                        wh_config.best_mse_onlyh = test_result_dict["test_mean_height_loss"]
                        wh_config.best_epoch_h = epoch
                        
                        torch.save({
                            'current_epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': wh_config.optimizer.state_dict(),
                            "scheduler_state_dict": wh_config.lr_scheduler.state_dict(),
                            'best_final_loss': wh_config.best_final_loss,
                            'best_loss_weight': wh_config.best_mse_weight,
                            'best_loss_height': wh_config.best_mse_height,
                            "best_loss_onlyw": wh_config.best_mse_onlyw,
                            'best_loss_onlyh': test_result_dict["test_mean_height_loss"],
                            "best_epoch_wh": wh_config.best_epoch_wh,
                            "best_epoch_w": wh_config.best_epoch_w,
                            "best_epoch_h": epoch
                        }, wh_config.save_bestloss_model_path_h)
                        
                    #* save better MAE weight
                    if test_result_dict["test_mean_weight_mae"] < wh_config.best_mae_onlyw:
                        wh_config.best_mae_onlyw = test_result_dict["test_mean_weight_mae"]
                        wh_config.best_mae_epoch_w = epoch
                        
                        torch.save({
                            'current_epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': wh_config.optimizer.state_dict(),
                            "scheduler_state_dict": wh_config.lr_scheduler.state_dict(),
                            'best_final_loss': wh_config.best_final_loss,
                            'best_loss_weight': wh_config.best_mse_weight,
                            'best_loss_height': wh_config.best_mse_height,
                            "best_loss_onlyw": wh_config.best_mse_onlyw,
                            'best_loss_onlyh': wh_config.best_mse_onlyh,
                            "best_mae_onlyw": test_result_dict["test_mean_weight_mae"],
                            'best_mae_onlyh': wh_config.best_mae_onlyh,
                            "best_epoch_wh": wh_config.best_epoch_wh,
                            "best_epoch_w": wh_config.best_epoch_w,
                            "best_epoch_h": wh_config.best_epoch_h,
                            "best_epoch_w": epoch,
                            "best_epoch_h": wh_config.best_mae_epoch_h
                        }, wh_config.save_bestmae_model_path_w)
                        
                    #* save better MAE height
                    if test_result_dict["test_mean_height_mae"] < wh_config.best_mae_onlyh:
                        wh_config.best_mae_onlyh = test_result_dict["test_mean_height_mae"]
                        wh_config.best_mae_epoch_h = epoch
                        
                        torch.save({
                            'current_epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': wh_config.optimizer.state_dict(),
                            "scheduler_state_dict": wh_config.lr_scheduler.state_dict(),
                            'best_final_loss': wh_config.best_final_loss,
                            'best_loss_weight': wh_config.best_mse_weight,
                            'best_loss_height': wh_config.best_mse_height,
                            "best_loss_onlyw": wh_config.best_mse_onlyw,
                            'best_loss_onlyh': wh_config.best_mse_onlyh,
                            "best_mae_onlyw": wh_config.best_mae_onlyw,
                            'best_mae_onlyh': test_result_dict["test_mean_height_mae"],
                            "best_epoch_wh": wh_config.best_epoch_wh,
                            "best_epoch_w": wh_config.best_epoch_w,
                            "best_epoch_h": wh_config.best_epoch_h,
                            "best_epoch_w": wh_config.best_mae_epoch_w,
                            "best_epoch_h": epoch
                        }, wh_config.save_bestmae_model_path_h)
                    
                    test_history = {
                        "test_mean_final_loss": test_result_dict["test_mean_final_loss"],
                        "test_mean_weight_loss": test_result_dict["test_mean_weight_loss"],
                        "test_mean_height_loss": test_result_dict["test_mean_height_loss"],
                        "test_mean_weight_r2": test_result_dict["test_mean_weight_r2"],
                        "test_mean_height_r2": test_result_dict["test_mean_height_r2"],
                        "test_mean_weight_mae": test_result_dict["test_mean_weight_mae"],
                        "test_mean_height_mae": test_result_dict["test_mean_height_mae"],
                        "test_mean_weight_mse": test_result_dict["test_mean_weight_mse"],
                        "test_mean_height_mse": test_result_dict["test_mean_height_mse"],
                        "test_mean_weight_rmse": test_result_dict["test_mean_weight_rmse"],
                        "test_mean_height_rmse": test_result_dict["test_mean_height_rmse"],
                        "test_mean_weight_mape": test_result_dict["test_mean_weight_mape"],
                        "test_mean_height_mape": test_result_dict["test_mean_height_mape"],
                        
                        'best_final_loss': wh_config.best_final_loss,
                        'best_loss_weight': wh_config.best_mse_weight,
                        'best_loss_height': wh_config.best_mse_height,
                        "best_loss_onlyw": wh_config.best_mse_onlyw,
                        'best_loss_onlyh': wh_config.best_mse_onlyh,
                        "best_mae_onlyw": wh_config.best_mae_onlyw,
                        'best_mae_onlyh': wh_config.best_mae_onlyh,
                        "best_epoch_wh": wh_config.best_epoch_wh,
                        "best_epoch_w": wh_config.best_epoch_w,
                        "best_epoch_h": wh_config.best_epoch_h,
                        "best_mae_epoch_w": wh_config.best_mae_epoch_w,
                        "best_mae_epoch_h": wh_config.best_mae_epoch_h,
                        
                        "test_init_time": test_init_time,
                        "test_final_time": test_final_time,
                        "test_elapsed_time": test_elapsed_time,
                        "total_running_time": total_running_time,
                        "test_elapsed_time_converted": convert_seconds(test_elapsed_time),
                        "total_running_time_convertes": convert_seconds(total_running_time),
                        
                        "test_targets": test_result_dict["test_y_target_accumulator"],
                        "test_preds": test_result_dict["test_y_pred_accumulator"],
                        "test_ids": test_result_dict["test_ids_accumulator"],
                        "test_worst_weight": test_result_dict["test_worst_weight"],
                        "test_worst_height": test_result_dict["test_worst_height"],
                        "test_worst_final_loss": test_result_dict["test_worst_final_loss"]                  
                    }
                    
                    test_history_dict = {
                        "test": test_history
                    }
                    history["epoch_" + str(epoch)] = test_history_dict
                        
                    #* save plot
                    plot_report_path = os.path.join(wh_config.save_model_dir, "test" + str(epoch) + "_pred_result.png")
                    save_plot_prediction(
                        fig_path=plot_report_path,
                        plot_title="Prediction Report = Epoch {}".format(epoch),
                        y_true_wh=test_result_dict["test_y_target_accumulator"],
                        y_pred_wh=test_result_dict["test_y_pred_accumulator"],
                        ids=test_result_dict["test_ids_accumulator"]
                    )
                    
                    #* save items to the final PDF
                    pdf_items.append(plot_report_path)
                    
                
                #* Capture verbose output in a string
                with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
                    wh_config.lr_scheduler.step()
                    verbose_output += "[Scheduler] Epoch {}\n".format(epoch)
                    verbose_output += buffer.getvalue()
                    verbose_output += "\n"
                
                epoch_final_time = time.time()
                epoch_elapsed_time = epoch_final_time - epoch_init_time
                total_running_time = epoch_final_time - test_final_time
                
                train_history = {
                    "mean_final_loss": train_result_dict["train_mean_final_loss"],
                    "mean_weight_loss": train_result_dict["train_mean_weight_loss"],
                    "mean_height_loss": train_result_dict["train_mean_height_loss"],
                    "mean_weight_r2": train_result_dict["train_mean_weight_r2"],
                    "mean_height_r2": train_result_dict["train_mean_height_r2"],
                    "mean_weight_mae": train_result_dict["train_mean_weight_mae"],
                    "mean_height_mae": train_result_dict["train_mean_height_mae"],
                    "mean_weight_mse": train_result_dict["train_mean_weight_mse"],
                    "mean_height_mse": train_result_dict["train_mean_height_mse"],
                    "mean_weight_rmse": train_result_dict["train_mean_weight_rmse"],
                    "mean_height_rmse": train_result_dict["train_mean_height_rmse"],
                    "mean_weight_mape": train_result_dict["train_mean_weight_mape"],
                    "mean_height_mape": train_result_dict["train_mean_height_mape"],
                    
                    'best_final_loss': wh_config.best_final_loss,
                    'best_loss_weight': wh_config.best_mse_weight,
                    'best_loss_height': wh_config.best_mse_height,
                    "best_loss_onlyw": wh_config.best_mse_onlyw,
                    'best_loss_onlyh': wh_config.best_mse_onlyh,
                    "best_mae_onlyw": wh_config.best_mae_onlyw,
                    'best_mae_onlyh': wh_config.best_mae_onlyh,
                    "best_epoch_wh": wh_config.best_epoch_wh,
                    "best_epoch_w": wh_config.best_epoch_w,
                    "best_epoch_h": wh_config.best_epoch_h,
                    "best_mae_epoch_w": wh_config.best_mae_epoch_w,
                    "best_mae_epoch_h": wh_config.best_mae_epoch_h,
                    
                    "init_time": epoch_init_time,
                    "final_time": epoch_final_time,
                    "elapsed_time": epoch_elapsed_time,
                    "total_running_time": total_running_time,
                    "elapsed_time_converted": convert_seconds(epoch_elapsed_time),
                    "total_running_time_convertes": convert_seconds(total_running_time),
                    
                    "targets_train": train_result_dict["train_y_pred_accumulator"],
                    "preds_train": train_result_dict["train_y_target_accumulator"],
                    "worst_weight": train_result_dict["train_worst_weight"],
                    "worst_height": train_result_dict["train_worst_height"],
                    "worst_final_loss": train_result_dict["train_worst_final_loss"],
                }
                
                history["epoch_"+str(epoch)]["train"] = train_history
                config.CONFIG["current_project_info"]["results"]["use"] = results
                saveorupdate_json(json_path=wh_config.upadate_json_path, config=config.CONFIG)
                
                #* chackpoint
                torch.save({
                    'current_epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': wh_config.optimizer.state_dict(),
                    "scheduler_state_dict": wh_config.lr_scheduler.state_dict(),
                    'best_final_loss': wh_config.best_final_loss,
                    'best_loss_weight': wh_config.best_mse_weight,
                    'best_loss_height': wh_config.best_mse_height,
                    "best_loss_onlyw": wh_config.best_mse_onlyw,
                    'best_loss_onlyh': wh_config.best_mse_onlyh,
                    "best_mae_onlyw": wh_config.best_mae_onlyw,
                    'best_mae_onlyh': wh_config.best_mae_epoch_h,
                    "best_epoch_wh": wh_config.best_epoch_wh,
                    "best_epoch_w": wh_config.best_epoch_w,
                    "best_epoch_h": wh_config.best_epoch_h,
                    "best_epoch_w": wh_config.best_mae_epoch_w,
                    "best_epoch_h": wh_config.best_mae_epoch_h
                }, wh_config.save_checkpoint_path)
                
                #* bar
                time_now = time.time()
                pbar.set_postfix({
                    "b_final_loss":"{:.3f}".format(wh_config.best_final_loss),
                    # "b_loss_weight":"{:.3f}".format(wh_config.best_mse_weight),
                    # "b_loss_height":"{:.3f}".format(wh_config.best_mse_height),
                    "t_per_epoch":convert_seconds(time_now - start_time),
                    "b_e_wh":wh_config.best_epoch_wh,
                    "b_e_w":wh_config.best_epoch_w,
                    "b_e_h":wh_config.best_epoch_h,
                    "b_mae_e_w":wh_config.best_mae_epoch_w,
                    "b_mae_e_h":wh_config.best_mae_epoch_h
                    })
                pbar.update(1)
                
        print('Training is finished')
        print("[RESULTS - BEST RESULTS]")
        print(config.LINE)      
        print('Best Weight MSE: {:.3f}'.format(wh_config.best_mse_weight))
        print('Best Height MSE: {:.3f}'.format(wh_config.best_mse_height))
        print('Best Weight RMSE: {:.3f} kg'.format(np.sqrt(wh_config.best_mse_weight)))
        print('Best Height RMSE: {:.3f} cm'.format(np.sqrt(wh_config.best_mse_height)))
        print('Best Weight-Only MSE: {:.3f}'.format(wh_config.best_mse_onlyw))
        print('Best Height-Only MSE: {:.3f}'.format(wh_config.best_mse_onlyh))
        print('Best Weight-Only RMSE: {:.3f} kg'.format(np.sqrt(wh_config.best_mse_onlyw)))
        print('Best Height-Only RMSE: {:.3f} cm'.format(np.sqrt(wh_config.best_mse_onlyh)))
        print('Best Weight-Only MAE: {:.3f} kg'.format(wh_config.best_mae_onlyw))
        print('Best Height-Only MAE: {:.3f} cm'.format(wh_config.best_mae_onlyh))
        print("Best Loss Epoch: {}".format(wh_config.best_epoch_wh))
        print("Best Epoch Weight-Only: {}".format(wh_config.best_epoch_w))
        print("Best Epoch Height-Only: {}".format(wh_config.best_epoch_h))
        print("Best MAE Epoch Weight-Only: {}".format(wh_config.best_mae_epoch_w))
        print("Best MAE Epoch Height-Only: {}".format(wh_config.best_mae_epoch_h))
        
        #* time
        total_end_time = time.time()
        total_elapsed_time = total_end_time - start_time
        time_per_epoch = total_elapsed_time / wh_config.epochs

        print('Time' + config.LINE)
        print('Start Time: {}'.format(start_time))
        print('Total End Time: {}'.format(total_end_time))
        print('Time per Epoch: {}'.format(convert_seconds(time_per_epoch)))
        print('Total Elapsed Time: {}'.format(convert_seconds(total_elapsed_time)))

        #* save final model
        torch.save({
            'current_epoch': wh_config.epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': wh_config.optimizer.state_dict(),
            "scheduler_state_dict": wh_config.lr_scheduler.state_dict(),
            'best_final_loss': wh_config.best_final_loss,
            'best_loss_weight': wh_config.best_mse_weight,
            'best_loss_height': wh_config.best_mse_height,
            "best_loss_onlyw": wh_config.best_mse_onlyw,
            'best_loss_onlyh': wh_config.best_mse_onlyh,
            "best_mae_onlyw": wh_config.best_mae_onlyw,
            'best_mae_onlyh': wh_config.best_mae_epoch_h,
            "best_epoch_wh": wh_config.best_epoch_wh,
            "best_epoch_w": wh_config.best_epoch_w,
            "best_epoch_h": wh_config.best_epoch_h,
            "best_epoch_w": wh_config.best_mae_epoch_w,
            "best_epoch_h": wh_config.best_mae_epoch_h
        }, wh_config.save_final_path)
        
        #* pdf
        mpdf = ManagePDF_R(path=wh_config.save_pdf_path, imagespath=pdf_items, erase_image=False)
        mpdf.save_pdf()
        
        #* save results json
        results["final"] = {
            'best_final_loss': wh_config.best_final_loss,
            'best_loss_weight': wh_config.best_mse_weight,
            'best_loss_height': wh_config.best_mse_height,
            "best_loss_onlyw": wh_config.best_mse_onlyw,
            'best_loss_onlyh': wh_config.best_mse_onlyh,
            "best_mae_onlyw": wh_config.best_mae_onlyw,
            'best_mae_onlyh': wh_config.best_mae_onlyh,
            "best_epoch_wh": wh_config.best_epoch_wh,
            "best_epoch_w": wh_config.best_epoch_w,
            "best_epoch_h": wh_config.best_epoch_h,
            "best_mae_epoch_w": wh_config.best_mae_epoch_w,
            "best_mae_epoch_h": wh_config.best_mae_epoch_h,
            "time_per_epoch": convert_seconds(time_per_epoch),
            "total_elapsed_time": convert_seconds(total_elapsed_time)
        }
        results["history"] = history
        config.CONFIG["current_project_info"]["results"]["use"] = results
        saveorupdate_json(json_path=wh_config.upadate_json_path, config=config.CONFIG)
    