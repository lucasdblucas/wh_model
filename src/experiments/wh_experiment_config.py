import torch
import numpy as np
import os

from about_models import models
from experiments.experiment_config_interface import ConfigExperimentInterface
from errors import framework_error

class WH_twoout_ConfigExperiment(ConfigExperimentInterface):
    
    def __init__(self, config) -> None:
        train_config = config.CONFIG["current_project_info"]["train_infe_config"]
        task_config = config.CONFIG["current_project_info"]["task_config"]
        deep_config = train_config["deep_config"]
        regression_config = task_config["regression_config"]
        
        self.target_weight = True if "weight" in regression_config["targets"]["use"] else False
        self.target_height = True if "height" in regression_config["targets"]["use"] else False 
        self.target_bmi = True if "bmi" in regression_config["targets"]["use"] else False 
        self.num_targets = len(regression_config["targets"]["use"])
        
        #//TODO: implement the other models kind
        model = None
        
        #* get the model. If there is another type, put it here.
        self.use_cuda = train_config["cuda"]["use"] and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        #* get the model. If there is another type, put it here.
        model, init_model_dict = self.get_model(config=config)
        super(WH_twoout_ConfigExperiment, self).__init__(config=config, model=model)
        self.model = model
        
        #* load config if it should initiate from another model/state
        if init_model_dict:
            self.optimizer.load_state_dict(init_model_dict["optimizer_state_dict"])
            
            #//TODO: set the save function for scheduler state_dict too.
            # Set the current learning rate in the scheduler
            self.lr_scheduler._step_count = init_model_dict["current_epoch"] + 1
            self.lr_scheduler.last_epoch = init_model_dict["current_epoch"]
            self.lr_scheduler._last_lr = [self.optimizer.param_groups[0]['lr']]
            
            #* 'testing' is initialized in super class. And if it is 'True', the number of epochs should be actualized to initiate from the checkpoint epoch then run just the epochs defined on testing. 
            if self.testing:
                self.epochs += init_model_dict["current_epoch"] + 1
        
        self.init_epoch = 0 if init_model_dict == None else init_model_dict['current_epoch'] + 1
        assert (init_model_dict == None) or (init_model_dict['current_epoch'] + 1 <= deep_config["epochs"]["use"]), "The initial epoch value is not valid"
        
        #* loss function
        self.criterion_config = self.get_criterion_config()
        
        #* specifics paths
        self.save_bestloss_model_path = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"])
        self.save_bestloss_model_path_w = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlyweight-B.pt")
        self.save_bestloss_model_path_h = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlyheight-B.pt")
        self.save_bestmae_model_path_w = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlyweight-B-MAE.pt")
        self.save_bestmae_model_path_h = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlyheight-B-MAE.pt")
        
        if init_model_dict == None or not task_config["init_from_checkpoint"]:
            self.best_mse_weight = np.inf
            self.best_mse_height = np.inf
            self.best_final_loss = np.inf
            self.best_mse_onlyw = np.inf
            self.best_mse_onlyh = np.inf
            self.best_mae_onlyw = np.inf
            self.best_mae_onlyh = np.inf
            self.best_epoch_wh = 0
            self.best_epoch_w = 0
            self.best_epoch_h = 0
            self.best_mae_epoch_w = 0
            self.best_mae_epoch_h = 0
        else:
            self.best_mse_weight = init_model_dict["best_loss_weight"]
            self.best_mse_height = init_model_dict["best_loss_height"]
            self.best_final_loss = init_model_dict["best_final_loss"]
            self.best_mse_onlyw = init_model_dict["best_loss_onlyw"]
            self.best_mse_onlyh = init_model_dict["best_loss_onlyh"]
            self.best_mae_onlyw = init_model_dict["best_mae_onlyw"]
            self.best_mae_onlyh = init_model_dict["best_mae_onlyh"]
            self.best_epoch_wh = init_model_dict["best_epoch_wh"]
            self.best_epoch_w = init_model_dict["best_epoch_w"]
            self.best_epoch_h = init_model_dict["best_epoch_h"]
            self.best_mae_epoch_w = init_model_dict["best_mae_epoch_w"]
            self.best_mae_epoch_h = init_model_dict["best_mae_epoch_h"]
        
    #* ##### define model ######
    #//TODO: define a generic structure.
    #*  Define a generic structure will allow that an checkpoint, best or final models would be load in the same function
    def get_model(self, config):
        train_config = config.CONFIG["current_project_info"]["train_infe_config"]
        model_function_name = train_config["model_func"]["use"]
        task_config = config.CONFIG["current_project_info"]["task_config"]
        
        model = models.__dict__[model_function_name]
        model = model(**config.CONFIG)
        
        checkpoint_dict = None
        if task_config["init_from_checkpoint"]["use"]:
            checkpoint = torch.load(task_config["init_from_checkpoint"]["path"])
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_dict = {
                #//TODO: change this
                'current_epoch': checkpoint["current_epoch"],
                # 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': checkpoint["optimizer_state_dict"],
                'best_final_loss': checkpoint["best_final_loss"],
                'best_loss_weight': checkpoint["best_loss_weight"],
                'best_loss_height': checkpoint["best_loss_height"],
                "best_loss_onlyw": checkpoint["best_loss_onlyw"],
                'best_loss_onlyh': checkpoint["best_loss_onlyh"],
                "best_mae_onlyw": checkpoint["best_mae_onlyw"],
                "best_mae_onlyh": checkpoint["best_mae_onlyh"],
                "best_epoch_wh": checkpoint["best_epoch_wh"],
                "best_epoch_w": checkpoint["best_epoch_w"],
                "best_epoch_h": checkpoint["best_epoch_h"],
                # "best_mae_epoch_w": checkpoint["best_mae_epoch_w"],
                # "best_mae_epoch_h": checkpoint["best_mae_epoch_h"]
            }
            
        print('\n[MODEL] Model:')
        print(model)
        
        return model, checkpoint_dict
    
    def get_criterion_config(self):
        
        criterion_config = {}
        if self.target_weight:
            criterion_config["weight"] = {
                "criterion": torch.nn.MSELoss().to(self.device),
                "idx": 0
            }
        if self.target_height:
            criterion_config["height"] = {
                "criterion": torch.nn.MSELoss().to(self.device),
                "idx": 1
            }
        if self.target_bmi:
            criterion_config["bmi"] = {
                "criterion": torch.nn.MSELoss().to(self.device),
                "idx": 2
            }
        
        for key in criterion_config.keys(): criterion_config[key]["name"] = criterion_config[key]["criterion"]._get_name()
        
        return criterion_config
    
class WH_oneout_ConfigExperiment(ConfigExperimentInterface):
    
    def __init__(self, config) -> None:
        train_config = config.CONFIG["current_project_info"]["train_infe_config"]
        task_config = config.CONFIG["current_project_info"]["task_config"]
        deep_config = train_config["deep_config"]
        regression_config = task_config["regression_config"]
        
        #//TODO: implement the other models kind
        model = None
        
        #* get the model. If there is another type, put it here.
        self.use_cuda = train_config["cuda"]["use"] and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        model, init_model_dict = self.get_model(config=config)
        super(WH_oneout_ConfigExperiment, self).__init__(config=config, model=model)
        self.model = model
        
        #* load config if it should initiate from another model/state
        if init_model_dict:
            self.optimizer.load_state_dict(init_model_dict["optimizer_state_dict"])
            
            #//TODO: set the save function for scheduler state_dict too.
            # Set the current learning rate in the scheduler
            self.lr_scheduler._step_count = init_model_dict["current_epoch"] + 1
            self.lr_scheduler.last_epoch = init_model_dict["current_epoch"]
            self.lr_scheduler._last_lr = [self.optimizer.param_groups[0]['lr']]
            
            #* 'testing' is initialized in super class. And if it is 'True', the number of epochs should be actualized to initiate from the checkpoint epoch then run just the epochs defined on testing. 
            if self.testing:
                self.epochs += init_model_dict["current_epoch"] + 1
                
        self.init_epoch = 0 if init_model_dict == None else init_model_dict['current_epoch'] + 1
        assert (init_model_dict == None) or (init_model_dict['current_epoch'] + 1 <= deep_config["epochs"]["use"]), "The initial epoch value is not valid"
        
        #* target - weight or height
        self.target_wh = regression_config["targets"]["use"][0]
        
        #* loss function
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.criterion_config = self.criterion._get_name()
        
        #* specifics paths
        self.save_bestloss_model_path = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"])
        self.save_bestmae_model_path = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "-mae-B.pt")
        
        if init_model_dict == None or not task_config["init_from_checkpoint"]:
            self.best_loss = np.inf
            self.best_mae = np.inf
            self.best_epoch_loss = 0
            self.best_epoch_mae = 0
        
        else:
            self.best_loss = init_model_dict["best_loss"]
            self.best_mae = init_model_dict["best_mae"]
            self.best_epoch_loss = init_model_dict["best_epoch_loss"]
            self.best_epoch_mae = init_model_dict["best_epoch_mae"]
        
    #* ##### define model ######
    #//TODO: define a generic structure.
    #*  Define a generic structure will allow that an checkpoint, best or final models would be load in the same function
    def get_model(self, config):
        train_config = config.CONFIG["current_project_info"]["train_infe_config"]
        model_function_name = train_config["model_func"]["use"]
        task_config = config.CONFIG["current_project_info"]["task_config"]
        
        model = models.__dict__[model_function_name]
        model = model(**config.CONFIG)
        model.to(self.device)
        
        checkpoint_dict = None
        if task_config["init_from_checkpoint"]["use"]:
            checkpoint = torch.load(task_config["init_from_checkpoint"]["path"], map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_dict = {
                #//TODO: change this
                'current_epoch': checkpoint["current_epoch"],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': checkpoint["optimizer_state_dict"],
                'best_loss': checkpoint["best_loss"],
                'best_mae': checkpoint["best_mae"],
                "best_epoch_loss": checkpoint["best_epoch_loss"],
                "best_epoch_mae": checkpoint["best_epoch_mae"]
            }
            
        print('\n[MODEL] Model:')
        print(model)
        
        return model, checkpoint_dict
    
class WH_multimodel_ConfigExperiment(ConfigExperimentInterface):
    
    def __init__(self, config) -> None:
        train_config = config.CONFIG["current_project_info"]["train_infe_config"]
        task_config = config.CONFIG["current_project_info"]["task_config"]
        deep_config = train_config["deep_config"]
        regression_config = task_config["regression_config"]
        
        self.target_weight = True if "weight" in regression_config["targets"]["use"] else False
        self.target_height = True if "height" in regression_config["targets"]["use"] else False 
        self.target_bmi = True if "bmi" in regression_config["targets"]["use"] else False 
        self.num_targets = len(regression_config["targets"]["use"])
        
        #//TODO: implement the other models kind
        model = None
        
        #* get the model. If there is another type, put it here.
        self.use_cuda = train_config["cuda"]["use"] and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        #* get the model. If there is another type, put it here.
        model, init_model_dict = self.get_model(config=config)
        super(WH_multimodel_ConfigExperiment, self).__init__(config=config, model=model)
        self.model = model
        
        #* load config if it should initiate from another model/state
        if init_model_dict:
            self.optimizer.load_state_dict(init_model_dict["optimizer_state_dict"])
            
            #//TODO: set the save function for scheduler state_dict too.
            # Set the current learning rate in the scheduler
            self.lr_scheduler._step_count = init_model_dict["current_epoch"] + 1
            self.lr_scheduler.last_epoch = init_model_dict["current_epoch"]
            self.lr_scheduler._last_lr = [self.optimizer.param_groups[0]['lr']]
            
            #* 'testing' is initialized in super class. And if it is 'True', the number of epochs should be actualized to initiate from the checkpoint epoch then run just the epochs defined on testing. 
            if self.testing:
                self.epochs += init_model_dict["current_epoch"] + 1
        
        self.init_epoch = 0 if init_model_dict == None else init_model_dict['current_epoch'] + 1
        assert (init_model_dict == None) or (init_model_dict['current_epoch'] + 1 <= deep_config["epochs"]["use"]), "The initial epoch value is not valid"
        
        #* loss function
        self.criterion_config = self.get_criterion_config()
        
        #* specifics paths
        if self.target_weight:
            self.save_bestloss_model_path_w = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlyweight-B.pt")
            self.save_bestmae_model_path_w = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlyweight-B-MAE.pt")
        if self.target_height:
            self.save_bestloss_model_path_h = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlyheight-B.pt")
            self.save_bestmae_model_path_h = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlyheight-B-MAE.pt")
        if self.target_bmi:
            self.save_bestloss_model_path_b = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlybmi-B.pt")
            self.save_bestmae_model_path_b = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"][:-5] + "onlybmi-B-MAE.pt")
        
        if self.num_targets == 0:
            raise framework_error("Regression Configuration shoud have at least one target.")
        elif self.num_targets > 1:
            self.save_bestloss_model_path = os.path.join(self.save_model_dir, train_config["best_model_save_name"]["use"])
        
        #* Inicial Values            
        if init_model_dict == None or not task_config["init_from_checkpoint"]:
            if self.target_weight:
                self.best_mse_weight = np.inf
                self.best_mae_weight = np.inf
                self.best_epoch_w = 0
                self.best_mae_epoch_w = 0
            if self.target_height:
                self.best_mse_height = np.inf
                self.best_mae_height = np.inf
                self.best_epoch_h = 0
                self.best_mae_epoch_h = 0
            if self.target_bmi:
                self.best_mse_bmi = np.inf
                self.best_mae_bmi = np.inf
                self.best_epoch_b = 0
                self.best_mae_epoch_b = 0
                
            if self.num_targets > 1:
                self.best_loss = np.inf
                self.best_epoch_loss = 0
                if self.target_weight:
                    self.best_mse_onlyw = np.inf
                    self.best_mae_onlyw = np.inf
                if self.target_height:
                    self.best_mse_onlyh = np.inf
                    self.best_mae_onlyh = np.inf
                if self.target_bmi:
                    self.best_mse_onlyb = np.inf
                    self.best_mae_onlyb = np.inf
            
        else:
            if self.target_weight:
                self.best_mse_weight = init_model_dict["best_loss_weight"]
                self.best_epoch_w = init_model_dict["best_epoch_w"]
                self.best_mae_w = init_model_dict["best_mae_w"]
                self.best_mae_epoch_w = init_model_dict["best_mae_epoch_w"]
            if self.target_height:
                self.best_mse_height = init_model_dict["best_loss_height"]
                self.best_epoch_h = init_model_dict["best_epoch_h"]
                self.best_mae_h = init_model_dict["best_mae_h"]
                self.best_mae_epoch_h = init_model_dict["best_mae_epoch_h"]
            if self.target_bmi:
                self.best_mse_bmi = init_model_dict["best_loss_bmi"]
                self.best_epoch_b = init_model_dict["best_epoch_b"]
                self.best_mae_b = init_model_dict["best_mae_b"]
                self.best_mae_epoch_b = init_model_dict["best_mae_epoch_b"]
                
            if self.num_targets > 1:
                self.best_loss = init_model_dict["best_loss"]
                self.best_epoch_loss = init_model_dict["best_epoch_wh"]
                if self.target_weight:
                    self.best_mse_onlyw = init_model_dict["best_loss_onlyw"]
                    self.best_mae_onlyw = init_model_dict["best_mae_onlyw"]
                if self.target_height:
                    self.best_mse_onlyh = init_model_dict["best_loss_onlyh"]
                    self.best_mae_onlyh = init_model_dict["best_mae_onlyh"]
                if self.target_bmi:
                    self.best_mse_onlyb = init_model_dict["best_loss_onlyb"]
                    self.best_mae_onlyb = init_model_dict["best_mae_onlyb"]
                    
    #* ##### define model ######
    #//TODO: define a generic structure.
    #*  Define a generic structure will allow that an checkpoint, best or final models would be load in the same function
    def get_model(self, config):
        train_config = config.CONFIG["current_project_info"]["train_infe_config"]
        model_function_name = train_config["model_func"]["use"]
        task_config = config.CONFIG["current_project_info"]["task_config"]
        
        model = models.__dict__[model_function_name]
        model = model(**config.CONFIG)
        
        checkpoint_dict = None
        if task_config["init_from_checkpoint"]["use"]:
            checkpoint = torch.load(task_config["init_from_checkpoint"]["path"])
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_dict = {
                #//TODO: change this
                'current_epoch': checkpoint["current_epoch"],
                # 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': checkpoint["optimizer_state_dict"],
                
                'best_loss': checkpoint["best_loss"],
                "best_epoch_wh": checkpoint["best_epoch_wh"],
                'best_loss_weight': checkpoint["best_loss_weight"],
                'best_loss_height': checkpoint["best_loss_height"],
                "best_loss_bmi": checkpoint["best_loss_bmi"],
                
                'best_mae_w': checkpoint["best_mae_w"],
                'best_mae_h': checkpoint["best_mae_h"],
                'best_mae_b': checkpoint["best_mae_b"],
                
                "best_loss_onlyw": checkpoint["best_loss_onlyw"],
                'best_loss_onlyh': checkpoint["best_loss_onlyh"],
                'best_loss_onlyb': checkpoint["best_loss_onlyb"],
                
                "best_epoch_w": checkpoint["best_epoch_w"],
                "best_epoch_h": checkpoint["best_epoch_h"],
                "best_epoch_b": checkpoint["best_epoch_b"],
                
                "best_mae_epoch_w": checkpoint["best_mae_epoch_w"],
                "best_mae_epoch_h": checkpoint["best_mae_epoch_h"],
                "best_mae_epoch_b": checkpoint["best_mae_epoch_b"]
            }
            
        print('\n[MODEL] Model:')
        print(model)
        
        return model, checkpoint_dict
    
    def get_config_paths(self):
        string_return = ""
        
        #* where save models - best and final
        if self.num_targets > 1: 
            string_return.join("Best Loss Model Path: {}".format(self.save_bestloss_model_path))
        if self.target_weight: 
            string_return.join("Best Loss Weight Path: {}".format(self.save_bestloss_model_path_w))
            string_return.join("Best MAE Weight Path: {}".format(self.save_bestmae_model_path_w))
        if self.target_height:
            string_return.join("Best Loss Height Path: {}".format(self.save_bestloss_model_path_h))
            string_return.join("Best MAE Height Path: {}".format(self.save_bestmae_model_path_h))
        if self.target_bmi:
            string_return.join("Best Loss BMI Path: {}".format(self.save_bestloss_model_path_h))
            string_return.join("Best MAE BMI Path: {}".format(self.save_bestmae_model_path_h))
        
        return string_return
    
    def get_config_criterion(self):
        
        string_return = ""
        if self.target_weight: string_return.join("\n".join(self.criterion_config["weight"]._get_name()))
        if self.target_height: string_return.join("\n".join(self.criterion_config["height"]._get_name()))
        if self.target_bmi: string_return.join("\n".join(self.criterion_config["bmi"]._get_name()))
        
        return string_return
    
    def get_config_initValues(self):
        
        string_return = ""
        if self.num_targets > 1:
            string_return.join("\n[TRAIN] Inicial Best MSE Weight and MSE Height (loss): {}".format(self.best_final_loss))
            string_return.join("[TRAIN] Inicial Best Epoch Weight and Height (loss): {}".format(self.best_epoch_wh))
            if self.target_weight:
                string_return.join("[TRAIN] Inicial Best MSE Only-Weight (loss): {}".format(self.best_mse_onlyw))
                string_return.join("[TRAIN] Inicial Best MAE Only-Weight (loss): {}".format(self.best_mae_onlyw))        
            if self.target_height: 
                string_return.join("[TRAIN] Inicial Best MSE Only-Height (loss): {}".format(self.best_mse_onlyh))
                string_return.join("[TRAIN] Inicial Best MAE Only-Height (loss): {}".format(self.best_mae_onlyh))    
            if self.target_bmi:
                string_return.join("[TRAIN] Inicial Best MSE Only-BMI (loss): {}".format(self.best_mse_height))
                string_return.join("[TRAIN] Inicial Best MAE Only-BMI (loss): {}".format(self.best_mae_onlyh))    
        if self.target_weight:
            string_return.join("[TRAIN] Inicial Best MSE Weight: {}".format(self.best_mse_weight))
            string_return.join("[TRAIN] Inicial Best MAE Weight: {}".format(self.best_mse_weight))        
            string_return.join("[TRAIN] Inicial Best MSE Epoch Weight: {}".format(self.best_mae_epoch_w))
            string_return.join("[TRAIN] Inicial Best MAE Epoch Weight: {}".format(self.best_mae_epoch_w))
        if self.target_height: 
            string_return.join("[TRAIN] Inicial Best MSE Height: {}".format(self.best_mse_height))
            string_return.join("[TRAIN] Inicial Best MAE Height: {}".format(self.best_mse_height))    
            string_return.join("[TRAIN] Inicial Best MSE Epoch Height: {}".format(self.best_mae_epoch_h))
            string_return.join("[TRAIN] Inicial Best MAE Epoch Height: {}".format(self.best_mae_epoch_h))
        if self.target_bmi:
            string_return.join("[TRAIN] Inicial Best MSE BMI: {}".format(self.best_mse_bmi))
            string_return.join("[TRAIN] Inicial Best MAE BMI: {}".format(self.best_mse_bmi))    
            string_return.join("[TRAIN] Inicial Best MSE Epoch BMI: {}".format(self.best_mae_epoch_b))
            string_return.join("[TRAIN] Inicial Best MAE Epoch BMI: {}".format(self.best_mae_epoch_b))
            
        return string_return
    
    def get_criterion_config(self):
        
        criterion_config = {}
        if self.target_weight:
            criterion_config["weight"] = {
                "criterion": torch.nn.MSELoss().to(self.device),
                "idx": 0
            }
        if self.target_height:
            criterion_config["height"] = {
                "criterion": torch.nn.MSELoss().to(self.device),
                "idx": 1
            }
        if self.target_bmi:
            criterion_config["bmi"] = {
                "criterion": torch.nn.MSELoss().to(self.device),
                "idx": 2
            }
        
        return criterion_config