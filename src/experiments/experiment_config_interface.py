import torch
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn

from utils.utils_time import get_time_string
from configs.config import Config
from torch.utils.data import DataLoader

class ConfigExperimentInterface():
        
    def __init__(self, config, model) -> None:
        train_config = config.CONFIG["current_project_info"]["train_infe_config"]
        general_info = config.CONFIG["general_info"]
        task_config = config.CONFIG["current_project_info"]["task_config"]
        deep_config = train_config["deep_config"]
        scheduler_config = deep_config["scheduler_config"]
        
        self.model = model
        
        self.optimizer_config_dict = self.get_optimizer_config(deep_config=deep_config)
        self.optimizer = self.get_optimizer(
            optimizer_type=deep_config["optimizer"]["use"], 
            optimizer_config=self.optimizer_config_dict
            )
        
        self.scheduler_config_dict = self.get_scheduler_config(scheduler_config=scheduler_config)
        self.lr_scheduler = self.get_scheduler(scheduler_config_dict=self.scheduler_config_dict)
        
        self.epochs = deep_config["epochs"]["use"]
        self.test_epochs = range(0, self.epochs, deep_config["test_epochs"]["use"])
        # self.test_epochs_print = [x + 1 for x in self.test_epochs]
        
        #* paths
        self.save_model_dir = os.path.join(general_info["save_model_path"]["use"], train_config["save_directory"]["use"])
        self.save_checkpoint_path = os.path.join(self.save_model_dir, task_config["use_checkpoint"]["name"])
        self.upadate_json_path: str = os.path.join(self.save_model_dir, os.path.basename(config.PATH_JSON_CONFIG))
        self.save_final_path = os.path.join(self.save_model_dir, train_config["final_model_save_name"]["use"])
        self.save_pdf_path = os.path.join(self.save_model_dir, get_time_string() + "_report.pdf")
        
        #* device
        #* checking cuda
        self.use_cuda = train_config["cuda"]["use"] and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        #* set model to device if GPU
        self.num_gpus = False
        if self.use_cuda:
            self.num_gpus = torch.cuda.device_count()
            cudnn.enabled = True
            cudnn.benchmark = True
            self.model.cuda()
            if self.num_gpus > 1:
                self.model = torch.nn.DataParallel(self.model, range(self.num_gpus))
        
        #* if it is a testing
        self.testing = task_config["testing_config"]["use"]
        if self.testing:
            self.epochs = task_config["testing_config"]["num_epochs"]["use"]
    
    def get_optimizer_config(self, deep_config):
        optimizer_type = deep_config["optimizer"]["use"]
        
        optimizer_config = {}
        if optimizer_type == deep_config["optimizer"]["choices"][0]: #* adam
            optimizer_config["lr"] = deep_config["lr"]["use"]
            optimizer_config["weight_decay"] = deep_config["decay"]["use"]
            
        elif optimizer_type == deep_config["optimizer"]["choices"][0]: #* sgd
            optimizer_config["lr"] = deep_config["lr"]["use"]
            optimizer_config["momentum"] = deep_config["momentum"]["use"]
            optimizer_config["weight_decay"] = deep_config["decay"]["use"]
            optimizer_config["nesterov"] = deep_config["nesterov"]["use"]
        
        return optimizer_config

    def get_optimizer(self, optimizer_type, optimizer_config):
        #* optimizer for trainable parameters 
        trainable_parameters = filter(lambda x: x.requires_grad, self.model.parameters())  #* filter will return only the trainable parametrs
        optimizer = None
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                trainable_parameters, 
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["weight_decay"]
                )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                trainable_parameters, 
                lr=optimizer_config["lr"], 
                momentum=optimizer_config["momentum"], 
                weight_decay=optimizer_config["weight_decay"], 
                nesterov=optimizer_config["nesterov"]
                )
        
        return optimizer
    
    def get_scheduler_config(self, scheduler_config):
        scheduler_config_dict = {}
        if scheduler_config["use"] == scheduler_config["choices"][0]: #* multisteplr
            scheduler_config_dict["type"] = scheduler_config["choices"][0]
            scheduler_config_dict["milestones"] = scheduler_config[scheduler_config["use"]]["milestones"]["use"]
            scheduler_config_dict["gamma"] = scheduler_config["lr_gamma"]["use"]
            scheduler_config_dict["verbose"] = scheduler_config["verbose"]["use"]
        
        elif scheduler_config_dict["use"] == scheduler_config["choices"][1]: #* reducelronplateu
            scheduler_config_dict["type"] = scheduler_config["choices"][1]
            scheduler_config_dict["mode"] = scheduler_config[scheduler_config["use"]]["mode"]["use"]
            scheduler_config_dict["factor"] = scheduler_config["lr_gamma"]["use"]
            scheduler_config_dict["patience"] = scheduler_config[scheduler_config["use"]]["patience"]["use"] 
            scheduler_config_dict["verbose"] = scheduler_config["verbose"]["use"]
        
        return scheduler_config_dict
    
    def get_scheduler(self, scheduler_config_dict):
        lr_scheduler = None
        if scheduler_config_dict["type"] == "multisteplr": #* multisteplr
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=scheduler_config_dict["milestones"], 
                gamma=scheduler_config_dict["gamma"],
                verbose=scheduler_config_dict["verbose"]
                )
        elif scheduler_config_dict["type"] == "reducelronplateu": #* reducelronplateu
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode=scheduler_config_dict["mode"], 
                factor=scheduler_config_dict["factor"], 
                patience=scheduler_config_dict["patience"],
                verbose=scheduler_config_dict["verbose"]
                )
        
        return lr_scheduler