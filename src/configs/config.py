from about_models import models
from beeprint import pp
from functools import reduce
from utils.util_dictionary import get_value_key
from utils.utils_json import saveorupdate_json

import json
import datetime as dt
import os

class Config(object):
    
    def get_model_names(self):
        return sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
    
    def fancy_print_dict(self, d: dict):
        pp(d, sort_keys=False) #The width argument specifies the maximum width of the output before wrapping to a new line (here, we use 80 characters).
    
    def update_json(): ## atualizar só no final???
        pass
    
    def check_config(self, dictionary: dict):
        
        now = dt.datetime.now()
        all_models_list = self.get_model_names()
        skinreader_models_list = [name for name in all_models_list if "skinreader" in name]
        wh_models_list = [name for name in all_models_list if "wh" in name]
        update = dictionary["file_info"]["update"]["use"] # true or false
        
        for k_1, v_1 in dictionary.items():
            if k_1 == "current_project_info":
                for k_2, v_2 in v_1.items():
                        
                    if k_2 == "command":
                        str_error = "'command' should be provided. Possible commands: {}. [HELP MESSAGE] {}".format(v_2["choices"], v_2["help"])
                        assert v_2["use"] in v_2["choices"], str_error
                        
                    if k_2 == "task_config":
                        for k_3, v_3 in v_2.items():
                            if k_3 == "pred_type":    
                                if not v_1["command"]["use"] == v_1["command"]["choices"][2]: #show_image
                                    str_error = "'{}', if provided, should be among: '{}'. [HELP MESSAGE]: {}.".format(k_3, v_3["choices"], v_3["help"])
                                    if v_3["use"] == "default":
                                        v_3["use"] = v_3["default"]
                                    assert v_3["use"] in v_3["choices"], str_error
                                else:
                                    str_error = "if 'command' is 'show_image' pred_type should satay 'false'."
                                    assert not v_3["use"], str_error
                            
                            if k_3 == "regression_config":
                                if v_2["pred_type"]["use"] == v_2["pred_type"]["choices"][1]:    
                                    str_error = "'{}' should be provided, and in the format list. [HELP MESSAGE]: {}.".format(k_3, v_3["help"])
                                    assert len(v_3["targets"]["use"]) > 0, str_error
                                else:
                                    v_3["targets"]["use"] = None
                            
                            if k_3 == "classification_config":
                                if v_2["pred_type"]["use"] == v_2["pred_type"]["choices"][0]: #* classification
                                    if v_3["targets"]["use"] != None:    
                                        str_error = "'{}' should be provided, and in the format list. [HELP MESSAGE]: {}.".format(k_3, v_3["help"])
                                        assert len(v_3["targets"]["use"]) > 0, str_error
                                else:
                                    v_3["targets"]["use"] = None
                            
                            # TODO: Check the update for this field. "tags"
                            if k_3 == "tags":
                                if v_3["use"]:
                                    general_tags = get_value_key(dictionary=dictionary, key_list=v_3["other_key"])
                                    for tag in v_3["use"]:
                                        if tag not in general_tags:
                                            general_tags.append(tag)
                                else: 
                                    v_3["use"] = []
                            
                            if k_3 == "use_checkpoint":
                                if v_3["use"]:
                                    if not v_3["name"]:
                                        v_3["name"] = "{}_{}-{}-{}-{}-checkpoint.pt".format(v_1["about_current_project"]["name"], now.year, now.month, now.day, now.hour)
                                    else:
                                        name_aux = "_{}-{}-{}-{}-checkpoint".format(now.year, now.month, now.day, now.hour)
                                        v_3["name"] = v_3["name"][:-3] + name_aux + v_3["name"][-3:]
                            
                            if k_3 == "init_from_checkpoint":
                                if v_3["use"]:
                                    str_error = "If '{}' is set to use, it should be provided. [HELP MESSAGE]: {}.".format(k_3, v_3["help"])
                                    assert v_3["path"], str_error
                                    str_error = "'{}' doesn't exist, it should be provided and should be a valid and existent file path. [HELP MESSAGE]: {}.".format(k_3, v_3["help"])
                                    assert os.path.exists(v_3["path"]), str_error
                                    
                            if k_3 == "testing_config":
                                if v_3["use"] != None or v_3["use"]: 
                                    str_error = "'{}' shold be bool type. [HELP MESSAGE]: {}.".format(k_3, v_3["help"])
                                    assert isinstance(v_3["use"], bool), str_error
                                    
                                    if v_3["use"]:
                                        for k_4, v_4 in v_3.items():    
                                            if k_4 == "num_epochs":
                                                str_error = "'num_epochs' must be greater than 0, and integer type."
                                                if v_4["use"] == "default":
                                                    v_4["use"] = v_4["default"]
                                                assert isinstance(v_4["use"], int) and v_4["use"] > 0, str_error 
                                            
                                            if k_4 == "num_elements":
                                                str_error = "'num_elements' must be greater than 0, and integer type."
                                                if v_4["use"] == "default":
                                                    v_4["use"] = v_4["default"]
                                                assert isinstance(v_4["use"], int) and v_4["use"] > 0, str_error 
                                        
                    if k_2 == "data_config":
                        for k_3, v_3 in v_2.items():
                            if k_3 == "data_file_name":
                                str_error = "'{}' should be provided. [HELP MESSAGE]: {} data_path = {}".format(k_3, v_3["help"], dictionary["general_info"]["data_path"]["default"])
                                assert v_3["use"] and len(v_3["use"]) > 0, str_error
                                
                            if k_3 == "file_type":
                                str_error = "'{}' should be one of the types: {}. The default value is {}.".format(k_3, v_3["choices"], v_3["default"])
                                if v_3["use"] == "default":
                                    v_3["use"] = v_3["default"]
                                assert v_3["use"] in v_3["choices"] , str_error
                                
                            if k_3 == "split_data_type":
                                str_error = "'{}' should be one of the types: '{}'. [HELP MESSAGE]: {}.".format(k_3, v_3["choices"], v_3["help"])
                                if v_3["use"] == "default":
                                    if v_1["command"]["use"] == v_1["command"]["choices"][1]: #inference
                                        v_3["use"] = v_3["choices"][3]
                                        assert v_3["use"] in v_3["choices"] , str_error
                                    if v_1["command"]["use"] == v_1["command"]["choices"][0]: #train
                                        v_3["use"] = v_3["choices"][1]
                                        assert v_3["use"] in v_3["choices"] , str_error
                                    if v_1["command"]["use"] == v_1["command"]["choices"][2]: #show_image
                                        assert not v_3["use"], str_error
                                
                            if k_3 == "num_workers":
                                str_error = "'{}' should be more than 0. The default value is '{}'.".format(k_3, v_3["default"])
                                if not v_3["use"]:
                                    v_3["use"] = v_3["default"]
                                assert v_3["use"] > 0 , str_error
                                
                            if k_3 == "train_size":
                                if v_3["use"] == "default":
                                    if v_2["split_data_type"]["use"] == v_2["split_data_type"]["choices"][0]:
                                        v_3["use"] = 1.0
                                    if v_2["split_data_type"]["use"] in v_2["split_data_type"]["choices"][1:-2]:
                                        v_3["use"] = 0.8
                                        
                            if k_3 == "val_size":
                                if not v_3["use"]:
                                    if v_2["split_data_type"]["use"] == v_2["split_data_type"]["choices"][2]:
                                        v_3["use"] = 0.2
                                        
                            if k_3 == "infe_size":
                                if not v_3["use"]:
                                    if v_2["split_data_type"]["use"] == v_2["split_data_type"]["choices"][3]:
                                        v_3["use"] = 0.2
                                        
                            if k_3 == "num_folds":
                                if not v_3["use"]:
                                    if v_2["split_data_type"]["use"] == v_2["split_data_type"]["choices"][4]:
                                        v_3["use"] = v_3["default"]
                            
                            if k_3 == "image_channels":
                                pass
                            
                            if k_3 == "extra_scaling":
                                str_error = "'{}' should be the bool type. [HELP MESSAGE]: '{}'.".format(k_3, v_3["help"])
                                assert isinstance(v_3["use"], bool), str_error
                            
                            if k_3 == "data_batch_size":
                                if v_3["use"] == "default":
                                    v_3["use"] = v_3["default"]
                                str_error = "The '{}' should be greater than 0. [HELP MESSAGE]: {}.".format(k_3, v_3["help"])
                                assert v_3["use"] > 0, str_error
                            
                            if k_3 == "show_images_config":
                                if v_1["command"]["use"] == v_1["command"]["choices"][2]: #* show_images
                                    for k_4, v_4 in v_3.items():
                                        if k_4 == "num_images_to_show":
                                            pass
                                
                            if k_3 == "transform_config":
                                for k_4, v_4 in v_3.items():
                                    if k_4 == "resize":
                                        if v_4["use"]:
                                            str_error = "'{}', if provided, the length should be equal to '2' - [weight, height]. [HELP MESSAGE]: '{}'.".format(k_4, v_4["help"])
                                            assert len(v_4["use"]) == 2 , str_error
                                    
                                    if k_4 == "totensor":
                                        str_error = "'{}' should be the bool type.".format(k_4)
                                        assert isinstance(v_4["use"], bool), str_error
                                        
                                    if k_4 == "grayscale":
                                        str_error = "'{}' should be the bool type. [HELP MESSAGE]: '{}'.".format(k_4, v_4["help"])
                                        assert isinstance(v_4["use"], bool), str_error
                                    
                                    if k_4 == "normalize_input":
                                        str_error = "'{}' should be the bool type or null. [HELP MESSAGE]: '{}'.".format(k_4, v_4["help"])
                                        assert isinstance(v_4["use"], bool) or v_4["use"] == None, str_error
                                    
                                    if k_4 == "normalize_output":
                                        str_error = "'{}' should be the bool type or null. [HELP MESSAGE]: '{}'.".format(k_4, v_4["help"])
                                        assert isinstance(v_4["use"], bool) or v_4["use"] == None, str_error
                                        
                                        if v_4["use"]:
                                            str_error = "'to_norm' should has at least one dependent variable to normalize."
                                            assert len(v_4["to_norm"]) > 0, str_error
                                    
                                    if k_4 == "augment":
                                        if v_4["use"]:
                                            if v_4["name"] == "random":
                                                for k_5, v_5 in v_4.items():
                                                    if k_5 == "random_vertical_flip":
                                                        pass
                                                        # str_error = "'{}' should be the bool type.".format(k_5)
                                                        # assert isinstance(v_5["random_vertical_flip"], bool), str_error                   
                                                    
                                                    if k_5 == "random_horizontal_flip":
                                                        pass
                                                        # str_error = "'{}' should be the bool type.".format(k_5)
                                                        # assert isinstance(v_5["random_vertical_flip"], bool), str_error
                                                    
                                                    if k_5 == "random_rotation":
                                                        if v_5["use"]:
                                                            str_error = "If '{}' will be used, 'degrees' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["degrees"], str_error
                                                    
                                                    if k_5 == "random_crop":
                                                        if v_5["use"]:
                                                            str_error = "If '{}' will be used, 'size' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["size"] and isinstance(v_5["size"], int), str_error
                                                            str_error = "If '{}' will be used, 'pad_if_needed' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["pad_if_needed"] == True or v_5["pad_if_needed"] == None, str_error
                                                            str_error = "If '{}' will be used, 'padding_mode' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["padding_mode"] and (isinstance(v_5["padding_mode"], str) or isinstance(v_5["padding_mode"], str)), str_error
                                                    
                                                    if k_5 == "random_gaussian_blur":
                                                        if v_5["use"]:
                                                            str_error = "If '{}' will be used, 'kernel_size' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["kernel_size"], str_error
                                                            str_error = "If '{}' will be used, 'sigma' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["sigma"], str_error
                                                            
                                                    if k_5 == "random_color_jitter":
                                                        if v_5["use"]:
                                                            str_error = "If '{}' will be used, 'brightness' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["brightness"], str_error
                                                            str_error = "If '{}' will be used, 'contrast' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["contrast"], str_error
                                                            str_error = "If '{}' will be used, 'saturation' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["saturation"], str_error
                                                            str_error = "If '{}' will be used, 'hue' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["hue"], str_error
                                                    
                                                    if k_5 == "random_affine":
                                                        if v_5["use"]:
                                                            str_error = "If '{}' will be used, 'degrees' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["degrees"], str_error
                                                            str_error = "If '{}' will be used, 'translate' should be provided and bool type. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert isinstance(v_5["translate"], bool) , str_error
                                                            str_error = "If '{}' will be used, 'scale' should be provided and bool type. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert isinstance(v_5["scale"], bool), str_error
                                                            str_error = "If '{}' will be used, 'shear' should be provided. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert v_5["shear"], str_error
                                                            str_error = "If '{}' will be used, 'center' should be provided and bool type. [HEL MESSAGE]: {}".format(k_5, v_5["help"])
                                                            assert isinstance(v_5["center"], bool), str_error
                                                    
                                                    if k_5 == "random_zoom_out":
                                                        pass
                                                    
                                                    if k_5 == "coarse_drop":
                                                        pass
                            
                            if k_3 == "isic_config":
                                if v_3["use"] not in [None, False]:
                                    flag_check: bool = False
                                    for k_4, v_4 in v_3.items():
                                        if k_4 == "downsampler":
                                            str_error = "'{}' should be bool type. 'false' if not used and 'true' if should be used. [HELP MESSAGE]: {}".format(k_4, v_4["help"])
                                            assert isinstance(v_4["use"], bool) or v_4["use"] == None, str_error
                                            flag_check = True
                                            
                                        if k_4 == "oversample_to_balance":
                                            str_error = "'{}' should be bool type. 'false' if not used and 'true' if should be used. [HELP MESSAGE]: {}".format(k_4, v_4["help"])
                                            assert isinstance(v_4["use"], bool)  or v_4["use"] == None, str_error
                                            flag_check = True
                                            
                                        if k_4 == "min_number_per_class":
                                            if v_4["use"]:
                                                str_error = "'{}' should be 'int' type and greater than '1'. [HELP MESSAGE]: {}".format(k_4, v_4["help"])
                                                assert (isinstance(v_4["min"], int) and v_4["min"] >= 1)  or v_4["use"] == None, str_error
                                                flag_check = True
                                                
                                    str_error = "If '{}' is used, at least one configuration ({}) should be used too. Otherwise change 'use' fro 'isi_config' to 'false' or 'null'".format(k_3, v_3.keys())    
                                    assert flag_check, str_error
                                        
                            if k_3 == "leprosy_config":
                                if v_3["use"]:
                                    if v_3["augmentation"]["use"]:
                                        for k_4, v_4 in v_3['augmentation'].items():
                                            if k_4 == "oversample_minority_class":
                                                if v_4["use"]:
                                                    str_error = " if '{}' is required, the 'times' parameter should be provided.".format(k_4)
                                                    assert v_4["use"] and v_4["times"], str_error
                            
                            if k_3 == "leprosy_fusion_config":
                                if v_3["use"]:
                                    if v_3["augmentation"]["use"]:
                                        for k_4, v_4 in v_3['augmentation'].items():
                                            if k_4 == "oversample_minority_class":
                                                if v_4["use"]:
                                                    str_error = " if '{}' is required, the 'times' parameter should be provided.".format(k_4)
                                                    assert v_4["use"] and v_4["times"], str_error
                                if v_3["metadata_features"]["use"]:
                                    pass
                            
                            if k_3 == "hansenXisic_config":
                                if v_3["use"] not in [None, False]:
                                    pass
                                
                            if k_3 == "silhouettes_gray_config":
                                pass            
                    if k_2 == "train_infe_config":
                        for k_3, v_3 in v_2.items():
                            if k_3 == "model_func":
                                all_models_name = get_value_key(dictionary=dictionary, key_list=v_3["other_key"])
                                if len(all_models_name) < len(all_models_list) or update or v_3["use"] not in all_models_name:
                                    all_models_name = all_models_list
                                str_error = "The '{}' should be among: '{}'. [HELP MESSAGE]: {}.".format(k_3, v_3["choices"], v_3["help"])
                                if v_1["command"]["use"] == v_1["command"]["choices"][2]:
                                    assert not v_3["use"], str_error
                                else:
                                    assert v_3["use"] in all_models_name, str_error
                                
                            if k_3 == "init_from_model":
                                if v_1["command"]["use"] == v_1["command"]["choices"][1]: #* inference
                                    str_error = "if 'command' is 'inference' the '{}' should be provided. [HELP MESSAGE]: {}.".format(k_3, v_3["help"])
                                    assert v_3["use"] and len(v_3["use"]) > 0, str_error
                                if not v_3["use"]:
                                    v_3["use"] = ""
                                    
                            if k_3 == "save_directory":
                                if v_1["command"]["use"] == "train":
                                    if v_3["use"] == "default":
                                        v_3["use"] = v_1["about_current_project"]["name"]
                                    model_dir = os.path.join(dictionary["general_info"]["save_model_path"]["use"], v_3["use"])
                                    if not os.path.exists(model_dir):
                                        os.makedirs(model_dir)
                                    str_error = "'{}' is not string OR doesn't exist, it should be provided and should be a valid and existent file path. [HELP MESSAGE]: {}".format(k_3, v_3["help"])
                                    assert isinstance(v_3["use"], str) and len(v_3["use"])  > 0 and os.path.exists(model_dir), str_error
                                    
                            if k_3 == "best_model_save_name":    
                                if v_1["command"]["use"] == "train":
                                    if v_3["use"] == "default":
                                        v_3["use"] = "{}_{}-{}-{}-{}-B.pt".format(v_1["about_current_project"]["name"], now.year, now.month, now.day, now.hour)
                            
                            if k_3 == "final_model_save_name":    
                                if v_1["command"]["use"] == "train":
                                    if v_3["use"] == "default":
                                        v_3["use"] = "{}_{}-{}-{}-{}-F.pt".format(v_1["about_current_project"]["name"], now.year, now.month, now.day, now.hour)
                            
                            if k_3 == "cuda":
                                str_error = "The '{}' should be bool type. [HELP MESSAGE]: {}".format(k_3, v_3["help"])
                                assert isinstance(v_3["use"], bool), str_error
                            
                            if k_3 == "seed":
                                if v_3["use"] == "default":
                                    v_3["use"] = v_3["default"]
                                
                            if k_3 == "machine_config":
                                pass
                            
                            if k_3 == "deep_config":
                                for k_4, v_4 in v_3.items():
                                    if k_4 == "batch_size":
                                        if v_4["use"] == "default":
                                            v_4["use"] = v_4["default"]
                                            
                                        str_error = "The '{}' should be greater than 0. [HELP MESSAGE]: {}.".format(k_4, v_4["help"])
                                        assert v_4["use"] > 0, str_error
                                
                                    if k_4 == "epochs":
                                        if v_1["command"]["use"] == "train":
                                            if v_4["use"] == "default":
                                                v_4["use"] = v_4["default"]        
                                            
                                            str_error = "The '{}' should not be less or equal to 0. [HELP MESSAGE]: {}".format(k_4, v_4["help"])
                                            assert v_4["use"] > 0, str_error
                                    
                                    if k_4 == "optimizer":
                                        if v_1["command"]["use"] == "train":
                                            if v_4["use"] == "default":
                                                v_4["use"] = v_4["default"]
                                            str_error = "The '{}' should not be among the optimizers: {}. [HELP MESSAGE]: {}".format(k_4, v_4["choices"], v_4["help"])
                                            assert v_4["use"] and v_4["use"] in v_4["choices"], str_error
                                    
                                    if k_4 == "momentum":
                                        if v_1["command"]["use"] == "train" and v_3["optimizer"] == "sgd": # sgd
                                            if not v_4["use"]:
                                                v_4["use"] = v_4["default"]
                                        
                                    if k_4 == "nesterov":
                                        if v_1["command"]["use"] == "train" and v_3["optimizer"] == "sgd": # sgd
                                            pass
                                
                                    if k_4 == "decay":
                                        if v_1["command"]["use"] == "train": # sgd or adam
                                            if v_4["use"] == "default":
                                                v_4["use"] = v_4["default"]
                                    
                                    if k_4 == "lr":
                                        if v_1["command"]["use"] == "train": # sgd or adam
                                            if v_4["use"] == "default":
                                                v_4["use"] = v_4["default"]
                                    if k_4 == "dropout":
                                        pass
                                            
                                    if k_4 == "test_epochs":
                                        if v_1["command"]["use"] == "train":
                                            if v_4["use"] == "default":
                                                v_4["use"] = v_4["default"]
                                            str_error = "The '{}' should not be a valid int number, bigger than zero. [HELP MESSAGE]: {}".format(k_4, v_4["help"])
                                            assert v_4["use"], str_error
                                            
                                    if k_4 == "scheduler_config":
                                        if v_4["use"]:
                                            str_error = "'{}', if provided, should be among: '{}'. [HELP MESSAGE]: {}.".format(k_4, v_4["choices"], v_4["help"])
                                            assert v_4["use"] in v_4["choices"], str_error
                                            
                                            if v_4["lr_gamma"]["use"] == "default":
                                                v_4["lr_gamma"]["use"] = v_4["lr_gamma"]["default"]
                                            
                                            if v_4["use"] == v_4["choices"][0]: #* multisteplr
                                                for k_5, v_5 in v_4[v_4["choices"][0]].items():
                                                    if k_5 == "milestones":
                                                        if not v_5["use"]:
                                                            v_5["use"] = v_5["default"]
                                            
                                            if v_4["verbose"]["use"]:
                                                pass    
                                            
                                            if v_4["use"] == v_4["choices"][1]: #* reducelronplateu
                                                for k_5, v_5 in v_4[v_4["choices"][1]].items():
                                                    if k_5 == "mode":
                                                        str_error = "'{}' should be provided if {} is {}. [HELP MESSAGE]: {}.".format(k_5, k_4, v_4["choices"][1], v_5["help"])
                                                        assert v_5["use"], str_error
                                                            
                                                    if k_5 == "patience":
                                                        str_error = "'{}' should be provided if {} is {}. [HELP MESSAGE]: {}.".format(k_5, k_4, v_4["choices"][1], v_5["help"])
                                                        assert v_5["use"], str_error
    
            if k_1 == "general_info":
                for k_2, v_2 in v_1.items():
                    if k_2 == "data_path":
                        if not v_2["use"]:
                            v_2["use"] = os.path.join(os.path.curdir, v_2["default"]) 
                        str_error = "The '{}' is required. [HELP MESSAGE]: {}".format(k_2, v_2["help"])
                        assert v_2["use"] and len(v_2["use"]) > 0, str_error
                    
                    if k_2 == "save_model_path":
                        if not v_2["use"]:
                            v_2["use"] = os.path.join(os.path.curdir, v_2["default"])
                        str_error = "The '{}' is required. [HELP MESSAGE]: {}".format(k_2, v_2["help"])
                        assert v_2["use"] and len(v_2["use"]) > 0, str_error
                        
                    if k_2 == "models_names":
                        for k_3, v_3 in v_2.items():
                            if k_3 == "all_models_names":
                                if len(v_3["use"]) < len(all_models_list) or update:
                                    v_3["use"] = all_models_list
                                if len(v_3["default"]) < len(all_models_list) or update:
                                    v_3["default"] = all_models_list
                                    
                            if k_3 == "skinreader_models":
                                if len(v_3["use"]) < len(skinreader_models_list) or update:
                                    v_3["use"] = skinreader_models_list
                                if len(v_3["default"]) < len(skinreader_models_list)or update:
                                    v_3["default"] = skinreader_models_list
                            
                            if k_3 == "wh_models":
                                if len(v_3["use"]) < len(wh_models_list) or update:
                                    v_3["use"] = wh_models_list
                                if len(v_3["default"]) < len(wh_models_list) or update:
                                    v_3["default"] = wh_models_list
        return True
    
    def print_project_config(self, dictionary: dict):
        
        regression_targets = {
            "targets": dictionary["current_project_info"]["task_config"]["regression_config"]["targets"]["use"]
        }
        
        classification_targets = {
            "targets": dictionary["current_project_info"]["task_config"]["classification_config"]["targets"]["use"]
        }
        
        testing_config = {
            "num_epochs": dictionary["current_project_info"]["task_config"]["testing_config"]["num_epochs"]["use"],
            "num_elements": dictionary["current_project_info"]["task_config"]["testing_config"]["num_elements"]["use"]
        }
        
        task_config = {
            "pred_type":  dictionary["current_project_info"]["task_config"]["pred_type"]["use"],
            "regression_config": regression_targets,
            "classification_config": classification_targets,
            "tags": dictionary["current_project_info"]["task_config"]["tags"]["use"],
            "use_checkpoint": dictionary["current_project_info"]["task_config"]["use_checkpoint"]["name"],
            "init_from_checkpoint": dictionary["current_project_info"]["task_config"]["init_from_checkpoint"]["path"],
            "testing": testing_config
        }
        
        random_affine = {
            "use": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_affine"]["use"],
            "degrees": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_affine"]["degrees"],
            "translate": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_affine"]["translate"],
            "scale": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_affine"]["scale"],
            "shear": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_affine"]["shear"],
            "center": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_affine"]["center"]
        }
        
        random_zoom_out = {
            "probability": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_zoom_out"]["probability"]
        }
        
        coarse_drop = {
            "max_holes": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["coarse_drop"]["max_holes"],
            "max_height": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["coarse_drop"]["max_height"],
            "max_width": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["coarse_drop"]["max_width"],
            "probability": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["coarse_drop"]["probability"]
        }
        
        random_color_jitter = {
            "use": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_color_jitter"]["use"],
            "brightness": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_color_jitter"]["brightness"],
            "contrast": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_color_jitter"]["contrast"],
            "saturation": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_color_jitter"]["saturation"],
            "hue": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_color_jitter"]["hue"]
        }        
        
        random_gaussian_blur = {
            "use": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_gaussian_blur"]["use"],
            "kernel_size": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_gaussian_blur"]["kernel_size"],
            "sigma": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_gaussian_blur"]["sigma"]
        }
        
        random_crop = {
            "use": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_crop"]["use"],
            "size": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_crop"]["size"],
            "pad_if_needed": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_crop"]["pad_if_needed"],
            "padding_mode": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_crop"]["padding_mode"],
        }
        
        random_rotation = {
            "use": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_rotation"]["use"],
            "degrees": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_rotation"]["degrees"]
        }
        
        random_config = {
            "random_vertical_flip": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_vertical_flip"],
            "random_horizontal_flip": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["random_config"]["random_horizontal_flip"],
            "random_rotation": random_rotation,
            "random_crop": random_crop,
            "random_gaussian_blur": random_gaussian_blur,
            "random_color_jitter": random_color_jitter,
            "random_affine": random_affine,
            "random_zoom_out": random_zoom_out,
            "random_coarse_drop": coarse_drop
        }
        
        augment = {
            "use": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["use"],
            "name": dictionary["current_project_info"]["data_config"]["transform_config"]["augment"]["name"],
            "random_config": random_config
        }
        
        show_images_config = {
            "num_images_to_show": dictionary["current_project_info"]["data_config"]["show_images_config"]["num_images_to_show"]["use"]
        }        
        
        normalize_output = {
            "use": dictionary["current_project_info"]["data_config"]["transform_config"]["normalize_output"]["use"],
            "to_norm": dictionary["current_project_info"]["data_config"]["transform_config"]["normalize_output"]["to_norm"]
        }
        
        transform_config = {
            "resize":  dictionary["current_project_info"]["data_config"]["transform_config"]["resize"]["use"],
            "totensor":  dictionary["current_project_info"]["data_config"]["transform_config"]["totensor"]["use"],
            "grayscale":  dictionary["current_project_info"]["data_config"]["transform_config"]["grayscale"]["use"],
            "normalize_input":  dictionary["current_project_info"]["data_config"]["transform_config"]["normalize_input"]["use"],
            "normalize_output":  normalize_output,
            "augment": augment
        }
        
        min_number_per_class = {
            "use": dictionary["current_project_info"]["data_config"]["isic_config"]["min_number_per_class"]["use"],
            "min": dictionary["current_project_info"]["data_config"]["isic_config"]["min_number_per_class"]["min"]
        }
        
        isic_config = {
            "downsample": dictionary["current_project_info"]["data_config"]["isic_config"]["downsample"]["use"],
            "oversample_to_balance": dictionary["current_project_info"]["data_config"]["isic_config"]["oversample_to_balance"]["use"],
            "min_number_per_class": min_number_per_class
        }
        
        #* leprosy_config
        oversample_minority_class = {
            "use": dictionary["current_project_info"]["data_config"]["leprosy_config"]["augmentation"]["oversample_minority_class"]["use"],
            "times": dictionary["current_project_info"]["data_config"]["leprosy_config"]["augmentation"]["oversample_minority_class"]["times"]
        }
        leprosy_config = {
            "augmentation": {
                "use": dictionary["current_project_info"]["data_config"]["leprosy_config"]["augmentation"]["use"],
                "oversample_minority_class": oversample_minority_class
            }
        }
        
        #* leprosy_fusion_config
        oversample_minority_class = {
            "use": dictionary["current_project_info"]["data_config"]["leprosy_fusion_config"]["augmentation"]["oversample_minority_class"]["use"],
            "times": dictionary["current_project_info"]["data_config"]["leprosy_fusion_config"]["augmentation"]["oversample_minority_class"]["times"]
        }
        metadata_features = {
            "use": dictionary["current_project_info"]["data_config"]["leprosy_fusion_config"]["metadata_features"]["use"],
            "features_list": dictionary["current_project_info"]["data_config"]["leprosy_fusion_config"]["metadata_features"]["features_list"],
            "drop_null": dictionary["current_project_info"]["data_config"]["leprosy_fusion_config"]["metadata_features"]["drop_null"]   
        }
        leprosy_fusion_config = {
            "augmentation": {
                "use": dictionary["current_project_info"]["data_config"]["leprosy_fusion_config"]["augmentation"]["use"],
                "oversample_minority_class": oversample_minority_class
            },
            "metadata_features": metadata_features
        }
        
        #* data_config
        data_config = {
            "data_file_name": dictionary["current_project_info"]["data_config"]["data_file_name"]["use"],
            "file_type": dictionary["current_project_info"]["data_config"]["file_type"]["use"],
            "split_data_type": dictionary["current_project_info"]["data_config"]["split_data_type"]["use"],
            "num_workers": dictionary["current_project_info"]["data_config"]["num_workers"]["use"],
            "train_size": dictionary["current_project_info"]["data_config"]["train_size"]["use"],
            "val_size": dictionary["current_project_info"]["data_config"]["val_size"]["use"],
            "infe_size": dictionary["current_project_info"]["data_config"]["infe_size"]["use"],
            "num_folds": dictionary["current_project_info"]["data_config"]["num_folds"]["use"],
            "data_batch_size": dictionary["current_project_info"]["data_config"]["data_batch_size"]["use"],
            "show_images_config": show_images_config,
            "tranform_config": transform_config,
            "isic_config": isic_config,
            "leprosy_config": leprosy_config,
            "leprosy_fusion_config": leprosy_fusion_config
        }
        
        machine_config = {
            
        }
        
        multisteplr = {
            "milestones": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["scheduler_config"]["multisteplr"]["milestones"]["use"]
        }
        reducelronplateu = {
            "mode": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["scheduler_config"]["reducelronplateu"]["mode"]["use"],
            "patience": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["scheduler_config"]["reducelronplateu"]["patience"]["use"]
        }
        scheduler_config = {
            "use": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["scheduler_config"]["use"],
            "lr_gamma": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["scheduler_config"]["lr_gamma"],
            "multisteplr": multisteplr,
            "reducelronplateu": reducelronplateu
        }
        
        deep_config = {
            "batch_size": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["batch_size"]["use"],
            "epochs": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["epochs"]["use"],
            "optimizer": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["optimizer"]["use"],
            "momentum": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["momentum"]["use"],
            "nesterov": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["nesterov"]["use"],
            "decay": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["decay"]["use"],
            "lr": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["lr"]["use"],
            "scheduler_config": scheduler_config,
            "dropout": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["dropout"]["use"],
            "test_epochs": dictionary["current_project_info"]["train_infe_config"]["deep_config"]["test_epochs"]["use"],
        }
        
        model_func = {
            "function": dictionary["current_project_info"]["train_infe_config"]["model_func"]["use"],
            'params': dictionary["current_project_info"]["train_infe_config"]["model_func"]["params"]
        }
                
        train_infe_config = {
            "model_func": model_func,
            "init_from_model": dictionary["current_project_info"]["train_infe_config"]["init_from_model"]["use"],
            "save_directory": dictionary["current_project_info"]["train_infe_config"]["save_directory"]["use"],
            "best_model_save_name": dictionary["current_project_info"]["train_infe_config"]["best_model_save_name"]["use"],
            "final_model_save_name": dictionary["current_project_info"]["train_infe_config"]["final_model_save_name"]["use"],
            "cuda": dictionary["current_project_info"]["train_infe_config"]["cuda"]["use"],        
            "seed": dictionary["current_project_info"]["train_infe_config"]["seed"]["use"],
            "machine_config": machine_config,
            "deep_config": deep_config
        }
        
        about_current_project = {
            "use": dictionary["current_project_info"]["about_current_project"]["use"],
            "name": dictionary["current_project_info"]["about_current_project"]["name"]
        }
        
        d_of_interest = {
            "current_project_info": {
                "about_current_project": about_current_project,
                "command": dictionary["current_project_info"]["command"]["use"],
                "pred_config": task_config,
                "data_config": data_config,
                "train_infe_config": train_infe_config,
                "results": {}
            }
        }
        
        self.fancy_print_dict(d_of_interest)
        
        return d_of_interest
        
    def filter_dict(self, dictionary, keys_to_remove):
        return_dict = {}
        if isinstance(keys_to_remove, list):
            for k_1, v_1 in dictionary.items():
                if k_1 not in keys_to_remove:
                    return_dict[k_1] = v_1
        
        return return_dict
        
    def create_jsonfile(self, args):
        base_config_path = "./config_bases"
        files = os.listdir(base_config_path)
        
        keys_not_required = {}
        dictionary_default = {}
        for file in files:
            if files.endswith(args.function):
                dictionary_default = json.load(os.path.join(base_config_path, file))
                
                if args.function in dictionary_default["current_project_info"]["general_info"]["skinreader_models"]["use"]:
                    for k_1, v_1 in dictionary_default["current_project_info"]["general_info"]["skinreader_models"]["function_default_config"].items():
                        if args.function == k_1:
                            keys_not_required = dictionary_default["current_project_info"]["general_info"]["skinreader_models"]["function_default_config"][k_1]
                        
                        if len(keys_not_required.keys() > 0): break
            if len(keys_not_required.keys() > 0): break
        
        current_project_info =  {}
        for k_1, v_1 in dictionary_default["current_project_info"].items():
            if k_1 in keys_not_required.keys():
                if not keys_not_required[k_1] == "all":
                    current_project_info[k_1] = self.filter_dict(dictionary_default[k_1], keys_not_required[k_1])

            else:
                current_project_info[k_1] = dictionary_default[k_1]    

        complement_path = args.json_path[:-6] + "complement" + args.json_path[-6:]
        
        file_info = dictionary_default["file_info"]
        file_info["type"]["use"] = file_info["choices"][0] #reduced
        file_info["type"]["complement_file"]["use"] = complement_path
        
        dictionary_default["file_info"]["type"]["use"] = file_info["choices"][1] # complement
        dictionary_default["file_info"]["type"]["reduced_file"]["use"] = args.json_path
        
        dictionary_of_interest = {
            "current_project_info": current_project_info,
            "file_info": file_info
        }    
        
        with open(args.json_path, 'w') as f:
            json.dump(dictionary_of_interest, f)

        with open(complement_path, 'w') as f:
            json.dump(dictionary_default, f)
            
        return True
    
    
    def __init__(self, args) -> None:
        
        
        # if args.reduced_json:
        #     assert args.function, "If you want to use a reduced json version, the program should know witch Project Function you want to use, so 'function' should be provieded."
        #     if os.path.exists(args.json_path):
        #         "The json with the path and name provided already exists. \nThe program will use it as the reduced form and condere that the complment already exists."
        #     else:
        #         self.create_jsonfile(args)
        
        
        # print("Now use the reduced json to configure your Project and execute the bash file again.")
        
        self.LINE = "-"*100
        print(self.LINE + "[Checking JSON File]" + self.LINE)
        self.PATH_JSON_CONFIG = args.json_path
        self.PATH_JSON_GENERAL_CONFIG = None
        self.CONFIG = None
        self.CCONFIG = None
        general_config_dic = None
        
        #* loading and checking project and general info
        with open(self.PATH_JSON_CONFIG) as file:
            self.CONFIG = json.load(file)
            
            #* general config file
            with open(self.CONFIG["general_info"]["path"]) as file_general_config:
                general_config_dic = json.load(file_general_config)
                self.PATH_JSON_GENERAL_CONFIG = self.CONFIG["general_info"]["path"]
            
            self.CONFIG["general_info"] = general_config_dic["general_info"].copy()
            self.CONFIG["file_info"] = general_config_dic["file_info"].copy()
            
            assert self.check_config(self.CONFIG), "Something went wrong with the json file."
            self.fancy_print_dict(self.CONFIG)
            print("[Checking DONE]\n[Valid Configuration]")
            print(self.LINE + "[Current Configuration]" + self.LINE)
            self.CCONFIG = self.print_project_config(self.CONFIG)
            
        #* save general updated configuration, if it was updated
        if not self.CONFIG["general_info"] == general_config_dic["general_info"] or not self.CONFIG["file_info"] == general_config_dic["file_info"]:
            print("[PRINT] Saving General Config updates")
            general_config_dic["general_info"] = self.CONFIG["general_info"]
            general_config_dic["file_info"] = self.CONFIG["file_info"]
            
            saveorupdate_json(json_path=self.CONFIG["general_info"]["path"], config=general_config_dic)
        