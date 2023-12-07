import json
import numpy as np

results_path = "C:/Users/lucas/Documents/Area_de_Trabalho/framework-models/src/save/01_wh_definitive/01_wh_04_config_02def.json"

results_dictionary = None
with open(results_path) as file:
    results_dictionary = json.load(file)
    


if results_dictionary != None:
    
    history_dictionary = results_dictionary["current_project_info"]["results"]["use"]["history"]
    best_weight_dict = {
        "best_weight_mae": np.inf,
        "epoch": "epoch_0"
    }
    best_height_dict = {
        "best_height_mae": np.inf,
        "epoch": "epoch_0"
    }
    for k_0, v_0 in history_dictionary.items():
        
        for k_1, v_1 in v_0.items():
            if k_1 == "test":
                if best_weight_dict["best_weight_mae"] > v_1["test_mean_weight_mae"]:
                    best_weight_dict["best_weight_mae"] = v_1["test_mean_weight_mae"]
                    best_weight_dict["epoch"] = k_0
                if best_height_dict["best_height_mae"] > v_1["test_mean_height_mae"]:
                    best_height_dict["best_height_mae"] = v_1["test_mean_height_mae"]
                    best_height_dict["epoch"] = k_0
    
    
    print("[RESULT] best weight dict: {}".format(best_weight_dict))
    print("[RESULT] best height dict: {}".format(best_height_dict))