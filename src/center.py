import sys
sys.path.insert(0, " ../../src")

from argparse import ArgumentParser
from configs.config import Config
from utils.utils_images import display_images_from_dataloader
from experiments.skinreader_experiment import SkinreaderExperiment
from experiments.wh_experiment import WHPredExperiment

def get_args():
    parser = ArgumentParser(
        prog='Center',
        description='Center to execute models algorithms.',
        usage='%(prog)s [options]'
    )
    
    parser.add_argument(
        '--function', 
        type=str,
        required=False,
        help="Main Function, witch will construct the model architecture and define all parameters necessaries to the project."
    )
    
    parser.add_argument(
        "--reduced_json",
        action="store_true",
        help="Use a reduced version with default configuration for this project. The function name will used for this. Default to False."
    )
    
    parser.add_argument(
        '--json_path', 
        type=str, 
        required=True, 
        help="Path to json file with the configuration of the experiment."
    )
    
    return parser.parse_args()

def wh_models(config):
    data_config = config.CONFIG["current_project_info"]["data_config"]
    
    train_loader, val_loader, test_loader, pred_loader = None, None, None, None
    wh_pred = WHPredExperiment()
    
    #* train/test
    if data_config["split_data_type"]["use"] == data_config["split_data_type"]["choices"][1]:
        #* Define dataset
        train_loader, _, test_loader, _ = wh_pred.define_data(config=config)
        #* Initiate training
        wh_pred.define_model_and_train(
            config=config, 
            train_loader=train_loader,
            test_loader=test_loader
        )
    
    # # prediction
    # elif data_config["split_data_type"]["use"] == data_config["split_data_type"]["choices"][3]:
    #     _, _, _, infe_loader = wh_pred.define_data(config=config)

if __name__ == "__main__":
    
    args = get_args()
    
    config = Config(args)
    
    general_config = config.CONFIG["general_info"]
    command = config.CONFIG["current_project_info"]["command"]
    infe_train_config = config.CONFIG["current_project_info"]["train_infe_config"]
    
    if infe_train_config["model_func"]["use"] in general_config["models_names"]["wh_models"]["use"]: #* WH models
        wh_models(config=config)