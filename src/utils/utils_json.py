import json

def saveorupdate_json(json_path, config):
    ## update the JSON config file with new info and results
    with open(json_path, 'w') as f:
        json.dump(config, f)