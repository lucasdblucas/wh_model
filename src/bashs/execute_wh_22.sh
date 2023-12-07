# !/bin/bash

function wh_pred() {
    # 1 model_name
    python ../center.py \
        --json_path $1
}

config_list=(
    "../configs/wh_22_sewrn_4i_org11_bs10_config.json"
)

for config_name in "${config_list[@]}"; do
    wh_pred "$config_name"
done
