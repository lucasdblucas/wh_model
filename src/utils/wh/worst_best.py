import sys
sys.path.insert(0, "../src")

import json
import numpy as np
# from utils.utils_plots_regression import compute_score

# result_json_path = "src/save/wh_03_wrse_03/03_wh_wrse_03_config.json"
# epoch = "epoch_45"
result_json_path = "src/save/01_wh_definitive/01_wh_04_config_02def.json"
epoch = "epoch_44"
n = 10

result_dict = None
with open(result_json_path) as file_config:
    result_dic = json.load(file_config)

targets = result_dic["current_project_info"]["results"]["use"]["history"][epoch]["test"]["test_targets_test"]
preds = result_dic["current_project_info"]["results"]["use"]["history"][epoch]["test"]["test_preds_test"]
ids = result_dic["current_project_info"]["results"]["use"]["history"][epoch]["test"]["test_ids_test"]

y_true_w = np.array([y[0] for y in targets])
y_true_h = np.array([y[1] for y in targets])
y_pred_w = np.array([y[0] for y in preds])
y_pred_h = np.array([y[1] for y in preds])

# w_score_dict = compute_score(y_true= y_true_w, y_pred=y_pred_w)
# h_score_dict = compute_score(y_true= y_true_h, y_pred=y_pred_h)

np_mse_w = ((y_true_w - y_pred_w) **2)
np_rmse_w = np.sqrt(np_mse_w)
np_mae_w = np.abs((y_true_w - y_pred_w))

np_mse_h = ((y_true_h - y_pred_h) **2)
np_rmse_h = np.sqrt(np_mse_h)
np_mae_h = np.abs((y_true_h - y_pred_h))

np_mae_wh = (np_mae_w + np_mae_h)
np_mse_wh = (np_mse_w + np_mse_h)
np_rmse_wh = np.sqrt(np_mse_wh)

worst_index_wh = np.argpartition(np_mae_wh, -n)[-n:] 
worst_index_w = np.argpartition(np_mae_w, -n)[-n:]
worst_index_h = np.argpartition(np_mae_h, -n)[-n:]

best_index_wh = np.argpartition(np_mae_wh, n)[:n]
best_index_w = np.argpartition(np_mae_w, n)[:n]
best_index_h = np.argpartition(np_mae_h, n)[:n]

# test_index = sorted(enumerate(np_mae_w), key=lambda x: x[1], reverse=True)
worst_index_w = sorted(worst_index_w, key=lambda x: np_mae_w[x], reverse=True)
worst_index_wh = sorted(worst_index_wh, key=lambda x: np_mae_wh[x], reverse=True)
best_index_h = sorted(best_index_h, key=lambda x: np_mae_h[x])

# print(worst_index_w)
# print("\n")
# print(worst_index_h)
# print("\n")
# print(best_index_w)
# print("\n")
# print(best_index_h)



# print("[worst_index_wh]")
# for i, idx in enumerate(range(n)):
#     print("idx: {}".format(i))
#     print("MAE - WH: {}".format(np_mae_wh[worst_index_wh[idx]]).replace(".", ","))
#     print("MAE - W: {}".format(np_mae_w[worst_index_wh[idx]]).replace(".", ","))
#     print("MAE - H: {}".format(np_mae_h[worst_index_wh[idx]]).replace(".", ","))
#     print("RMSE - WH: {}".format(np_rmse_wh[worst_index_wh[idx]]).replace(".", ","))
#     print("RMSE - W: {}".format(np_rmse_w[worst_index_wh[idx]]).replace(".", ","))
#     print("RMSE - H: {}".format(np_rmse_h[worst_index_wh[idx]]).replace(".", ","))
#     print("MSE - WH: {}".format(np_mse_wh[worst_index_wh[idx]]).replace(".", ","))
#     print("MSE - W: {}".format(np_mse_w[worst_index_wh[idx]]).replace(".", ","))
#     print("MSE - H: {}".format(np_mse_h[worst_index_wh[idx]]).replace(".", ","))
#     print("ID: {}".format(ids[worst_index_wh[idx]]))
#     print("Weight Target: {}".format(y_true_w[worst_index_wh[idx]]).replace(".", ","))
#     print("Weight Prediction: {}".format(y_pred_w[worst_index_wh[idx]]).replace(".", ","))
#     print("Height Target: {}".format(y_true_h[worst_index_wh[idx]]).replace(".", ","))
#     print("Height Prediction: {}".format(y_pred_h[worst_index_wh[idx]]).replace(".", ","))
#     print("\n")

# print("[worst_index_w]")
# for i, idx in enumerate(range(n)):
#     print("idx: {}".format(i))
#     print("MAE - WH: {}".format(np_mae_wh[worst_index_w[idx]]).replace(".", ","))
#     print("MAE - W: {}".format(np_mae_w[worst_index_w[idx]]).replace(".", ","))
#     print("MAE - H: {}".format(np_mae_h[worst_index_w[idx]]).replace(".", ","))
#     print("RMSE - WH: {}".format(np_rmse_wh[worst_index_w[idx]]).replace(".", ","))
#     print("RMSE - W: {}".format(np_rmse_w[worst_index_w[idx]]).replace(".", ","))
#     print("RMSE - H: {}".format(np_rmse_h[worst_index_w[idx]]).replace(".", ","))
#     print("MSE - WH: {}".format(np_mse_wh[worst_index_w[idx]]).replace(".", ","))
#     print("MSE - W: {}".format(np_mse_w[worst_index_w[idx]]).replace(".", ","))
#     print("MSE - H: {}".format(np_mse_h[worst_index_w[idx]]).replace(".", ","))
#     print("ID: {}".format(ids[worst_index_w[idx]]))
#     print("Weight Target: {}".format(y_true_w[worst_index_w[idx]]).replace(".", ","))
#     print("Weight Prediction: {}".format(y_pred_w[worst_index_w[idx]]).replace(".", ","))
#     print("Height Target: {}".format(y_true_h[worst_index_w[idx]]).replace(".", ","))
#     print("Height Prediction: {}".format(y_pred_h[worst_index_w[idx]]).replace(".", ","))
    # print("\n")
    

print("[best_index_h]")
for i, idx in enumerate(range(n)):
    print("idx: {}".format(i))
    print("MAE - WH: {}".format(np_mae_wh[best_index_h[idx]]).replace(".", ","))
    print("MAE - W: {}".format(np_mae_w[best_index_h[idx]]).replace(".", ","))
    print("MAE - H: {}".format(np_mae_h[best_index_h[idx]]).replace(".", ","))
    print("RMSE - WH: {}".format(np_rmse_wh[best_index_h[idx]]).replace(".", ","))
    print("RMSE - W: {}".format(np_rmse_w[best_index_h[idx]]).replace(".", ","))
    print("RMSE - H: {}".format(np_rmse_h[best_index_h[idx]]).replace(".", ","))
    print("MSE - WH: {}".format(np_mse_wh[best_index_h[idx]]).replace(".", ","))
    print("MSE - W: {}".format(np_mse_w[best_index_h[idx]]).replace(".", ","))
    print("MSE - H: {}".format(np_mse_h[best_index_h[idx]]).replace(".", ","))
    print("ID: {}".format(ids[best_index_h[idx]]))
    print("Weight Target: {}".format(y_true_w[best_index_h[idx]]).replace(".", ","))
    print("Weight Prediction: {}".format(y_pred_w[best_index_h[idx]]).replace(".", ","))
    print("Height Target: {}".format(y_true_h[best_index_h[idx]]).replace(".", ","))
    print("Height Prediction: {}".format(y_pred_h[best_index_h[idx]]).replace(".", ","))
    print("\n")