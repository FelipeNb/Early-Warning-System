import pandas as pd
# ____READ FROM CSV____

def getTrainingSettings(df):
    dic = {}
    dic['location'] = str(df['location'][0])
    dic['delete_columns'] = []
    if(df['delete_columns'][0] != -9999):
        dic['delete_columns'] = str(
            df['delete_columns'][0].replace("-", ",")).split(',')
    dic['change_to_int'] = []
    if(df['change_to_int'][0] != -9999):
        dic['change_to_int'] = str(
            df['change_to_int'][0].replace("-", ",")).split(',')
    dic['change_to_float'] = []
    if(df['change_to_float'][0] != -9999):
        dic['change_to_float'] = str(
            df['change_to_float'][0].replace("-", ",")).split(',')
    dic['what_is_exp'] = str(df['what_is_exp'][0].replace("-", ",")).split(',')
    dic['classificatory_data'] = str(df['classificatory_data'][0])
    return dic


def getConfigurationModels(df):
    dic = {}
    if (str(df['regularization_log_reg'][0]) != 'none' and str(df['regularization_log_reg'][0]) != 'auto'):
        dic['regularization_log_reg'] = float(df['regularization_log_reg'][0])
    elif (str(df['regularization_log_reg'][0]) == 'auto'):
        dic['regularization_log_reg'] = str(df['regularization_log_reg'][0])
    else:
        dic['regularization_log_reg'] = None
    if (str(df['solver_log_reg'][0]) != 'none'):
        dic['solver_log_reg'] = str(df['solver_log_reg'][0])
    else:
        dic['solver_log_reg'] = None
    if (str(df['multiclass_log_reg'][0]) != 'none'):
        dic['multiclass_log_reg'] = str(df['multiclass_log_reg'][0])
    else:
        dic['multiclass_log_reg'] = None
    if (str(df['random_state_log_reg'][0]) != 'none' and str(df['random_state_log_reg'][0]) != 'auto'):
        dic['random_state_log_reg'] = int(df['random_state_log_reg'][0])
    elif (str(df['random_state_log_reg'][0]) == 'auto'):
        dic['random_state_log_reg'] = str(df['random_state_log_reg'][0])
    else:
        dic['random_state_log_reg'] = None
    if (str(df['regularization_svm'][0]) != 'none' and str(df['regularization_svm'][0]) != 'auto'):
        dic['regularization_svm'] = float(df['regularization_svm'][0])
    elif (str(df['regularization_svm'][0]) == 'auto'):
        dic['regularization_svm'] = str(df['regularization_svm'][0])
    else:
        dic['regularization_svm'] = None
    if (str(df['kernel_svm'][0]) != 'none'):
        dic['kernel_svm'] = str(df['kernel_svm'][0])
    else:
        dic['kernel_svm'] = None
    if (str(df['gamma_svm'][0]) != 'none' and str(df['gamma_svm'][0]) != 'auto'):
        dic['gamma_svm'] = int(df['gamma_svm'][0])
    elif (str(df['gamma_svm'][0]) == 'auto'):
        dic['gamma_svm'] = str(df['gamma_svm'][0])
    else:
        dic['gamma_svm'] = None
    if (str(df['random_state_svm'][0]) != 'none' and str(df['random_state_svm'][0]) != 'auto'):
        dic['random_state_svm'] = int(df['random_state_svm'][0])
    elif (str(df['random_state_svm'][0]) == 'auto'):
        dic['random_state_svm'] = str(df['random_state_svm'][0])
    else:
        dic['random_state_svm'] = None
    if (str(df['criterion_decision_tree'][0]) != 'none'):
        dic['criterion_decision_tree'] = str(df['criterion_decision_tree'][0])
    else:
        dic['criterion_decision_tree'] = None
    if (str(df['max_depth_decision_tree'][0]) != 'none' and str(df['max_depth_decision_tree'][0]) != 'auto'):
        dic['max_depth_decision_tree'] = int(df['max_depth_decision_tree'][0])
    elif (str(df['max_depth_decision_tree'][0]) == 'auto'):
        dic['max_depth_decision_tree'] = str(df['max_depth_decision_tree'][0])
    else:
        dic['max_depth_decision_tree'] = None
    if (str(df['random_state_decision_tree'][0]) != 'none' and str(df['random_state_decision_tree'][0]) != 'auto'):
        dic['random_state_decision_tree'] = int(
            df['random_state_decision_tree'][0])
    elif (str(df['random_state_decision_tree'][0]) == 'auto'):
        dic['random_state_decision_tree'] = str(
            df['random_state_decision_tree'][0])
    else:
        dic['random_state_decision_tree'] = None
    if (str(df['criterion_random_forest'][0]) != 'none'):
        dic['criterion_random_forest'] = str(df['criterion_random_forest'][0])
    else:
        dic['criterion_random_forest'] = None
    if (str(df['n_estimators_random_forest'][0]) != 'none' and str(df['n_estimators_random_forest'][0]) != 'auto'):
        dic['n_estimators_random_forest'] = int(
            df['n_estimators_random_forest'][0])
    elif (str(df['n_estimators_random_forest'][0]) == 'auto'):
        dic['n_estimators_random_forest'] = str(
            df['n_estimators_random_forest'][0])
    else:
        dic['n_estimators_random_forest'] = None
    if (str(df['n_processors_random_forest'][0]) != 'none' and str(df['n_processors_random_forest'][0]) != 'auto'):
        dic['n_processors_random_forest'] = int(
            df['n_processors_random_forest'][0])
    elif (str(df['n_processors_random_forest'][0]) == 'auto'):
        dic['n_processors_random_forest'] = str(
            df['n_processors_random_forest'][0])
    else:
        dic['n_processors_random_forest'] = None
    if (str(df['random_state_random_forest'][0]) != 'none' and str(df['random_state_random_forest'][0]) != 'auto'):
        dic['random_state_random_forest'] = int(
            df['random_state_random_forest'][0])
    elif (str(df['random_state_random_forest'][0]) == 'auto'):
        dic['random_state_random_forest'] = str(
            df['random_state_random_forest'][0])
    else:
        dic['random_state_random_forest'] = None
    if (str(df['type_distance_knn'][0]) != 'none' and str(df['type_distance_knn'][0]) != 'auto'):
        dic['type_distance_knn'] = int(df['type_distance_knn'][0])
    elif (str(df['type_distance_knn'][0]) == 'auto'):
        dic['type_distance_knn'] = str(df['type_distance_knn'][0])
    else:
        dic['type_distance_knn'] = None
    if (str(df['n_neighbours_knn'][0]) != 'none' and str(df['n_neighbours_knn'][0]) != 'auto'):
        dic['n_neighbours_knn'] = int(df['n_neighbours_knn'][0])
    elif (str(df['n_neighbours_knn'][0]) == 'auto'):
        dic['n_neighbours_knn'] = str(df['n_neighbours_knn'][0])
    else:
        dic['n_neighbours_knn'] = None
    if (str(df['metric_knn'][0]) != 'none'):
        dic['metric_knn'] = str(df['metric_knn'][0])
    else:
        dic['metric_knn'] = None
    if (str(df['loss_deep'][0]) != 'none'):
        dic['loss_deep'] = str(df['loss_deep'][0])
    else:
        dic['loss_deep'] = None
    if (str(df['metric_deep'][0]) != 'none'):
        dic['metric_deep'] = str(df['metric_deep'][0])
    else:
        dic['metric_deep'] = None
    if (str(df['optmizer_deep'][0]) != 'none'):
        dic['optmizer_deep'] = str(df['optmizer_deep'][0])
    else:
        dic['optmizer_deep'] = None
    if (str(df['activation_deep'][0]) != 'none'):
        dic['activation_deep'] = str(df['activation_deep'][0])
    else:
        dic['activation_deep'] = None
    if (str(df['activation_output_deep'][0]) != 'none'):
        dic['activation_output_deep'] = str(df['activation_output_deep'][0])
    else:
        dic['activation_output_deep'] = None
    return dic
