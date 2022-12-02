import logging
import os
from collections import Counter
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import hydra
from omegaconf import DictConfig
import dataframe_image as dfi
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../config/classification", config_name="config")
def dynamic_fraud_classification(cfg: DictConfig):
    log.info(cfg)
    model = get_model(cfg.model)
    data_name = cfg.data.name
    data_path = cfg.data.path
    sampling = cfg.over_sampling
    result_dir = cfg.result_dir
    group_time = cfg.group_time
    dynamic_clf(model, data_path, data_name, over_sampling=sampling, result_dir=result_dir,
                group_time_length=group_time)


def get_model(cfg: DictConfig):
    match cfg.name:
        case 'RandomForest':
            return RandomForestClassifier(n_estimators=cfg.num_estimator)
        case 'KNN':
            return KNeighborsClassifier(n_neighbors=cfg.num_neighbors)
        case 'DecisionTree':
            return DecisionTreeClassifier(max_depth=cfg.max_depth)
        case 'SCV':
            return SVC(gamma=cfg.gamma, C=cfg.c)
        case "Ada":
            return AdaBoostClassifier(n_estimators=cfg.num_estimator)


def dynamic_clf(model, data_path, data_name, over_sampling=None, result_dir="../../data/results",
                group_time_length=7200):
    df = pd.read_csv(data_path)
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Amount'], axis=1)
    max_time = df["Time"].max()
    group_df = df.groupby(pd.cut(df["Time"], np.arange(-1, max_time + group_time_length, group_time_length)))
    acc_l = []
    pre_l = []
    rec_l = []
    f1_l = []
    tn_l = []
    fp_l = []
    fn_l = []
    tp_l = []
    group_time_l = []
    train_data = pd.DataFrame()
    for group_time, group in group_df:
        # print("")
        log.info(
            f"===== Processing group:{group_time} , group number:{int((group_time.left + 1) / group_time_length + 1)}=====")
        group_number = int((group_time.left + 1) / group_time_length + 1)
        if group_number > 3:
            log.info(f"Predicting group {group_number}")
            X_test = group[group.columns.difference(["Class", "Time"])]
            y_train = group["Class"]
            model_prediction = predict_with_metric(model, X_test, y_train)
            group_time_l.append(group_time)
            acc_l.append(model_prediction["Accuracy"])
            pre_l.append(model_prediction["Precision"])
            rec_l.append(model_prediction["Recall"])
            f1_l.append(model_prediction["F1"])
            tn_l.append(model_prediction["TN"])
            fp_l.append(model_prediction["FP"])
            fn_l.append(model_prediction["FN"])
            tp_l.append(model_prediction["TP"])
        train_data = pd.concat([train_data, group])
        X_train = train_data[train_data.columns.difference(["Class", "Time"])]
        y_train = train_data["Class"]
        if over_sampling == "ROS":
            log.info("Oversampling with ROS")
            ros = RandomOverSampler(random_state=1234)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        if over_sampling == "SMOTE":
            log.info("Oversampling with SMOTE")
            smote = RandomOverSampler(random_state=1234)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        # Check the number of records after over sampling
        log.info(sorted(Counter(y_train).items()))
        # Training model
        model.fit(X_train, y_train)
    metric_data = list(
        zip(group_time_l, acc_l, pre_l, rec_l, f1_l, tn_l, fp_l, fn_l, tp_l))
    df = pd.DataFrame(metric_data, columns=['period', 'Accuracy', 'Precision', 'Recall', 'F-1', 'TN', 'FP', 'FN', 'TP'])
    # Create directory for the results
    result_directory = f"{result_dir}/{data_name}"
    if not os.path.exists(f"{result_dir}/{data_name}"):
        os.makedirs(result_directory)

    df.to_csv(f"{result_directory}/{type(model).__name__}.csv")
    dfi.export(df.style.hide_index(), f"{result_directory}/{type(model).__name__}.png")


def predict_with_metric(model, X_test, y_test):
    prediction = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, prediction)
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tp = conf_matrix[1][1]
    return {"Prediction": list(prediction.flatten()), "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction), "Recall": recall_score(y_test, prediction),
            "F1": f1_score(y_test, prediction), "TN": tn, "FP": fp, "FN": fn, "TP": tp}
