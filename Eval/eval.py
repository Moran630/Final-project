import os
import numpy as np
import pandas as pd
import sklearn.metrics
import warnings
warnings.filterwarnings("ignore")

def load_df(file, rename_dict=None, sheet_name=None):
    if file.endswith('.csv'):
        df = pd.read_csv(file, encoding='utf-8')
    elif file.endswith('.xlsx') or file.endswith('.xls'):
        df = pd.read_excel(file, sheet_name=0 if sheet_name is None else sheet_name, encoding='utf-8')
    else:
        print('Bad file %s with invalid format, please check in manual!' % file)
        return None

    if rename_dict is not None:
        df = df.rename(columns=rename_dict)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_accuracy(pred_csv, ground_truth_csv, pid_uid_csv, method='macro'):
    pid_uid_df = pd.read_csv(pid_uid_csv)
    uid_pid_map = {}
    for data in pid_uid_df.values:
        pid = data[0]
        uid = data[1]
        uid_pid_map[uid] = pid
    
    gt_df = pd.read_csv(ground_truth_csv)
    gt_columns = gt_df.columns.values.tolist()
    # print('gt_columns:', gt_columns, len(gt_df))

    pred_df = pd.read_csv(pred_csv)
    pred_columns = pred_df.columns.values.tolist()
    # print('pred_columns:', pred_columns, len(pred_df))

    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets
    
    gt_columns_sorted = ['series_id'] + pred_columns[1:] + ['bowel_weight', 'extravasation_weight', 'kidney_weight', 'liver_weight', 'spleen_weight', 'any_injury_weight']
    gt_records_sorted = []
    for data in pred_df.values:
        uid = data[0]
        pid = uid_pid_map[uid]
        items_gt = gt_df[gt_df['patient_id'] == pid]
        assert len(items_gt) == 1
        gt_values = items_gt.values[0]
        gt_values_req = gt_values[1:]
        items_gt_sorted = [uid] + gt_values_req.tolist()
        gt_records_sorted.append(items_gt_sorted)
    
    gt_df_sorted = pd.DataFrame(columns=gt_columns_sorted, data=gt_records_sorted)
    gt_df_sorted.drop_duplicates(inplace=True)

    label_group_acc = []
    print("Accuracy")
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        for col in col_group:
            if col not in gt_df_sorted.columns:
                raise (f'Missing submission column {col}')

        y_true_argmax = gt_df_sorted[col_group].values.argmax(axis=1)
        y_pred_argmax = pred_df[col_group].values.argmax(axis=1)
        # y_pred = pred_df[col_group].values
        # y_pred[:, 0] = y_pred[:, 0] - 0.6
        # y_pred_argmax = y_pred.argmax(axis=1)
        acc = sklearn.metrics.accuracy_score(
                y_true=y_true_argmax,
                y_pred=y_pred_argmax,
                # sample_weight=gt_df_sorted[f'{category}_weight'].values
            )
        print(category + ':', acc)
        label_group_acc.append(acc)
    print('mean accuracy:', np.mean(label_group_acc))

    label_group_recall = []
    print("Recall")
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        for col in col_group:
            if col not in gt_df_sorted.columns:
                raise (f'Missing submission column {col}')

        y_true_argmax = gt_df_sorted[col_group].values.argmax(axis=1)
        y_pred_argmax = pred_df[col_group].values.argmax(axis=1)
        # y_pred = pred_df[col_group].values
        # y_pred[:, 0] = y_pred[:, 0] - 0.6
        # y_pred_argmax = y_pred.argmax(axis=1)
        recall = sklearn.metrics.recall_score(
                y_true=y_true_argmax,
                y_pred=y_pred_argmax,
                # labels=[1, 2],
                # pos_label=2,
                average=method
            )
        print(category + ':', recall)
        label_group_recall.append(recall)
    print('mean recall:', np.mean(label_group_recall))

    label_group_precision = []
    print("Precision")
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        for col in col_group:
            if col not in gt_df_sorted.columns:
                raise (f'Missing submission column {col}')

        y_true_argmax = gt_df_sorted[col_group].values.argmax(axis=1)
        y_pred_argmax = pred_df[col_group].values.argmax(axis=1)
        precision = sklearn.metrics.precision_score(
                y_true=y_true_argmax,
                y_pred=y_pred_argmax,
                # labels=[1, 2],
                # pos_label=2,
                average=method
            )
        print(category + ':', precision)
        label_group_precision.append(precision)
    print('mean precision:', np.mean(label_group_precision))

    label_group_f1score = []
    print("F1 score")
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        for col in col_group:
            if col not in gt_df_sorted.columns:
                raise (f'Missing submission column {col}')

        y_true_argmax = gt_df_sorted[col_group].values.argmax(axis=1)
        y_pred_argmax = pred_df[col_group].values.argmax(axis=1)
        f1score = sklearn.metrics.f1_score(
                y_true=y_true_argmax,
                y_pred=y_pred_argmax,
                # labels=[1, 2],
                # pos_label=2,
                average=method
            )
        print(category + ':', f1score)
        label_group_f1score.append(f1score)
    print('mean f1 score:', np.mean(label_group_f1score))
    print()

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # print(f'BASE_DIR: {BASE_DIR}')
    # sys.path.append(BASE_DIR)
    os.chdir(BASE_DIR)
    """cnn with lstm"""
    get_accuracy(
        pred_csv='./data/cnn_lstm/submission_max.csv',
        ground_truth_csv='./data/test_with_weight.csv', 
        pid_uid_csv='./data/pid_uid.csv')

    """resnet 3d"""
    get_accuracy(
        pred_csv='./data/resnet/submission.csv',
        ground_truth_csv='./data/test_with_weight.csv', 
        pid_uid_csv='./data/pid_uid.csv')

    "swinunetr"
    get_accuracy(
        pred_csv='./data/swinunetr/submission.csv',
        ground_truth_csv='./data/test_with_weight.csv', 
        pid_uid_csv='./data/pid_uid.csv')

