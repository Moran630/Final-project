import os
import sys
import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics

# bowel_healthy,bowel_injury,extravasation_healthy,extravasation_injury,kidney_healthy,kidney_low,kidney_high,liver_healthy,liver_low,liver_high,spleen_healthy,spleen_low,spleen_high


def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df

def score(pred_csv, ground_truth_csv, pid_uid_csv):
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

    
    weight_dict = {'bowel_healthy': 1, 'bowel_injury': 2, 
                   'extravasation_healthy': 1, 'extravasation_injury': 6, 
                   'kidney_healthy': 1, 'kidney_low': 2, 'kidney_high': 4, 
                   'liver_healthy': 1, 'liver_low': 2, 'liver_high': 4, 
                   'spleen_healthy': 1, 'spleen_low': 2, 'spleen_high': 4, 
                   'any_injury': 6}
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


    label_group_losses = []
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        for col in col_group:
            if col not in gt_df_sorted.columns:
                raise (f'Missing submission column {col}')

        loss = sklearn.metrics.log_loss(
                y_true=gt_df_sorted[col_group].values,
                y_pred=pred_df[col_group].values,
                sample_weight=gt_df_sorted[f'{category}_weight'].values
            )
        print(category + ':', loss)
        label_group_losses.append(loss)

    # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    healthy_cols = [x + '_healthy' for x in all_target_categories]
    any_injury_labels = (1 - gt_df_sorted[healthy_cols]).max(axis=1)
    any_injury_predictions = (1 - pred_df[healthy_cols]).max(axis=1)
    any_injury_loss = sklearn.metrics.log_loss(
        y_true=any_injury_labels.values,
        y_pred=any_injury_predictions.values,
        sample_weight=gt_df_sorted['any_injury_weight'].values
    )
    print('any_injury_loss:', any_injury_loss)
    label_group_losses.append(any_injury_loss)
    print('mean:', np.mean(label_group_losses))
    print()


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # print(f'BASE_DIR: {BASE_DIR}')
    # sys.path.append(BASE_DIR)
    os.chdir(BASE_DIR)
    """cnn with lstm"""
    score(
        pred_csv='./data/cnn_lstm/submission_max.csv',
        ground_truth_csv='./data/test_with_weight.csv', 
        pid_uid_csv='./data/pid_uid.csv')
    
    """resnet 3d"""
    score(
          pred_csv='./data/resnet/submission.csv',
          ground_truth_csv='./data/test_with_weight.csv', 
        pid_uid_csv='./data/pid_uid.csv')

    "swinunetr"
    score(
        pred_csv='./data/swinunetr/submission.csv',
        ground_truth_csv='./data/test_with_weight.csv', 
        pid_uid_csv='./data/pid_uid.csv')