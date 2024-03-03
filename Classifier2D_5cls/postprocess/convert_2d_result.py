import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def norm(values):
    sum = 0
    for value in values:
        sum += value
    values_norm = []
    for value in values:
        values_norm.append(value / sum)
    return values_norm

# final_columns = ['img_file', 
#                 'bowel_healthy', 'bowel_injury', 
#                 'extravasation_healthy', 'extravasation_injury', 
#                 'kidney_healthy','kidney_low','kidney_high', 
#                 'liver_healthy','liver_low','liver_high', 
#                 'spleen_healthy','spleen_low','spleen_high']
def convert_2d(input_csv, method='max'):
    input_df = pd.read_csv(input_csv)
    input_columns = input_df.columns.values.tolist()
    final_columns = ['series_id'] + input_columns[1:]
    final_records = []
    result_dict = {}
    for data in tqdm(input_df.values):
        img_file = data[0]
        uid = img_file.split('/')[-2]
        uid = int(uid)
        if uid not in result_dict:
            result_dict[uid] = {}
            result_dict[uid]['bowel_healthy'] = [data[1]]
            result_dict[uid]['bowel_injury'] = [data[2]]
            result_dict[uid]['extravasation_healthy'] = [data[3]]
            result_dict[uid]['extravasation_injury'] = [data[4]]
            result_dict[uid]['kidney_healthy'] = [data[5]]
            result_dict[uid]['kidney_low'] = [data[6]]
            result_dict[uid]['kidney_high'] = [data[7]]
            result_dict[uid]['liver_healthy'] = [data[8]]
            result_dict[uid]['liver_low'] = [data[9]]
            result_dict[uid]['liver_high'] = [data[10]]
            result_dict[uid]['spleen_healthy'] = [data[11]]
            result_dict[uid]['spleen_low'] = [data[12]]
            result_dict[uid]['spleen_high'] = [data[13]]
        else:
            result_dict[uid]['bowel_healthy'].append(data[1])
            result_dict[uid]['bowel_injury'].append(data[2])
            result_dict[uid]['extravasation_healthy'].append(data[3])
            result_dict[uid]['extravasation_injury'].append(data[4])
            result_dict[uid]['kidney_healthy'].append(data[5])
            result_dict[uid]['kidney_low'].append(data[6])
            result_dict[uid]['kidney_high'].append(data[7])
            result_dict[uid]['liver_healthy'].append(data[8])
            result_dict[uid]['liver_low'].append(data[9])
            result_dict[uid]['liver_high'].append(data[10])
            result_dict[uid]['spleen_healthy'].append(data[11])
            result_dict[uid]['spleen_low'].append(data[12])
            result_dict[uid]['spleen_high'].append(data[13])
        
    for uid, result_info in result_dict.items():
        bowel_healthy_arr = np.array(result_info['bowel_healthy'])
        bowel_injury_arr = np.array(result_info['bowel_injury'])
        extravasation_healthy_arr = np.array(result_info['extravasation_healthy'])
        extravasation_injury_arr = np.array(result_info['extravasation_injury'])
        kidney_healthy_arr = np.array(result_info['kidney_healthy'])
        kidney_low_arr = np.array(result_info['kidney_low'])
        kidney_high_arr = np.array(result_info['kidney_high'])
        liver_healthy_arr = np.array(result_info['liver_healthy'])
        liver_low_arr = np.array(result_info['liver_low'])
        liver_high_arr = np.array(result_info['liver_high'])
        spleen_healthy_arr = np.array(result_info['spleen_healthy'])
        spleen_low_arr = np.array(result_info['spleen_low'])
        spleen_high_arr = np.array(result_info['spleen_high'])
        
        if method == 'max':
            bowel_healthy_result = np.min(bowel_healthy_arr)
            bowel_injury_result = np.max(bowel_injury_arr)
            extravasation_healthy_result = np.min(extravasation_healthy_arr)
            extravasation_injury_result = np.max(extravasation_injury_arr)

            kidney_healthy_result = np.min(kidney_healthy_arr)
            kidney_low_result = np.max(kidney_low_arr)
            kidney_high_result = np.max(kidney_high_arr)

            liver_healthy_result = np.min(liver_healthy_arr)
            liver_low_result = np.max(liver_low_arr)
            liver_high_result = np.max(liver_high_arr)

            spleen_healthy_result = np.min(spleen_healthy_arr)
            spleen_low_result = np.max(spleen_low_arr)
            spleen_high_result = np.max(spleen_high_arr)

        elif method == 'mean':
            bowel_healthy_result = np.mean(bowel_healthy_arr)
            bowel_injury_result = np.mean(bowel_injury_arr)
            extravasation_healthy_result = np.mean(extravasation_healthy_arr)
            extravasation_injury_result = np.mean(extravasation_injury_arr)

            kidney_healthy_result = np.mean(kidney_healthy_arr)
            kidney_low_result = np.mean(kidney_low_arr)
            kidney_high_result = np.mean(kidney_high_arr)

            liver_healthy_result = np.mean(liver_healthy_arr)
            liver_low_result = np.mean(liver_low_arr)
            liver_high_result = np.mean(liver_high_arr)

            spleen_healthy_result = np.mean(spleen_healthy_arr)
            spleen_low_result = np.mean(spleen_low_arr)
            spleen_high_result = np.mean(spleen_high_arr)
            
        bowel_healthy_result, bowel_injury_result = norm([bowel_healthy_result, bowel_injury_result])
        extravasation_healthy_result, extravasation_injury_result = norm([extravasation_healthy_result, extravasation_injury_result])
        kidney_healthy_result, kidney_low_result, kidney_high_result = norm([kidney_healthy_result, kidney_low_result, kidney_high_result])
        liver_healthy_result, liver_low_result, liver_high_result = norm([liver_healthy_result, liver_low_result, liver_high_result])
        spleen_healthy_result, spleen_low_result, spleen_high_result = norm([spleen_healthy_result, spleen_low_result, spleen_high_result])
        final_records.append([uid, bowel_healthy_result, bowel_injury_result, extravasation_healthy_result, extravasation_injury_result, kidney_healthy_result, kidney_low_result, kidney_high_result, liver_healthy_result, 
                              liver_low_result, liver_high_result, spleen_healthy_result, spleen_low_result, spleen_high_result])

    output_csv = input_csv.replace('.csv', '') + '_' + method + '.csv'
    print(output_csv)
    data_save = pd.DataFrame(columns=final_columns, data=final_records)
    data_save.drop_duplicates(inplace=True)
    data_save.to_csv(output_csv, index=False)


def make_fake_pred(input_csv, output_csv, data_region='spleen'):
    input_df = pd.read_csv(input_csv)
    input_columns = input_df.columns.values.tolist()
    uids = input_df['series_id'].values.tolist()
    final_columns = input_columns
    final_records = []
    for uid in uids:
        if data_region in  ['spleen', 'liver']:
            final_records.append([uid, 0.8, 0.1, 0.1])
        else:
            final_records.append([uid, 0.9, 0.1, 0.9, 0.1])
    
    data_save = pd.DataFrame(columns=final_columns, data=final_records)
    data_save.drop_duplicates(inplace=True)
    data_save.to_csv(output_csv, index=False)

if __name__ == '__main__':
    convert_2d(input_csv='/data/wangfy/github/kaggle/RSNA/code/Classifier2D_5cls/outputs_cls5_rnn_timestep16_slide_reverse_shuffle_125/test_result/91/submission.csv', 
               method='max')

