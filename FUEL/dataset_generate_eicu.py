import numpy as np
import torch
import pandas as pd
import os
import json

dataset = "eicu"
iseicu = dataset == 'eicu'
issingle = False
isnewsort = True
isweekplus = True

def process_csv(filename, label_name, favorable_class, categorical_attributes, continuous_attributes, features_to_keep):
    header = 'infer'
    df = pd.read_csv(filename, delimiter = ',', header = header, na_values = [])
    if header == None: raise "error"
    df = df[features_to_keep]
    if iseicu and isweekplus:
        df[label_name] = df[label_name] - df['hospitaladmitoffset']

    # apply one-hot encoding to convert the categorical attributes into vectors
    df = pd.get_dummies(df, columns = categorical_attributes)

    # normalize numerical attributes to the range within [0, 1]
    def scale(vec):
        minimum = min(vec)
        maximum = max(vec)
        return (vec-minimum)/(maximum-minimum)

    if iseicu:
        if 'age' in features_to_keep:
            df.loc[df['age'] == '> 89', 'age'] = 90
            df['age'] = df['age'].astype('float64')
        for attr in continuous_attributes:
            df[attr].fillna(df[attr].mean(), inplace = True)
        
    df[continuous_attributes] = df[continuous_attributes].apply(scale, axis = 0)

    if not iseicu:
        df.loc[df[label_name] != favorable_class, label_name] = 114514
        df.loc[df[label_name] == favorable_class, label_name] = 1
        df.loc[df[label_name] == 114514, label_name] = 0
        df[label_name] = df[label_name].astype('category').cat.codes
    else:
        df.loc[df[label_name] <= 10080, label_name] = 0
        df.loc[df[label_name] > 10080, label_name] = 1
        
    df = df.astype('float64')
    return df

np.random.seed(1)
torch.manual_seed(0)

if dataset == "bank":
    filename = 'data/bank_cat_age.csv'
    categorical_attributes = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    continuous_attributes = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    features_to_keep = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 
                        'balance', 'duration', 'campaign', 'pdays', 'previous', 'y']
    label_name = 'y'
    favorable_class = 'yes'

    sensitive_attribute = "loan_no"
    def sensitive_condition(df):
        return df[sensitive_attribute] == 0
    def client1_condition(df):
        return df["marital_married"] == 1
elif dataset == "compas":
    filename = 'data/compas-scores-two-years.csv'
    # categorical_attributes = ['sex', 'age_cat', 'c_charge_degree', 'c_charge_desc', 'race']
    categorical_attributes = ['sex', 'age_cat', 'c_charge_degree',  'race']
    continuous_attributes = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    # features_to_keep = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
    #         'priors_count', 'c_charge_degree', 'c_charge_desc','two_year_recid']
    features_to_keep = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
            'priors_count', 'c_charge_degree', 'two_year_recid']
    label_name = 'two_year_recid'
    favorable_class = 0

    sensitive_attribute = "race_African-American"
    def sensitive_condition(df):
        return df[sensitive_attribute] == 1
    def client1_condition(df):
        return df["age"] > 0.1
elif dataset == "eicu":
    filename = 'data/patient.csv'
    categorical_attributes = ['gender', 'ethnicity', 'hospitaladmitsource', 'hospitaldischargelocation', 'hospitaldischargestatus', 'unittype', 'unitadmitsource', 'unitstaytype', 'unitdischargelocation', 'unitdischargestatus']
    continuous_attributes = ['age', 'admissionheight', 'hospitaladmitoffset', 'hospitaldischargeoffset', 'unitvisitnumber', 'admissionweight', 
                             'dischargeweight']
    label_name = 'unitdischargeoffset'
    features_to_keep = categorical_attributes + continuous_attributes + ['hospitalid', label_name]
    favorable_class = 0

    sensitive_attribute = "ethnicity_African American"

    def sensitive_condition(df):
        return df[sensitive_attribute] == 1

data_frame = process_csv(filename, label_name, favorable_class, categorical_attributes, continuous_attributes, features_to_keep)
print(data_frame)
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
if iseicu:
    print(data_frame.columns.to_list())
    print((data_frame[label_name] == 1).sum())
    print((data_frame[label_name] == 0).sum())
index = data_frame.drop(label_name, axis=1, inplace=False).columns.tolist().index(sensitive_attribute)
print('the column {} index: {}'.format(sensitive_attribute, index))
avg1 = data_frame.loc[sensitive_condition(data_frame), label_name].mean()
avg2 = data_frame.loc[~ sensitive_condition(data_frame), label_name].mean()
print("is suitable to be sensitive attribute? {} vs {}".format(avg1, avg2))
# print(bank[bank[client_attribute] == 0][sensitive_attribute][0:20])
# print(bank[bank[client_attribute] == 1].drop('y', axis=1, inplace=False).iloc[0:20, index])


# exit(0)

def handle(df, output_dir):
    dict_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    def handle_client(client_df, client_name):
        # print(df['y'].values)
        # print(df.values)
        dict_data['users'].append(client_name)
        if not iseicu:
            dict_data['user_data'][client_name] = {
                'x': client_df.drop(label_name, axis=1, inplace=False).values.tolist(), 
                'y': client_df[label_name].values.tolist(),
            }
        else:
            dict_data['user_data'][client_name] = {
                'x': client_df.drop([label_name, sensitive_attribute], axis=1, inplace=False).values.tolist(), 
                'y': client_df[label_name].values.tolist(),
                'race': client_df[sensitive_attribute].values.tolist(),
            }
        dict_data['num_samples'].append(client_df.shape[0])
        print("{} samples number: {}".format(client_name, client_df.shape[0]))
        # print(train_data)
        # print(client_df.shape[0])
        # print(client_df.shape[1])
        # print(client_df.columns.tolist())
        # print(len(dict_data['user_data'][client_name]['x'][1]))
        # print(len(dict_data['user_data'][client_name]['x'][111]))
        # print(len(dict_data['user_data'][client_name]['x']))
        # print(len(dict_data['user_data'][client_name]['y']))
        # print(dict_data['user_data'][client_name]['y'][1])
    if not iseicu:
        client_1_df = df[client1_condition(df)]
        handle_client(client_df=client_1_df, client_name='client_1')
        client_2_df = df[~ client1_condition(df)]
        handle_client(client_df=client_2_df, client_name='client_2')
    else:
        client_data_frames = [data_frame_tuple[1].drop('hospitalid', axis = 1) for data_frame_tuple in df.groupby('hospitalid')]
        for i, client_df in enumerate(client_data_frames):
            handle_client(client_df=client_df, client_name='hospital_{}'.format(i))
    with open(output_dir, 'w') as outfile:
        json.dump(dict_data, outfile)
if not iseicu:
    train = data_frame.iloc[:int(len(data_frame)*.7)]
    test = data_frame.iloc[int(len(data_frame)*.7):]
else:
    data_frames = [data_frame_tuple[1] for data_frame_tuple in data_frame.groupby('hospitalid')]
    def sensitive_mean(df):
        return df.loc[sensitive_condition(df), label_name].mean()
    def sensitive_rate(df):
        return sensitive_condition(df).sum() / df.shape[0]
    def disparity(df):
        return df.loc[sensitive_condition(df), label_name].mean() - df.loc[~sensitive_condition(df), label_name].mean()
    sensitive_rate_list = [sensitive_rate(df) for df in data_frames]
    sensitive_rate_avg = sum(sensitive_rate_list)/len(sensitive_rate_list)
    number_avg = sum([df.shape[0] for df in data_frames]) / len(data_frames)
    print('sensitive rate avg: {}'.format(sensitive_rate_avg))
    print('number avg: {}'.format(number_avg))
    print('disparity avg: {}'.format(disparity(data_frame)))
    # data_frames = [df for df in data_frames if sensitive_rate(df) > sensitive_rate_avg * 2 and df.shape[0] > number_avg]
    data_frames = [df for df in data_frames if df.shape[0] > number_avg]
    if isnewsort:
        data_frames = sorted(data_frames, key = disparity, reverse=True)
    else:
        data_frames = sorted(data_frames, key = sensitive_mean)
    client_num = 11
    if not issingle:
        client_data_frames = []
        df_num = int(len(data_frames) / client_num)
        print(f'a client has {df_num} hospital\'s data')
        for i in range(client_num):
            client_data_frame = pd.concat([data_frames[j] for j in range(df_num * i, df_num * (i+1))])
            client_data_frame['hospitalid'] = i
            client_data_frames.append(client_data_frame)
            # print(client_data_frames[i])
            # print(sensitive_mean(client_data_frames[i]))
            # print(len(client_data_frames[i]))
        data_frames = client_data_frames
    else:
        data_frames = data_frames[0:client_num]
    print(len(data_frames))
    for df in data_frames:
        print(disparity(df))
    train = pd.concat([df.iloc[:int(len(df)*.7)] for df in data_frames])
    test = pd.concat([df.iloc[int(len(df)*.7):] for df in data_frames])

handle(train, "data/{}/train/mytrain.json".format(dataset))
handle(test, "data/{}/test/mytest.json".format(dataset))