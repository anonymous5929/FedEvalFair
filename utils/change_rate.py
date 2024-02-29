import pandas as pd
import numpy as np
import os 

root_dir = 'processed_data/bias_results'
output_file = 'processed_data/change_rate.xlsx'
change_rate_dict = {}
contrary_rate_dict = {}
index_labels = []
for bias_name in os.listdir(root_dir):
    input_dir = os.path.join(root_dir, bias_name, 'tradition', '1-0.xlsx')
    print(input_dir)
    df_origin = pd.read_excel(input_dir, sheet_name='origin', engine='openpyxl')
    df_another = pd.read_excel(input_dir, sheet_name='another', engine='openpyxl')
    change_rate_bias_dict = {}
    contrary_rate_bias_dict = {}
    for (_, client_origins), (_, client_anothers) in zip(df_origin.iterrows(), df_another.iterrows()):
        for client_i, (origin, another) in enumerate(zip(client_origins.iteritems(), client_anothers.iteritems())):
            origin = origin[1]
            another = another[1]
            if origin != 0:
                change_rate = abs(abs(origin) - abs(another)) / abs(origin) 
            else:
                change_rate = 1
            change_rate_bias_dict.setdefault(f'client_{client_i}', []).append(change_rate)
            contrary_rate_bias_dict.setdefault(f'client_{client_i}', []).append(1 if origin*another < 0 else 0)
    index_labels.append(bias_name)
    for client_name, change_rate_list in change_rate_bias_dict.items():
        change_rate_dict.setdefault(client_name, []).append(np.mean(change_rate_list))
    for client_name, contrary_rate_list in contrary_rate_bias_dict.items():
        contrary_rate_dict.setdefault(client_name, []).append(np.mean(contrary_rate_list))

excel_writer = pd.ExcelWriter(output_file, engine='openpyxl', mode='w')
pd.DataFrame(change_rate_dict, index=index_labels).to_excel(excel_writer, sheet_name='change_rate')
pd.DataFrame(contrary_rate_dict, index=index_labels).to_excel(excel_writer, sheet_name='contrary_rate')
excel_writer.close()
print('done')