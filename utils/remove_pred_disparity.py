import pandas as pd
import os 



def handle_bootstrap(input_dir, output_dir):
    df = pd.read_excel(input_dir, header=0, engine='openpyxl')
    df.drop('client_0_pred_disparity', axis=1)
    df.drop('client_1_pred_disparity', axis=1)
    excel_writer = pd.ExcelWriter(output_dir, engine='openpyxl', mode='w')
    df.to_excel(excel_writer = excel_writer, index = False, sheet_name='bootstrap')
    excel_writer.close()

for rate in [1, 10]:
    input_dir = f'FUEL/out/bank_noloan_specific_{int(int(rate)/10)}-{int(rate)%10}_1-0/results/bootstrap.xlsx'
    output_dir = f'results/bank_noloan/bootstrap_{int(int(rate)/10)}-{int(rate)%10}.xlsx'
    os.makedirs(os.path.split(output_dir)[0], exist_ok=True)
    handle_bootstrap(input_dir, output_dir)


