import pandas as pd
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='input_dir', type=str, required=True)
parser.add_argument('-o', dest='output_dir', type=str, required=True)
parser.add_argument('--bootstrap', action='store_true', default=False)
args = parser.parse_args()

class ResultAggregator:
    def __init__(self):
        self.data_origin = {}
        self.data_another = {}
        self.bootstrap_i = 0
    def handle_file(self, file_path, bootstrap_dir=None):
        print(file_path)
        df = pd.read_excel(file_path, header=None, sheet_name='origin', engine='openpyxl')
        for j, dp in enumerate(df.loc[1]):
            self.data_origin.setdefault(f'client_{j}', []).append(dp)
        df = pd.read_excel(file_path, header=None, sheet_name='another', engine='openpyxl')
        for j, dp in enumerate(df.loc[1]):
            self.data_another.setdefault(f'client_{j}', []).append(dp)
        if bootstrap_dir != None:
            excel_writer = pd.ExcelWriter(os.path.join(bootstrap_dir, f'{self.bootstrap_i}.xlsx'), engine='openpyxl', mode='w')
            df = pd.read_excel(file_path, header=0, sheet_name='origin_bootstrap', engine='openpyxl')
            df.to_excel(excel_writer, index=False, sheet_name='origin_bootstrap')
            df = pd.read_excel(file_path, header=0, sheet_name='another_bootstrap', engine='openpyxl')
            df.to_excel(excel_writer, index=False, sheet_name='another_bootstrap')
            excel_writer.close()
            self.bootstrap_i += 1
    def save(self, file_path):
        if self.data_origin == {} and self.data_another == {}:
            return
        excel_writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='w')
        pd.DataFrame(self.data_origin).to_excel(excel_writer, index=False, sheet_name='origin')
        pd.DataFrame(self.data_another).to_excel(excel_writer, index=False, sheet_name='another')
        excel_writer.close()

bootstrap = args.bootstrap
output_dir = args.output_dir
origin_result_dir = args.input_dir
os.makedirs(output_dir)
tradition_dir = os.path.join(output_dir, 'tradition')
os.makedirs(tradition_dir)
bootstrap_dir = os.path.join(output_dir, 'bootstrap')
os.makedirs(bootstrap_dir)
aggregators = []
for i in range(10):
    aggregators.append(ResultAggregator())
for unit_dir in os.listdir(origin_result_dir):
    if not 'eicu_DP_0-01_0-50_seed_' in unit_dir:
        continue
    for round_dir in os.listdir(os.path.join(origin_result_dir, unit_dir)):
        for i in range(10):
            rate = i + 1
            filename = f'{0.35:.3f}_{0.15*rate/10:.3f}_{0.5:.3f}.xlsx'
            file_path = os.path.join(origin_result_dir, unit_dir, round_dir, 'results', filename)
            if not os.path.exists(file_path):
                continue
            if rate == 10 and bootstrap:
                aggregators[i].handle_file(file_path, bootstrap_dir)
            else:
                aggregators[i].handle_file(file_path)
for i in range(10):
    rate = i + 1
    aggregators[i].save(os.path.join(tradition_dir, f'{int(int(rate)/10)}-{int(rate)%10}.xlsx'))
