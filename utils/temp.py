import pandas as pd
import os 

def generate_array(path):
    df = pd.read_excel(path, header=0, sheet_name='bootstrap', engine='openpyxl')
    data = []
    for i in range(11):
        data.append(df['client_{}_disparity'.format(i)].tolist())
    with open('dataset.txt', 'w') as file:
        file.write("dataset = ") 
        file.write(repr(data))

generate_array('2.xlsx')