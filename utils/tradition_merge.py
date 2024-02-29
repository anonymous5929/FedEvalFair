import pandas as pd
import os 

def tradition_merge(dir, newdir):
    os.makedirs(newdir)
    excel_writer = pd.ExcelWriter(os.path.join(newdir, 'total.xlsx'), engine='openpyxl', mode='w')
    data_origin = {}
    data_another = {}
    for i in range(11):
        data_origin['client_{}'.format(i)] = []
        data_another['client_{}'.format(i)] = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        print('path: {}'.format(path))
        df = pd.read_excel(path, header=None, sheet_name='origin', engine='openpyxl')
        for i, dp in enumerate(df.loc[1]):
            data_origin['client_{}'.format(i)].append(dp)
        df = pd.read_excel(path, header=None, sheet_name='another', engine='openpyxl')
        for i, dp in enumerate(df.loc[1]):
            data_another['client_{}'.format(i)].append(dp)
    data_frame = pd.DataFrame(data_origin)
    data_frame.to_excel(excel_writer = excel_writer, index = False, sheet_name='origin')
    data_frame = pd.DataFrame(data_another)
    data_frame.to_excel(excel_writer = excel_writer, index = False, sheet_name='another')
    excel_writer.close()


root = 'C:\\Users\\ZhengyangZhao\\Desktop\\wanzheng'
newroot = 'C:\\Users\\ZhengyangZhao\\Desktop\\wanzheng_total'

for base, dirs, _ in os.walk(root):
    for dir in dirs:
        dir = os.path.join(base, dir)
        if os.path.exists(os.path.join(dir, '0.xlsx')):
            tradition_merge(dir, os.path.join(newroot, os.path.relpath(dir, root)))
    # print(os.path.join(root, dir))
    


