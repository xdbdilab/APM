import pandas as pd
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time
# google_file = 'google2019_usages.csv'
ali_file = 'ali2018usage.csv'
output_dir = './{0}_data12-10/'

cpu_train_file = '{0}_cpu_train.csv'
cpu_test_file = '{0}_cpu_test.csv'
mem_train_file = '{0}_mem_train.csv'
mem_test_file = '{0}_mem_test.csv'



dataset_size = 1000000


# ali: 1 for 10s-scale predict, 6 for minute-scale predict, 360 for hour-scale predict
# time_scale = 1
ali_scale_table = {
    'seconds': 1,
    'minutes': 6,
    'hours': 360
}

# google: 10 for 10s-scale predict, 60 for minute-scale predict, 3600 for hour-scale predict
google_scale_table = {
    'seconds': 10,
    'minutes': 60,
    'hours': 3600
}


def get_csv_data(file_path, wl, pl, var, time_scale):
    df = pd.read_csv(file_path)
    data = []
    for i in tqdm(range(0, len(df) - (wl + pl + 1) * time_scale, 1)):
        tmp = []
        for j in range(0, wl + pl, 1):
            value = df[var].values[i + j * time_scale: i + (j+1) * time_scale]
            value = np.mean(np.array(value))
            if np.isnan(value):
                front = np.mean(np.array(df[var].values[i + (j-1) * time_scale: i + j * time_scale]))
                next = np.mean(np.array(df[var].values[i + (j+1) * time_scale: i + (j+2) * time_scale]))
                value = (front + next) / 2
            elif np.isinf(value):
                value = float(value)
            tmp.append(value)
        data.append(tmp)

    mu = np.mean(data, axis=1)
    sigma = np.std(data, axis=1)
    standard = []
    for i in range(len(data)):
        if sigma[i] == 0:
            continue
        temp = (np.array(data[i]) - mu[i]) / sigma[i]
        standard.append(temp)
    standard = pd.DataFrame(standard)
    return standard


if __name__ == '__main__':
    wl = 12
    pl = 10
    scales = ['seconds', 'minutes', 'hours']
    for scale in scales:
        op_dir = output_dir.format(scale)
        if not os.path.exists(op_dir):
            os.makedirs(op_dir)
        # google
        # cpu = get_csv_data(google_file, wl, pl, 'cpu', google_scale_table[scale])
        # mem = get_csv_data(google_file, wl, pl, 'mem', google_scale_table[scale])
        # train_cpu, test_cpu = train_test_split(cpu.values[:dataset_size, :], test_size=0.2, shuffle=False)
        # train_mem, test_mem = train_test_split(mem.values[:dataset_size, :], test_size=0.2, shuffle=False)
        # train_cpu = pd.DataFrame(train_cpu)
        # train_cpu.reset_index(inplace=True, drop=True)
        # train_cpu.to_csv(os.path.join(op_dir, cpu_train_file.format('google')))
        # test_cpu = pd.DataFrame(test_cpu)
        # test_cpu.reset_index(inplace=True, drop=True)
        # test_cpu.to_csv(os.path.join(op_dir, cpu_test_file.format('google')))
        # train_mem = pd.DataFrame(train_mem)
        # train_mem.reset_index(inplace=True, drop=True)
        # train_mem.to_csv(os.path.join(op_dir, mem_train_file.format('google')))
        # test_mem = pd.DataFrame(test_mem)
        # test_mem.reset_index(inplace=True, drop=True)
        # test_mem.to_csv(os.path.join(op_dir, mem_test_file.format('google')))

        # ali
        cpu = get_csv_data(ali_file, wl, pl, 'cpu', ali_scale_table[scale])
        mem = get_csv_data(ali_file, wl, pl, 'mem', ali_scale_table[scale])
        train_cpu, test_cpu = train_test_split(cpu.values[:dataset_size, :], test_size=0.2, shuffle=False)
        train_mem, test_mem = train_test_split(mem.values[:dataset_size, :], test_size=0.2, shuffle=False)
        train_cpu = pd.DataFrame(train_cpu)
        train_cpu.reset_index(inplace=True, drop=True)
        train_cpu.to_csv(os.path.join(op_dir, cpu_train_file.format('ali')))
        test_cpu = pd.DataFrame(test_cpu)
        test_cpu.reset_index(inplace=True, drop=True)
        test_cpu.to_csv(os.path.join(op_dir, cpu_test_file.format('ali')))
        train_mem = pd.DataFrame(train_mem)
        train_mem.reset_index(inplace=True, drop=True)
        train_mem.to_csv(os.path.join(op_dir, mem_train_file.format('ali')))
        test_mem = pd.DataFrame(test_mem)
        test_mem.reset_index(inplace=True, drop=True)
        test_mem.to_csv(os.path.join(op_dir, mem_test_file.format('ali')))
