import pandas as pd
import os

# data_path = r'data'
# seconds_path = os.path.join(data_path,r'seconds_data12-10')
#
# train_data = pd.read_csv(os.path.join(seconds_path,'ali_cpu_train.csv'))
# test_data = pd.read_csv(os.path.join(seconds_path,'ali_cpu_test.csv'))
#
# print(train_data)
# print(test_data)
df1 = pd.read_csv(r'Clustering/cluster_data/ali_cpu_seconds_4/3-cluster0_train.csv')
print(df1)
df2 = pd.read_csv(r'Clustering/cluster_data/ali_cpu_seconds_4/3-cluster1_train.csv')
print(df2)
df3 = pd.read_csv(r'Clustering/cluster_data/ali_cpu_seconds_4/3-cluster2_train.csv')
print(df3)
df4 = pd.read_csv(r'Clustering/cluster_data/ali_cpu_seconds_4/3-cluster3_train.csv')
print(df4)