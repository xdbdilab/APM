import random
import math
import numpy as np
import pandas as pd
from tslearn.piecewise import OneD_SymbolicAggregateApproximation
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import os
import warnings
warnings.filterwarnings('ignore')

class Canopy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.t1 = 0
        self.t2 = 0

    # Set the initial threshold
    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print('t1 needs to be larger than t2!')

    # Euclidean distance is used for distance calculation
    def euclideanDistance(self, vec1, vec2):
        return math.sqrt(((vec1 - vec2) ** 2).sum() / len(vec1))

    def getRandIndex(self):
        return random.randint(0, len(self.dataset) - 1)

    def clustering(self):
        if self.t1 == 0:
            print('Please set the threshold.')
        else:
            canopies = []
            centers = []
            # while len(self.dataset) != 0:
            while len(self.dataset) > 1:
                rand_index = self.getRandIndex()
                current_center = self.dataset[rand_index]
                current_center_list = []
                delete_list = []
                self.dataset = np.delete(self.dataset, rand_index,
                                         0)
                for datum_j in range(len(self.dataset)):
                    datum = self.dataset[datum_j]
                    distance = self.euclideanDistance(
                        current_center, datum)
                    if distance < self.t1:
                        current_center_list.append(datum)
                    if distance < self.t2:
                        delete_list.append(datum_j)
                # Removes an element from the dataset based on the subscript of the delete container
                self.dataset = np.delete(self.dataset, delete_list, 0)
                centers.append(current_center)
                canopies.append((current_center, current_center_list))
            print(len(canopies))
            return centers


class DataPre(object, ):
    def __init__(self, seq_length, seq_train, seq_interval, start_point):
        # self.Centroid = centroid
        # if dataset == 'Google':
        #     self.dataset = 'Google'
        #     self.path = "D:/PycharmProjects/APM/google2011usage.csv"
        # elif dataset == 'Ali':
        #     self.dataset = 'Ali'
        #     self.path = "D:/PycharmProjects/APM/ali2018usage.csv"
        self.train_path = None
        self.test_path = None
        self.seq_length = seq_length
        self.seq_interval = seq_interval
        self.start_point = start_point
        self.seq_train = seq_train

    def data_partition(self):
        """
        :return:
        """
        data = pd.read_csv(self.path)
        data = pd.DataFrame(data)
        if self.dataset == 'Google':
            cpu = data['cpu'] * 100
            for i in range(len(cpu)):
                cpu[i] = round(cpu[i], 6)
            # length = int(len(cpu) * 0.4)
            # cpu = cpu[:length]
        else:
            cpu = data['cpu']
        train_length = int(len(cpu) * 0.8)
        train = cpu[int(len(cpu) * 0.2):]
        test = cpu[:int(len(cpu) * 0.2)]
        train.to_csv('train.csv')
        self.train_path = 'train.csv'
        test.to_csv('test.csv')
        self.test_path = 'test.csv'

    def sax(self, data):
        n_paa_segments = 10
        n_sax_symbols_avg = 8
        n_sax_symbols_slop = 8
        sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                  alphabet_size_slope=n_sax_symbols_slop)
        Sax_data = sax.inverse_transform(sax.fit_transform(data))
        data_new = np.reshape(Sax_data, (Sax_data.shape[0], Sax_data.shape[1]))
        return data_new

    def distEclud(self, vecA, vecB):
        return np.sqrt(np.sum(np.power((vecA - vecB), 2)) / len(vecA))

    def distCos(self, vecA, vecB):
        # print(vecA)
        # print(vecB)
        return float(np.sum(np.array(vecA) * np.array(vecB))) / (
                self.distEclud(vecA, np.mat(np.zeros(len(vecA[0])))) * self.distEclud(vecB,
                                                                                      np.mat(np.zeros(len(vecB[0])))))

    def make_sample(self):
        """
        :return:
        """
        path = [self.train_path, self.test_path]
        for filename in path:
            if filename == self.train_path:
                raw_data = pd.read_csv(filename)
                df = pd.DataFrame(raw_data)
                data = []
                data_len = self.seq_length
                interval = self.seq_interval
                if self.dataset == 'Ali':
                    for i in range(self.start_point, 14161 - data_len, interval):
                        tmp = []
                        for j in range(0, data_len, 1):
                            value = df['cpu'].values[i + j]
                            if np.isnan(value):
                                value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
                            tmp.append(value)
                        data.append(tmp)
                    for i in range(16038 + self.start_point, len(raw_data) - data_len, interval):
                        tmp = []
                        for j in range(0, data_len, 1):
                            value = df['cpu'].values[i + j]
                            if np.isnan(value):
                                value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
                            tmp.append(value)
                        data.append(tmp)
                elif self.dataset == 'Google':
                    for i in range(self.start_point, len(raw_data) - data_len, interval):
                        tmp = []
                        for j in range(0, data_len, 1):
                            value = df['cpu'].values[i + j]
                            if np.isnan(value):
                                value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
                            elif np.isinf(value):
                                # print(value)
                                value = float(value)
                            tmp.append(value)
                        data.append(tmp)
                mu = np.mean(data, axis=1)
                sigma = np.std(data, axis=1)
                standard = []
                for i in range(len(data)):
                    if sigma[i] == 0:
                        # print(data[i])
                        continue
                    temp = (np.array(data[i]) - mu[i]) / sigma[i]
                    standard.append(temp)
                standard = pd.DataFrame(standard)
                standard.to_csv('D:/PycharmProjects/APM/LSTM_train.csv')

            else:
                raw_test = pd.read_csv(filename)
                df = pd.DataFrame(raw_test)
                data = []
                data_len = self.seq_length
                interval = self.seq_interval
                for i in range(self.start_point, len(raw_test) - data_len, interval):
                    tmp = []
                    for j in range(0, data_len, 1):
                        value = df['cpu'].values[i + j]
                        if np.isnan(value):
                            value = (df['cpu'].values[i + j + 1] + df['cpu'].values[i + j - 1]) / 2
                        elif np.isinf(value):
                            value = float(value)
                        tmp.append(value)
                    data.append(tmp)

                mu = np.mean(data, axis=1)
                sigma = np.std(data, axis=1)
                standard = []
                for i in range(len(data)):
                    temp = (np.array(data[i]) - mu[i]) / sigma[i]
                    standard.append(temp)
                standard = pd.DataFrame(standard)
                standard.to_csv('D:/PycharmProjects/APM/LSTM_test.csv')

    def sample_partition_double(self, centers_path, sample_path, pred_len, cluster_num, input_len, outputdir, sep_num):
        centers_df = pd.read_csv(centers_path)
        centers_df = centers_df.loc[:,~centers_df.columns.str.contains('^Unnamed')]
        centers_df = centers_df.loc[:,~centers_df.columns.str.contains('index')]
        centers = centers_df.values
        cluster_train_paths = []
        cluster_test_paths = []
        for i in range(cluster_num):
            cluster_train_paths.append(str(sep_num) + '-cluster' + str(i) + '_train.csv')
            cluster_test_paths.append(str(sep_num) + '-cluster' + str(i) + '_test.csv')
        cluster_samples = []
        for i in range(len(centers)):
            cluster_samples.append([])
            cluster_samples[i].append(centers[i])
        samples_df = pd.read_csv(sample_path)
        samples_df = samples_df.loc[:, ~samples_df.columns.str.contains('^Unnamed')]
        samples_df = samples_df.loc[:, ~samples_df.columns.str.contains('index')]
        samples = samples_df.values
        for sample in samples:
            dist = []
            for center in centers:
                dist.append(self.distEclud(sample[:input_len], center))
            for i in range(sep_num):
                index = np.argsort(np.array(dist))[i]
                cluster_samples[index].append(sample)
        for i in range(len(centers)):
            train_length = int(len(cluster_samples[i]) * 0.8)
            train_df = pd.DataFrame(cluster_samples[i][:train_length])
            test_df = pd.DataFrame(cluster_samples[i][train_length:])
            train_df.to_csv(os.path.join(outputdir, cluster_train_paths[i]), index=False)
            test_df.to_csv(os.path.join(outputdir, cluster_test_paths[i]), index=False)

    def sample_partition(self, centers_path, sample_path, pred_len, cluster_num):
        centers_df = pd.read_csv(centers_path)
        centers = centers_df.values

        cluster_train_paths = []
        cluster_test_paths = []
        for i in range(cluster_num):
            cluster_train_paths.append('cluster' + str(i) + '_train.csv')
            cluster_test_paths.append('cluster' + str(i) + '_test.csv')
        cluster_samples = []
        for i in range(len(centers)):
            cluster_samples.append([])
            cluster_samples[i].append(centers[i])

        samples_df = pd.read_csv(sample_path)
        samples = samples_df.values
        for sample in samples:
            dist = []
            for center in centers:
                dist.append(self.distEclud(sample[:-pred_len], center))
            min_index = dist.index(min(dist))
            cluster_samples[min_index].append(sample)
        for i in range(len(centers)):
            train_length = int(len(cluster_samples[i]) * 0.8)
            train_df = pd.DataFrame(cluster_samples[i][:train_length])
            test_df = pd.DataFrame(cluster_samples[i][train_length:])
            print(train_df.head())
            train_df.to_csv(cluster_train_paths[i])
            test_df.to_csv(cluster_test_paths[i])

    def random_sample_partition(self, sample_path, cluster_num, input_len):
        cluster_train_paths = []
        cluster_test_paths = []
        for i in range(cluster_num):
            cluster_train_paths.append('random' + str(i) + '_train.csv')
            cluster_test_paths.append('random' + str(i) + '_test.csv')

        samples_df = pd.read_csv(sample_path)
        samples_df = samples_df.loc[:,~samples_df.columns.str.contains('^Unnamed')]
        samples_df = samples_df.loc[:,~samples_df.columns.str.contains('index')]
        for i in range(cluster_num):
            temp_df = samples_df.sample(frac=0.5)
            temp_values = temp_df.values
            train_length = int(len(temp_values) * 0.8)
            train_df = pd.DataFrame(temp_values[:train_length])
            test_df = pd.DataFrame(temp_values[train_length:])
            train_df.to_csv(cluster_train_paths[i], index=False)
            test_df.to_csv(cluster_test_paths[i], index=False)


if __name__ == '__main__':
    input_len = 12
    seq_train = 30
    pred_len = 10
    num_cluster = 4
    seed = 29
    # dataset = 'google'
    dataset = 'ali'

    time_scale = 'second'
    # time_scale = 'minute'
    # time_scale = 'hour'

    data_type = 'cpu'
    # data_type = 'mem'

    cluster = DataPre(seq_length=input_len + pred_len, seq_train=seq_train, seq_interval=1, start_point=0)
    # cluster.data_partition()
    # cluster.make_sample()
    inputfile_path = '../data/{0}s_data12-10/{1}_{2}_train.csv'.format(time_scale, dataset, data_type)
    centerfile_path = 'centers.csv'
    outputdir = './cluster_data/{0}_{1}_{2}s_{3}/'.format(dataset, data_type, time_scale, num_cluster)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    raw_data = pd.read_csv(inputfile_path)
    # print(raw_data.head())
    df_data = pd.DataFrame(raw_data)
    df_data = df_data.loc[:,~df_data.columns.str.contains('^Unnamed')]
    df_data = df_data.loc[:,~df_data.columns.str.contains('index')]
    # print(df_data.head())
    df_data = df_data.sample(frac=0.5, replace=True, random_state=0, axis=0)
    # print(df_data.head())
    data_raw = df_data.values
    clustering_method = KMeans(num_cluster)
    # clustering_method = Birch(num_cluster)
    clustering_method.fit(data_raw[:,:input_len])
    centers = clustering_method.cluster_centers_
    # print(centers.shape)
    center_df = pd.DataFrame(centers)
    center_df.to_csv(os.path.join(outputdir, centerfile_path), index=False)
    # cluster.random_sample_partition('/27T/resource-simulator/APM/ali_cpu_train.csv', num_cluster, input_len)
    # for i in range(1, 4):
    cluster.sample_partition_double(centerfile_path, inputfile_path, pred_len, num_cluster, input_len, outputdir, sep_num=3)