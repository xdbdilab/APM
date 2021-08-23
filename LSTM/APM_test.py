import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from classifier_dataset import classifierDataset

from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from LSTM_dataset import lstmDataset
from tensorboardX import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(2)

class disc_network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(disc_network, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = F.softmax(out)
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def main():
    with open('APM_test_paras.yaml', 'r') as spec_file:
        para_string = spec_file.read()
        total_paras = yaml.load(para_string, Loader=yaml.FullLoader)

    test_id = total_paras['test_id']
    seed = total_paras['seed']
    cluster_type = total_paras['cluster_type']
    cluster_num = total_paras['cluster_num']
    test_path = total_paras['test_csv_path']
    sequence_length = total_paras['sequence_length']
    pred_step = total_paras['pred_step']
    input_size = total_paras['input_size']
    disc_id = total_paras['disc_id']
    disc_hidden_size = total_paras['disc_hidden_size']
    disc_num_layers = total_paras['disc_num_layers']
    model_id = total_paras['model_id']
    model_hidden_size = total_paras['model_hidden_size']
    model_num_layers = total_paras['model_num_layers']

    disc_path = str('./runs/' + disc_id + '/' + 'model.ckpt')
    model_paths = []
    for i in range(cluster_num):
        cluster_id = cluster_type + str(i)
        model_paths.append(
            str('./runs/' + model_id + '_' + cluster_id + '_model.ckpt'))
    models = []
    torch.manual_seed(seed)
    for model_path in model_paths:
        model = RNN(input_size, model_hidden_size, model_num_layers, 1).to(device)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        models.append(model)
    disc = disc_network(input_size, disc_hidden_size, disc_num_layers, cluster_num).to(device)
    disc.load_state_dict(torch.load(disc_path, map_location='cpu'))

    test_dataset = lstmDataset(csv_path=test_path, input_len=sequence_length,
                               pred_step=pred_step)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)
    criterion = nn.L1Loss()
    with torch.no_grad():
        loss_list = []
        for i, (input, target) in enumerate(test_loader):
            output_list = []
            input = input.reshape(-1, sequence_length, input_size).float().to(device)
            target = target.reshape(-1, 1).float().to(device)
            weights = disc(input)
            for model in models:
                # Forward pass
                output = model(input)
                # print(output)
                output_list.append(output.item())
            output = weights * np.array(output_list)
            loss = criterion(output, target)
            loss_list.append(loss)
        print('test ID: {0}, MAE: {1}'.format(test_id, sum(loss_list) / len(loss_list)))

main()
