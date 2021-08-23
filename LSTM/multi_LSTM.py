import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from LSTM_dataset import lstmDataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)

        # Forward propagate LSTM
        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x, h0)
        
  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def train(writer, model, criterion, device, dataloader, total_paras):
    experiment_id = total_paras['experiment_id']
    seed = total_paras['seed']
    sequence_length = total_paras['sequence_length']
    learning_rate = total_paras['learning_rate']
    weight_decay = total_paras['weight_decay']
    num_epochs = total_paras['num_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    torch.manual_seed(seed)
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        for i, (input, target) in enumerate(dataloader):
            input = input.reshape(-1, sequence_length, 1).float().to(device)
            target = target.reshape(-1, 1).float().to(device)
            # Forward pass
            outputs = model(input)
            loss = criterion(outputs, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                writer.add_scalar('mae_train_loss', loss.item(), epoch * total_step + i + 1)
        torch.save(model.state_dict(), 'runs/' + experiment_id + '/' + 'model.ckpt')


def sample_redivide(total_paras, cluster_id, model_paths):
    total_loss_list = []
    experiment_id = total_paras['experiment_id']
    seed = total_paras['seed']
    sequence_length = total_paras['sequence_length']
    pred_step = total_paras['pred_step']
    input_size = total_paras['input_size']
    hidden_size = total_paras['hidden_size']
    num_layers = total_paras['num_layers']
    num_classes = total_paras['num_classes']
    batch_size = total_paras['batch_size']
    num_epochs = total_paras['num_epochs']
    learning_rate = total_paras['learning_rate']
    weight_decay = total_paras['weight_decay']
    input_csv_director = total_paras['input_csv_director']
    models = []
    revised_cluster_samples = []
    for model_path in model_paths:
        model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        models.append(model)
        revised_cluster_samples.append([])
    train_path = input_csv_director + cluster_id + '_train.csv'
    test_path = input_csv_director + cluster_id + '_test.csv'
    train_dataset = lstmDataset(csv_path=train_path, input_len=sequence_length,
                                pred_step=pred_step)
    test_dataset = lstmDataset(csv_path=test_path, input_len=sequence_length,
                               pred_step=pred_step)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)
    criterion = nn.L1Loss()
    sample_list = []
    for i, (input, target) in enumerate(train_loader):
        output_list = []
        loss_list = []
        sample = input.view(-1).detach().numpy()
        input = input.reshape(-1, sequence_length, input_size).float().to(device)
        target = target.reshape(-1, num_classes).float().to(device)
        for model in models:
            # Forward pass
            output = model(input)
            output_list.append(output)
            loss = criterion(output, target)
            loss_list.append(loss.item())
        min_loss = np.sort(np.array(loss_list))[0] - 0.00000001
        min_index = np.argsort(np.array(loss_list))[0]
        weights = [0, 0, 0, 0]
        weights[min_index] = 1
        z = 0
        for i in range(len(loss_list)):
            eps = loss_list[i] - min_loss
            if eps < 0.01:
                weights[i] = 1 / eps
                z = z + 1 / eps
        weights = np.array(weights)
        # print(weights)
        if z > 0:
            weights = weights / z
        sample = np.append(loss_list, target.cpu().detach().numpy())
        # print(sample)
        sample_list.append(sample)
        y_pred = 0
        for i in range(len(weights)):
            y_pred = y_pred + weights[i] * output_list[i]
        total_loss_list.append(criterion(y_pred, target).detach().float().item())
    x = sample_list[:, : -1]
    y = sample_list[:, -1]
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    df = pd.DataFrame(np.array(sample_list))
    df.to_csv(input_csv_director + 'sampleLabels_train.csv', index=False)

    sample_list = []
    for i, (input, target) in enumerate(train_loader):
        output_list = []
        loss_list = []
        sample = input.view(-1).detach().numpy()
        input = input.reshape(-1, sequence_length, input_size).float().to(device)
        target = target.reshape(-1, num_classes).float().to(device)
        for model in models:
            # Forward pass
            output = model(input)
            output_list.append(output)
            loss = criterion(output, target)
            loss_list.append(loss.item())
        min_loss = np.sort(np.array(loss_list))[0] - 0.00000001
        min_index = np.argsort(np.array(loss_list))[0]
        weights = [0, 0, 0, 0]
        weights[min_index] = 1
        z = 0
        for i in range(len(loss_list)):
            eps = loss_list[i] - min_loss
            if eps < 0.01:
                weights[i] = 1 / eps
                z = z + 1 / eps
        weights = np.array(weights)
        if z > 0:
            weights = weights / z
        sample = np.append(loss_list, target.cpu().detach().numpy())
        sample_list.append(sample)
        y_pred = 0
        for i in range(len(weights)):
            y_pred = y_pred + weights[i] * output_list[i]
        total_loss_list.append(criterion(y_pred, target).detach().float().item())
    x = sample_list[:, : -1]
    y = sample_list[:, -1]
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    df = pd.DataFrame(np.array(sample_list))
    df.to_csv(input_csv_director + 'sampleLabels_test.csv', index=False)


def learner_lstm_train(experiment_id, model_path, cluster_id, total_paras):
    writer = SummaryWriter('runs/' + experiment_id + '/' + cluster_id)
    with open('runs/' + experiment_id + '/' + 'naive_LSTM_paras.yaml', 'w') as f:
        yaml.dump(total_paras, f)
    experiment_id = total_paras['experiment_id']
    seed = total_paras['seed']
    sequence_length = total_paras['sequence_length']
    pred_step = total_paras['pred_step']
    input_size = total_paras['input_size']
    hidden_size = total_paras['hidden_size']
    num_layers = total_paras['num_layers']
    num_classes = total_paras['num_classes']
    batch_size = total_paras['batch_size']
    num_epochs = total_paras['num_epochs']
    learning_rate = total_paras['learning_rate']
    weight_decay = total_paras['weight_decay']
    input_csv_director = total_paras['input_csv_director']

    train_path = input_csv_director + cluster_id + '_train.csv'
    test_path = input_csv_director + cluster_id + '_test.csv'
    torch.manual_seed(seed)
    train_dataset = lstmDataset(csv_path=train_path, input_len=sequence_length,
                                pred_step=pred_step)

    test_dataset = lstmDataset(csv_path=test_path, input_len=sequence_length,
                               pred_step=pred_step)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)

    # Recurrent neural network (many-to-one)
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

    # model.load_state_dict(torch.load(model_path))
    # model = torch.load(model_path)
    # for i, p in enumerate(model.state_dict()):
    #     print(p)
    # for i, p in enumerate(model.parameters()):
    #     if i < 4:
    #         p.requires_grad = False

    # Loss and optimizer
    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (input, target) in enumerate(train_loader):
            input = input.reshape(-1, sequence_length, input_size).float().to(device)
            # print(input)
            target = target.reshape(-1, num_classes).float().to(device)
            # Forward pass
            outputs = model(input)
            loss = criterion(outputs, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()

            if (i + 1) % 100 == 0:
                # print('{} Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                #     .format(cluster_id, epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                writer.add_scalar('mae_train_loss', loss.item(), epoch * total_step + i + 1)

    # Test the model
    with torch.no_grad():
        loss_list = []
        for i, (input, target) in enumerate(test_loader):
            input = input.reshape(-1, sequence_length, input_size).float().to(device)
            target = target.reshape(-1, num_classes).float().to(device)
            outputs = model(input)
            loss = criterion(outputs, target)
            loss_list.append(loss.item())
            writer.add_scalar('mae_test_loss', loss.item(), len(loss_list))
        loss_list = np.array(loss_list)
        print('{} Test MAE of the model : {}'.format(cluster_id, np.mean(loss_list)))

    # Save the model checkpoint
    writer.close()
    torch.save(model.state_dict(), experiment_id + '_' + cluster_id + '_model.ckpt')


def main():
    with open('multi_LSTM.yaml', 'r') as spec_file:
        para_string = spec_file.read()
        total_paras = yaml.load(para_string, Loader=yaml.FullLoader)

    experiment_id = total_paras['experiment_id']
    cluster_type = total_paras['cluster_type']
    cluster_num = total_paras['cluster_num']
    pretrained_model_path = total_paras['pretrained_model_path']
    model_paths = []
    for i in range(cluster_num):
        cluster_id = cluster_type + str(i)
        model_paths.append(str(experiment_id + '_' + cluster_id + '_model.ckpt'))

    for i in range(cluster_num):
        cluster_id = cluster_type + str(i)
        learner_lstm_train(experiment_id, pretrained_model_path, cluster_id, total_paras)
        sample_redivide(cluster_id, model_paths)

main()
