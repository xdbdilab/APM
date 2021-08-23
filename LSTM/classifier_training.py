import argparse
import os

import numpy as np
import ray
import torch
import torch.nn as nn
import yaml
from classifier_dataset import classifierDataset
from ray import tune
from ray.tune.integration.torch import (DistributedTrainableCreator,
                                        distributed_checkpoint_dir)
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(2)

# Recurrent neural network (many-to-one)
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
        out = F.softmax(out)
        return out


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.softmax(out)
        return out


# Train the model
def model_train(writer, model, criterion, optimizer, device, dataloader, total_paras):
    experiment_id = total_paras['experiment_id']
    seed = total_paras['seed']
    sequence_length = total_paras['sequence_length']
    num_epochs = total_paras['num_epochs']
    num_learner = total_paras['num_learner']
    #
    torch.manual_seed(seed)
    total_step = len(dataloader)
    correct = 0
    total = 0
    for epoch in range(num_epochs):
        for i, (input, target) in enumerate(dataloader):
            # input = input.reshape(-1, sequence_length, 1).float().to(device)
            input = input.reshape(-1, sequence_length).float().to(device)
            target = target.reshape(-1, num_learner).float().to(device)
       
            labels = target.gt(0)
            # Forward pass
            outputs = model(input)
            labels_pred = torch.argmax(outputs, dim=1)
            # print(labels_pred)
            # print(labels)
            # print(outputs.size())
            # print(target.size())
            loss = criterion(outputs, target)
            for y_hat, y in zip(labels_pred, labels):
                if y[y_hat]:
                    correct += 1
            total += labels_pred.size(0)
            # correct += (labels_pred == labels).sum().item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Accuracy: {:.8f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, correct/total))
                writer.add_scalar('acc_train_loss', correct/total, epoch * total_step + i + 1)
                total = 0
                correct = 0
        torch.save(model.state_dict(), 'runs/' + experiment_id + '/' + 'model.ckpt')


# Test the model

def model_test(writer, model, criterion, device, dataloader, total_paras):
    experiment_id = total_paras['experiment_id']
    seed = total_paras['seed']
    sequence_length = total_paras['sequence_length']
    input_size = total_paras['input_size']
    num_learner = total_paras['num_learner']
    torch.manual_seed(seed)
    total = 0
    correct = 0
    with torch.no_grad():
        loss_list = []
        for input, target in dataloader:
            # input = input.reshape(-1, sequence_length, input_size).float().to(device)
            input = input.reshape(-1, sequence_length).float().to(device)
            target = target.reshape(-1, num_learner).float().to(device)
            labels = target.gt(0)
            outputs = model(input)
            labels_pred = torch.argmax(outputs, dim=1)
            # print(labels_pred)
            loss = criterion(outputs, target)
            loss_list.append(loss.item())
            for y_hat, y in zip(labels_pred, labels):
                if y[y_hat]:
                    correct += 1
            total += labels_pred.size(0)

            # writer.add_scalar('mae_test_loss', loss.item(), len(loss_list))
        loss_list = np.array(loss_list)
        print('Test MAE of the model : {} %'.format(correct/total))

    # Save the model checkpoint
    writer.close()
    return correct/total


def train_lstm(config, checkpoint_dir=False):
    with open('classifier_training_paras.yaml', 'r') as spec_file:
        para_string = spec_file.read()
        total_paras = yaml.load(para_string, Loader=yaml.FullLoader)

    experiment_id = total_paras['experiment_id']
    seed = total_paras['seed']
    sequence_length = total_paras['sequence_length']
    # pred_step = total_paras['pred_step']
    input_size = total_paras['input_size']
    hidden_size = total_paras['hidden_size']
    num_layers = total_paras['num_layers']
    num_classes = total_paras['num_classes']
    batch_size = total_paras['batch_size']
    num_epochs = total_paras['num_epochs']
    learning_rate = total_paras['learning_rate']
    weight_decay = total_paras['weight_decay']
    train_csv_path = total_paras['train_csv_path']
    test_csv_path = total_paras['test_csv_path']
    num_learner = total_paras['num_learner']

    writer = SummaryWriter('runs/' + experiment_id)

    train_dataset = classifierDataset(csv_path=train_csv_path, input_len=sequence_length,
                                num_learner=num_learner)

    test_dataset = classifierDataset(csv_path=test_csv_path, input_len=sequence_length,
                               num_learner=num_learner)
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)
    # model = RNN(input_size, config['hidden_size'], config['num_layers'], num_classes).to(device)
    model = NeuralNet(input_size, config['hidden_size'], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
            model_state, optimizer_state = torch.load(f)

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    criterion = nn.BCEWithLogitsLoss()
    model = DistributedDataParallel(model)
    for epoch in range(1000):
        model_train(writer, model, criterion, optimizer, device, train_loader, total_paras)
        acc = model_test(writer, model, criterion, device, test_loader, total_paras)
        # Set this to run Tune.
        if epoch % 3 == 0:
            with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(acc=acc)


def tuning():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=4,
        help="Sets number of workers for training.")
    parser.add_argument(
        "--num-gpus-per-worker",
        action="store_true",
        default=1,
        help="enables CUDA training")
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=False,
        help="enables multi-node tuning")
    parser.add_argument(
        "--workers-per-node",
        type=int,
        default=4,
        help="Forces workers to be colocated on machines if set.")
    args = parser.parse_args()
    if args.cluster:
        options = dict(address="auto")
    else:
        options = dict(num_cpus=2)
    ray.init(**options)

    # for early stopping
    trainable_cls = DistributedTrainableCreator(
        train_lstm,
        num_workers=args.num_workers,
        num_gpus_per_worker=args.num_gpus_per_worker,
        num_workers_per_host=args.workers_per_node,
        num_cpus_per_worker=0)
    analysis = tune.run(
        trainable_cls,
        metric="acc",
        mode="max",
        stop={
            "acc": 0.80,
            "training_iteration": 30  
        },
        num_samples=1,
        config={
            'hidden_size': tune.choice([64, 128, 256]),
            "learning_rate": tune.loguniform(1e-6, 1e-3),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
        })

    print("Best config is:", analysis.best_config)


def train_main():
    with open('classifier_training_paras.yaml', 'r') as spec_file:
        para_string = spec_file.read()
        total_paras = yaml.load(para_string, Loader=yaml.FullLoader)

    experiment_id = total_paras['experiment_id']
    seed = total_paras['seed']
    sequence_length = total_paras['sequence_length']
    num_learner = total_paras['num_learner']
    input_size = total_paras['input_size']
    hidden_size = total_paras['hidden_size']
    num_layers = total_paras['num_layers']
    num_classes = total_paras['num_classes']
    batch_size = total_paras['batch_size']
    num_epochs = total_paras['num_epochs']
    learning_rate = total_paras['learning_rate']
    weight_decay = total_paras['weight_decay']
    train_csv_path = total_paras['train_csv_path']
    test_csv_path = total_paras['test_csv_path']

    writer = SummaryWriter('runs/' + experiment_id)
    with open('runs/' + experiment_id + '/' + 'naive_LSTM_paras.yaml', 'w') as f:
        yaml.dump(total_paras, f)

    # MNIST dataset
    train_dataset = classifierDataset(csv_path=train_csv_path, input_len=sequence_length,
                                num_learner=num_learner)

    test_dataset = classifierDataset(csv_path=test_csv_path, input_len=sequence_length,
                               num_learner=num_learner)
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)

    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for i, p in enumerate(list(model.state_dict())):
        print(i, p)
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()

    model_train(writer, model, criterion, optimizer, device, train_loader, total_paras)
    model_test(writer,  model, criterion, device, test_loader, total_paras)
    torch.save(model.state_dict(), 'runs/' + experiment_id + '/' + 'model.ckpt')


if __name__ == '__main__':
    # train_main()
    tuning()
