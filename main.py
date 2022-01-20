import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from preprocessing import preprocess


def SMAPE(outputs, targets):
    smape = (100 / len(targets)) * np.sum(np.abs(outputs - targets) / ((np.abs(targets) + np.abs(outputs)) / 2))
    return smape


class StoresDataset(Dataset):
    def __init__(self, set_type, split=0.8):
        self.scaler = StandardScaler()
        self.column_transformer = make_column_transformer((self.scaler, [6, 10, 15, 22]), remainder="passthrough")
        self.set_type = set_type
        print(f"Initializing {self.set_type} dataset")
        if set_type != "eval":
            self.data = np.load("train_data_array.npy")
            self.data = self.column_transformer.fit_transform(self.data)
            split_idx = int(split * len(self.data))
            self.targets = np.load("train_targets_array.npy")
            self.training_targets = self.targets[:split_idx]
            self.test_targets = self.targets[split_idx:]
            self.training_data = self.data[:split_idx]
            self.test_data = self.data[split_idx:]
        else:
            self.data = np.load(f"test_data_array.npy")
            self.data = self.column_transformer.fit_transform(self.data)
            self.eval_data = self.data

    def __len__(self):
        if self.set_type == "train":
            return len(self.training_data)
        elif self.set_type == "test":
            return len(self.test_data)
        else:
            return len(self.eval_data)

    def __getitem__(self, item):
        if self.set_type != "eval":
            if self.set_type == "train":
                data = torch.tensor(self.training_data[item])
                data = torch.nan_to_num(data)
                target = torch.tensor(self.training_targets[item].reshape(-1))
                return data, target
            else:
                data = torch.tensor(self.test_data[item])
                data = torch.nan_to_num(data)
                target = torch.tensor(self.test_targets[item].reshape(-1))
                return data, target
        else:
            data = torch.tensor(self.eval_data[item])
            data = torch.nan_to_num(data)
            return data, 0


class StoresModel(nn.Module):
    def __init__(self, D):
        super(StoresModel, self).__init__()
        self.D = D
        self.linear = nn.Sequential(
            nn.Linear(self.D, 8192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8192, 12280),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(12280, 8192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),
        )

    def forward(self, X):
        input_ = X.to(device)
        out = self.linear(input_)
        return out

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()


def train(model, device, criterion, optimizer, baseline_rmse, train_loader, test_loader, epochs):
    train_losses = []
    test_losses = []
    epochs = epochs
    t0 = datetime.now()
    print("Starting training...")
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}.")
        model.train()
        train_loss = []
        batch = 0
        for inputs, targets in train_loader:
            batch += 1
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            l1_weight = 0.05
            l1_parameters = []
            for parameter in model.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))

            loss += l1

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_losses.append(np.mean(train_loss))

        batch = 0
        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            batch += 1
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())

        test_losses.append(np.mean(test_loss))
        dt = datetime.now() - t0
        train_epoch_loss = train_losses[-1]
        test_epoch_loss = test_losses[-1]
        print(f"Epoch:          {epoch + 1}/{epochs}\n"
              f"Train loss:     {train_epoch_loss:.4f} (root {np.sqrt(train_epoch_loss):.4f})\n"
              f"Baseline diff:  {np.sqrt(train_epoch_loss) - baseline_rmse:.4f}\n"
              f"Test loss:      {test_epoch_loss:.4f} (root {np.sqrt(test_epoch_loss):.4f})\n"
              f"Baseline diff:  {np.sqrt(test_epoch_loss) - baseline_rmse:.4f}\n"
              f"Total duration: {dt}")

    plt.plot(train_losses, label="Train losses")
    plt.plot(test_losses, label="Test losses")
    plt.show()


def grade(model, device, baseline_rmse, train_loader, test_loader, bins_range):
    print("Starting grading...")
    with torch.no_grad():
        train_outputs = []
        train_targets = []
        model.train()
        batch = 0
        for inputs, targets in train_loader:
            batch += 1
            targets = targets.to(device)
            outputs = model(inputs).cpu().numpy().flatten().tolist()
            train_targets += targets.cpu().numpy().flatten().tolist()
            train_outputs += outputs

        train_outputs = np.array(train_outputs)
        train_targets = np.array(train_targets)
        train_diff = train_targets - train_outputs

        train_rmse = np.sqrt(((train_targets - train_outputs) ** 2).mean())

        test_outputs = []
        test_targets = []
        model.eval()
        batch = 0
        for inputs, targets in test_loader:
            batch += 1
            targets = targets.to(device)
            outputs = model(inputs).cpu().numpy().flatten().tolist()
            test_targets += targets.cpu().numpy().flatten().tolist()
            test_outputs += outputs

        test_outputs = np.array(test_outputs)
        test_targets = np.array(test_targets)
        test_diff = test_targets - test_outputs

        fig, axs = plt.subplots(1, 2)
        axs[0].hist(train_diff, bins=range(-bins_range, bins_range), color='blue')
        axs[0].set_title('Training diff')
        axs[1].hist(test_diff, bins=range(-bins_range, bins_range), color='orange')
        axs[1].set_title('Test diff')
        plt.show()

        test_smape = SMAPE(test_outputs, test_targets)
        test_rmse = np.sqrt(((test_targets - test_outputs) ** 2).mean())
        print(f"Train RMSE: {train_rmse:.4f}, baseline diff: {train_rmse - baseline_rmse:.4f}\n"
              f"Test RMSE:  {test_rmse:.4f}, baseline diff: {test_rmse - baseline_rmse:.4f}\n"
              f"Test SMAPE: {test_smape:.2f}%")


def generate_submission(model, device, eval_loader, filename, start_idx=26298):
    submission = pd.DataFrame(columns=['row_id', 'num_sold'])
    with torch.no_grad():
        model.eval()
        i = start_idx
        row_ids = []
        num_sales = []
        for input_, _ in eval_loader:
            input_ = input_.to(device)
            output = model(input_).cpu().numpy().flatten()
            row_ids.append(i)
            i += 1
            num_sales.append(int(np.round(output)))
    submission['row_id'] = row_ids
    submission['num_sold'] = num_sales
    submission.to_csv(filename, index=False)
    print(f"Saved submission to {filename}.")


plt.rcParams["figure.figsize"] = (16, 8)

# preprocess()

train_dataset = StoresDataset(set_type='train')
test_dataset = StoresDataset(set_type='test')

batch_sz = 1024
train_batches = train_dataset.__len__() / batch_sz
test_batches = test_dataset.__len__() / batch_sz
D = len(train_dataset.data[0])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=False)

baseline_rmse = 200.89605266688307

model = StoresModel(D)
device = torch.device("cuda:0")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=8e-5, weight_decay=1e-3)

train(model, device, criterion, optimizer, baseline_rmse, train_loader, test_loader, epochs=80)

grade(model, device, baseline_rmse, train_loader, test_loader, bins_range=500)

eval_dataset = StoresDataset(set_type='eval')
eval_loader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)

df = pd.read_csv('test.csv')
start_idx = df['row_id'][0]

generate_submission(model, device, eval_loader, 'submission.csv', start_idx=start_idx)
