import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from datetime import date
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def SMAPE(outputs, targets):
    smape = (100 / len(targets)) * np.sum(np.abs(outputs - targets) / ((np.abs(targets) + np.abs(outputs)) / 2))
    return smape


def convert_datetime(row):
    date_ = date.fromisoformat(row['date'])
    day = date_.day
    if date_.weekday() == 5:
        row['saturday'] = 1
    elif date_.weekday() == 6:
        row['sunday'] = 1
    if date_.month in [12, 1, 2]:
        row['winter'] = 1
    elif date_.month in [3, 4, 5]:
        row['spring'] = 1
    elif date_.month in [6, 7, 8]:
        row['summer'] = 1
    elif date_.month in [9, 10, 11]:
        row['autumn'] = 1
    row['sin_date'] = np.sin(2 * np.pi * (day / 365))
    row['cos_date'] = np.cos(2 * np.pi * (day / 365))
    return row


def convert_country(row):
    if row['country'] == 'Finland':
        row['finland'] = 1
    elif row['country'] == 'Norway':
        row['norway'] = 1
    elif row['country'] == 'Sweden':
        row['sweden'] = 1
    return row


def convert_product(row):
    if row['product'] == 'Kaggle Mug':
        row['mug'] = 1
    elif row['product'] == 'Kaggle Hat':
        row['hat'] = 1
    elif row['product'] == 'Kaggle Sticker':
        row['sticker'] = 1
    return row


def convert_store(row):
    if row['store'] == 'KaggleMart':
        row['store'] = 1
    else:
        row['store'] = 0
    return row


class StoresDataset(Dataset):
    def __init__(self, file, set_type, truncated=False, split=0.8):
        self.set_type = set_type
        print(f"Initializing {self.set_type} dataset, this may take a moment...")
        self.df = pd.read_csv(file)
        if self.set_type != "eval":
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        if truncated:
            self.df = self.df.truncate(after=5000)
        if self.set_type == "training":
            self.df = self.df.truncate(after=np.floor(len(self.df) * split))
        elif self.set_type == "test":
            self.df = self.df.truncate(before=np.floor(len(self.df) * split))
        if self.set_type != "eval":
            self.data = self.df.drop(columns=['row_id', 'num_sold'])
        else:
            self.data = self.df.drop(columns=['row_id'])
        self.data = self.data.apply(convert_datetime, axis=1)
        self.data = self.data.apply(convert_country, axis=1)
        self.data = self.data.apply(convert_product, axis=1)
        self.data = self.data.apply(convert_store, axis=1)
        self.data = self.data.drop(columns=['date', 'country', 'product'], axis=1)
        self.data = self.data.to_numpy(dtype="float32")
        if self.set_type != "eval":
            self.targets = self.df['num_sold']
            self.targets = self.targets.to_numpy(dtype='float32')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        data = torch.from_numpy(self.data[item])
        data = torch.nan_to_num(data)
        if self.set_type != "eval":
            target = torch.tensor(self.targets[item].reshape(-1))
            return data, target
        else:
            return data, 0


class StoresModel(nn.Module):
    def __init__(self, D):
        super(StoresModel, self).__init__()
        self.D = D
        self.linear = nn.Sequential(
            nn.Linear(self.D, 8192),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8192, 1),
        )

    def forward(self, X):
        input_ = X.to(device)
        out = self.linear(input_)
        return out


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
            num_sales.append(int(output))
    submission['row_id'] = row_ids
    submission['num_sold'] = num_sales
    submission.to_csv(filename, index=False)
    print(f"Saved submission to {filename}.")


plt.rcParams["figure.figsize"] = (16, 8)

train_dataset = StoresDataset('train.csv', set_type='training')
test_dataset = StoresDataset('train.csv', set_type='test')

batch_sz = 2000
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
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

train(model, device, criterion, optimizer, baseline_rmse, train_loader, test_loader, epochs=25)

grade(model, device, baseline_rmse, train_loader, test_loader, bins_range=500)

eval_dataset = StoresDataset('test.csv', set_type='eval')
eval_loader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)

generate_submission(model, device, eval_loader, 'submission.csv')
