import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

        # Weight initialization
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)

        # initialize bias with zeros
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
        init.zeros_(self.fc4.bias)

    def forward(self, x):
        # linear -> relu -> dropout -> batchnorm
        x = self.bn1(self.dropout1(F.relu(self.fc1(x))))
        x = self.bn2(self.dropout2(F.relu(self.fc2(x))))
        x = self.bn3(self.dropout3(F.relu(self.fc3(x))))
        x = self.fc4(x)  # Output layer
        return x


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_size=1, n_classes=10, hidden_size=[50]):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_size, out_channels=10, kernel_size=5, stride=1, padding=0
        )
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0
        )
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_dropout = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(320, hidden_size[0])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size[0], n_classes)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class EarlyStopper:
    def __init__(
        self, patience: int = 1, min_delta: float = 0.0, verbose: bool = False
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_weights = None
        self.verbose = verbose

    def early_stop(self, loss:float, model:nn.Module):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.save_best_weights(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early Stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                return True
        return False

    def save_best_weights(self, model):
        self.best_model_weights = deepcopy(model.state_dict())

    def restore_best_weights(self, model):
        model.load_state_dict(self.best_model_weights)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def train_regression(model, data_loader, optimizer, criterion, device):
    model.to(device).train()
    batch_loss = []

    for x_train, y_train in data_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)

        optimizer.zero_grad()

        y_hat = model(x_train)
        loss = criterion(y_hat, y_train)
        loss.backward()

        optimizer.step()

        batch_loss.append(loss.item())

    loss_total = sum(batch_loss) / len(batch_loss)

    return loss_total


def validate_regression(model, data_loader, criterion, device):
    model.to(device).eval()
    batch_loss = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            batch_loss.append(loss.item())
    loss_total = sum(batch_loss) / len(batch_loss)

    return loss_total


def predict_regression(model, data_loader, device):
    model.to(device).eval()
    predictions = []

    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch.to(device)
            y_hat = model(x_batch)
            predictions.append(y_hat)

    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    return predictions


# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(
    x_train.values,
    dtype=torch.float32,
).to(device)
x_val_tensor = torch.tensor(
    x_val.values,
    dtype=torch.float32,
).to(device)
x_test_tensor = torch.tensor(
    x_test.values,
    dtype=torch.float32,
).to(device)
x_pred_tensor = torch.tensor(
    x_pred_temp.values,
    dtype=torch.float32,
).to(device)
y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32,
).to(device)
y_val_tensor = torch.tensor(
    y_val.values,
    dtype=torch.float32,
).to(device)

# Define PyTorch DataLoader for training
train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
val_dataset = CustomDataset(x_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

train_losses = []
val_losses = []

input_dim = x_train.shape[1]
n_epochs = 100
hidden_dim = 64
patience = 10

model = NeuralNetwork(input_dim, hidden_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.1,
)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
)

early_stopper = EarlyStopper(patience=patience)

# Training loop
epochs = 700
for epoch in range(epochs):
    # Training
    train_loss = train_regression(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
    )
    train_losses.append(train_loss)

    # Validation
    val_loss = validate_regression(
        model,
        val_loader,
        criterion,
        device,
    )
    val_losses.append(val_loss)

    # Adjust learning rate
    scheduler.step(epoch)

    # Early Stopping
    if early_stopper.early_stop(val_loss, model):
        n_epochs.append(epoch)
        break
early_stopper.restore_best_weights(model)

# Prediction
y_train_hat = model(x_train_tensor).cpu().detach().numpy()
y_val_hat = model(x_val_tensor).cpu().detach().numpy()
y_test_hat = model(x_test_tensor).cpu().detach().numpy()
y_pred_hat = model(x_pred_tensor).cpu().detach().numpy()
