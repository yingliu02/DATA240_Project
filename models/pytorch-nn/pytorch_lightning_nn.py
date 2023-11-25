import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy, precision, recall

class AdvancedChurnPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(AdvancedChurnPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(p=0.3)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(p=0.3)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size 

    def forward(self, x):
        x = F.leaky_relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.layer3(x))
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_pred = torch.round(y_hat)
        y_true = y.int()
        acc = accuracy(y_pred, y_true, 'binary')
        prec = precision(y_pred, y_true, num_classes=2, average='macro', task='binary')
        rec = recall(y_pred, y_true, num_classes=2, average='macro', task='binary')
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_prec', prec)
        self.log('val_rec', rec)
        result = {'val_loss': loss, 'val_acc': acc, 'val_prec': prec, 'val_rec': rec}
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_pred = torch.round(y_hat)
        y_true = y.int()
        acc = accuracy(y_pred, y_true, 'binary')
        prec = precision(y_pred, y_true, num_classes=2, average='macro', task='binary')
        rec = recall(y_pred, y_true, num_classes=2, average='macro', task='binary')
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_prec', prec)
        self.log('test_rec', rec)
        result = {'test_loss': loss, 'test_acc': acc, 'test_prec': prec, 'test_rec': rec}
        return result

    # def on_validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
    #     avg_prec = torch.stack([x['val_prec'] for x in outputs]).mean()
    #     avg_rec = torch.stack([x['val_rec'] for x in outputs]).mean()
    #     self.log('avg_val_loss', avg_loss)
    #     self.log('avg_val_acc', avg_acc)
    #     self.log('avg_val_prec', avg_prec)
    #     self.log('avg_val_rec', avg_rec)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return [optimizer], [scheduler]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def split_data():
    # Load data
    data = pd.read_csv('../../Datasets/merged_data.csv')

    # Preprocess data
    X = data.drop(['is_churn', 'msno'], axis=1)
    # convert gender to 1, 0 and -1
    X['gender'] = X['gender'].replace(['female', 'male'], [0, 1]).fillna(-1).astype(int)
    y = data['is_churn']
    
    # Split data into training, validation and testing sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    return X_train, y_train, X_val, y_val, X_test, y_test

scaler = None

def train_dataloader():
    global scaler
    X_train, y_train, X_val, y_val, _, _ = split_data()

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    return [DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True), 
        DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)]
    
def test_dataloader():
    global scaler
    _, _, _, _, X_test, y_test = split_data()
    X_test = scaler.transform(X_test)  # Standardize test data

    # Convert data into PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

    # Create data loaders
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    return DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    
if __name__ == '__main__':
    model = AdvancedChurnPredictor(input_size=14, hidden_size=20, output_size=1, batch_size=64)
    trainer = pl.Trainer(max_epochs=20)
    [train_data, val_data] = train_dataloader()
    trainer.fit(model, train_data, val_data)
    test_data = test_dataloader()
    trainer.test(model, dataloaders=test_data)


# if __name__ == '__main__':
#     model = AdvancedChurnPredictor(input_size=14, hidden_size=20, output_size=1, batch_size=64)
#     trainer = pl.Trainer(max_epochs=10)
#     [train_data, val_data] = train_dataloader()
#     trainer.fit(model, train_data, val_data)
