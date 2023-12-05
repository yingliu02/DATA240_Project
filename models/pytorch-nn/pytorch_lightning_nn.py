import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy, precision, recall
# merge the datasets into a unified dataframe
import pandas as pd
# df = get_merged_data()


class ChurnPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(ChurnPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(p=0.3)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.batch_size = batch_size 

    def forward(self, x):
        x = F.leaky_relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout1(x)
        x = torch.sigmoid(self.layer2(x))
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return [optimizer], [scheduler]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = None

def train_dataloader(X_train, y_train):
    global scaler

    # Standardize data
    scaler = StandardScaler()
    X_train_standard = scaler.fit_transform(X_train)
    X_val_standard = scaler.transform(X_val)

    # Convert data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train_standard, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
    X_val_tensor = torch.tensor(X_val_standard, dtype=torch.float)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    return [DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True), 
        DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)]
    
def test_dataloader(X_test, y_test):
    global scaler
    X_test_standard = scaler.transform(X_test)  # Standardize test data

    # Convert data into PyTorch tensors
    X_test_tensor = torch.tensor(X_test_standard, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

    # Create data loaders
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    return DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)

if __name__ == '__main__':
    merged_raw_data_url = 'https://drive.google.com/file/d/1WDfh8HLYOtUNuhRZqKCScd1qb4l9sqyj/view?usp=sharing'
    merged_raw_data_url = 'https://drive.google.com/uc?id=' + merged_raw_data_url.split('/')[-2]

    df = pd.read_csv(merged_raw_data_url)
    df = df.set_index('msno')
    df.head()

    from sklearn.model_selection import train_test_split
    from imblearn.under_sampling import RandomUnderSampler

    df = pd.read_csv(merged_raw_data_url)
    df.drop('msno', axis=1, inplace=True)
    X = df.drop(['is_churn'], axis=1)
    y = df['is_churn']

    # Split the data into training and testing sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    # Combine X_train and y_train into a single DataFrame for undersampling
    train_data = pd.concat([X_train, y_train], axis=1)

    # Identify the minority class label
    minority_class_label = train_data['is_churn'].value_counts().idxmin()

    # Apply random undersampling on imbalanced target data
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

    # Train and test the model with the original data
    model = ChurnPredictor(input_size=14, hidden_size=20, output_size=1, batch_size=64)
    trainer = pl.Trainer(max_epochs=20)
    [train_data, val_data] = train_dataloader(X_train, y_train)
    trainer.fit(model, train_data, val_data)
    test_data = test_dataloader(X_test, y_test)
    trainer.test(model, dataloaders=test_data)

    # Train and test the model with the resampled data
    model_resampled = ChurnPredictor(input_size=14, hidden_size=20, output_size=1, batch_size=64)
    trainer_resampled = pl.Trainer(max_epochs=20)
    [train_data_resampled, val_data_resampled] = train_dataloader(X_resampled, y_resampled)
    trainer_resampled.fit(model_resampled, train_data_resampled, val_data_resampled)
    test_data_resampled = test_dataloader(X_test, y_test)
    trainer_resampled.test(model_resampled, dataloaders=test_data_resampled)
