from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torchmetrics import AUROC
import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy, precision, recall
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.integration import PyTorchLightningPruningCallback

class EnhancedChurnPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, batch_size, max_epochs, lr, weight_decay, dropout_rate):
        super(EnhancedChurnPredictor, self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.auroc = AUROC(task='binary')

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(self.output_layer(x))
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)

        # Compute and log accuracy
        y_pred = torch.round(y_hat)
        y_true = y.int()
        acc = accuracy(y_pred, y_true, 'binary')
        self.log('train_acc', acc)
        
         # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_pred = torch.round(y_hat)
        y_true = y.int()
        acc = accuracy(y_pred, y_true, 'binary')
        prec = precision(y_pred, y_true, num_classes=2, average='macro', task='binary')
        rec = recall(y_pred, y_true, num_classes=2, average='macro', task='binary')
        loss = F.binary_cross_entropy(y_hat, y)
        auroc_score = self.auroc(y_hat, y_true)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_prec', prec)
        self.log('test_rec', rec)
        self.log('test_auroc', auroc_score)
        result = {'test_loss': loss, 'test_acc': acc, 'test_prec': prec, 'test_rec': rec, 'test_auroc': auroc_score}
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        
        # Compute metrics
        y_pred = torch.round(y_hat)
        y_true = y.int()
        acc = accuracy(y_pred, y_true, 'binary')
        prec = precision(y_pred, y_true, num_classes=2, average='macro', task='binary')
        rec = recall(y_pred, y_true, num_classes=2, average='macro', task='binary')
        auroc_score = self.auroc(y_hat, y_true)
        
        self.log('val_acc', acc)
        self.log('val_prec', prec)
        self.log('val_rec', rec)
        self.log('val_auroc', auroc_score)
        
        return {'val_loss': loss, 'val_acc': acc, 'val_prec': prec, 'val_rec': rec, 'val_auroc': auroc_score}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader(self.batch_size)), epochs=self.hparams.max_epochs)
        return [optimizer], [scheduler]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = None

def train_dataloader(batch_size):
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
    return [DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True), 
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)]
    
def test_dataloader(batch_size):
    global scaler
    X_test_standard = scaler.transform(X_test)  # Standardize test data

    # Convert data into PyTorch tensors
    X_test_tensor = torch.tensor(X_test_standard, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

    # Create data loaders
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

# Define the StandardScaler instance
scaler = StandardScaler()

def objective(trial):
    # Hyperparameters to be tuned by Optuna.
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Model creation using the suggested hyperparameters.
    model = EnhancedChurnPredictor(
        input_size=14, 
        hidden_size=hidden_size, 
        output_size=1, 
        batch_size=batch_size, 
        max_epochs=5,
        lr=lr,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate
    )
    
    [train_data, val_data] = train_dataloader(batch_size)
    
    # Create the trainer with the current trial's hyperparameters.
    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=0.1, # Use a small portion of validation data for faster experiments.
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
        max_epochs=5
    )
    
    # Execute training and validation.
    trainer.fit(model, train_data, val_data)
    
    # Return the best validation AUROC.
    return trainer.callback_metrics["val_acc"].item()

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

    # # Train and test the model with the original data
    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: objective(trial), n_trials=10)
    # print(study.best_trial)

    # Apply random undersampling on imbalanced target data
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)

    # Train and test the model with the resampled data
    resampled_study = optuna.create_study(direction="maximize")
    resampled_study.optimize(lambda trial: objective(trial), n_trials=10)
    print(resampled_study.best_trial)
