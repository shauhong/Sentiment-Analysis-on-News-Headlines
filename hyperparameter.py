import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import optuna
import pickle
from dataset import News
from embedding import GloVe
from model import BiLSTM
from train import train, evaluate
from utils import load_config

def objective(trial):
    params = {
      'hidden_size': trial.suggest_int('hidden_size', 8, 64, step=8),
      'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
    }

    variant = trial.suggest_categorical('variant', ["glove-wiki-gigaword-50", "glove-wiki-gigaword-100","glove-wiki-gigaword-200", "glove-wiki-gigaword-300"])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    path = "data"

    config = load_config()
    glove = GloVe(variant=variant)

    model = BiLSTM(glove, **config['BiLSTM'], **params)
    model.to(device)
    optimizer = Adam(model.parameters(), **config['ADAM'], weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_dataset = News(path, glove, split='train')
    val_dataset = News(path, glove, split='validation')
    train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, **config['train'])
    val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.collate_fn, **config['validation'])

    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)

    return val_accuracy

def optimize(n_trials, objective, path="assets/hyperparameter.pkl"):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler())
    study.optimize(objective, n_trials=n_trials)
    with open(path, "wb") as f:
        pickle.dump(study.best_trial.params, f)