import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from dataset import News
from embedding import GloVe
from model import BiLSTM, AttentionBiLSTM
from utils import save_history, save_checkpoint, load_checkpoint, load_config


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train(model, train_dataloader, optimizer, criterion=nn.CrossEntropyLoss(), device="cpu"):
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('Accuracy', ':6.2f')
    model.train()

    for sentences, sentiments in train_dataloader:
        sentences, sentiments = sentences.to(device), sentiments.to(device)

        outputs = model(sentences)
        loss = criterion(outputs, sentiments)

        predictions = torch.argmax(
            F.softmax(outputs, dim=-1), dim=-1).detach().cpu().numpy()
        accuracy.update(accuracy_score(
            sentiments.cpu().numpy(), predictions), outputs.size(0))
        losses.update(loss.item(), outputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, accuracy.avg


def evaluate(model, val_dataloader, criterion=nn.CrossEntropyLoss(), device="cpu"):
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('Accuracy', ':6.2f')
    model.eval()

    with torch.no_grad():
        for sentences, sentiments in val_dataloader:
            sentences, sentiments = sentences.to(device), sentiments.to(device)

            outputs = model(sentences)
            loss = criterion(outputs, sentiments)

            predictions = torch.argmax(
                F.softmax(outputs, dim=-1), dim=-1).detach().cpu().numpy()
            accuracy.update(accuracy_score(
                sentiments.cpu().numpy(), predictions), outputs.size(0))
            losses.update(loss.item(), outputs.size(0))

    return losses.avg, accuracy.avg


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "data"
    epochs = 50
    history = {
        'train': {'loss': list(), 'accuracy': list()},
        'val': {'loss': list(), 'accuracy': list()}
    }
    glove = GloVe(**config['GloVe'])
    model = BiLSTM(glove, **config['BiLSTM'])
    model.to(device)
    optimizer = Adam(model.parameters(), **config['ADAM'])
    criterion = nn.CrossEntropyLoss()
    train_dataset = News(path, glove, split='train')
    val_dataset = News(path, glove, split='validation')
    train_dataloader = DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn, **config['train'])
    val_dataloader = DataLoader(
        val_dataset, collate_fn=val_dataset.collate_fn, **config['validation'])

    checkpoint = load_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']

    for epoch in range(epochs):
        train_loss, train_accuracy = train(
            model, train_dataloader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(
            model, val_dataloader, criterion, device)

        is_best = True if len(history['val']['accuracy']) > 0 and val_accuracy > max(
            history['val']['accuracy']) else False

        history['train']['loss'].append(train_loss)
        history['train']['accuracy'].append(train_accuracy)
        history['val']['loss'].append(val_loss)
        history['val']['accuracy'].append(val_accuracy)

        if is_best:
            save_checkpoint(epoch, model, optimizer, history,
                            path="assets/checkpoint.pt")

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {train_loss:.4e}, Accuracy: {train_accuracy:6.2f}, \
            Validation Loss: {val_loss:.4e}, Validation Accuracy: {val_accuracy:6.2f}")

    save_history(history, path="assets/history.pkl")


if __name__ == "__main__":
    main()
