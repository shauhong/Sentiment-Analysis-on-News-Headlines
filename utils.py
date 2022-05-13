import torch
import matplotlib.pyplot as plt
import os
import yaml
import pickle


def pad_sentences(sentences, pad_token):
    max_length = max([len(sentence) for sentence in sentences])
    sents_padded = [sentence + [pad_token] * (max_length - len(sentence)) for sentence in sentences]
    return sents_padded

def load_config(path="assets/config.yaml"):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def save_checkpoint(epoch, model, optimizer, history, path="assets/checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, path)
    
def load_checkpoint(path="assets/checkpoint.pt"):
    if os.path.exists(path):
        return torch.load(path)
    
def save_history(history, path="assets/history.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(history, f)
        
def load_history(path="assets/history.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            history = pickle.load(f)
        return history
    
def plot(epochs, losses, legends, title=None, xlabel="Epoch", ylabel="Loss", save=None, figsize=None):
    assert len(epochs) == len(losses) == len(legends)
    for i in range(len(epochs)):
        plt.plot(epochs[i], losses[i], label=legends[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    if figsize is not None:
        plt.figure(figsize=figsize)
    if save is not None:
        plt.savefig(save)
    plt.show()
    
def predictions2labels(predictions, labels):
    results = list()
    for i in range(len(predictions)):
        results.append(labels[predictions[i]])
    return results