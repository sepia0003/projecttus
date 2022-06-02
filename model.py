# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import torch
import torch.utils.data
import pickle
import matplotlib.pyplot as plt


args = {
    "seed": 1234,
    "n_epoch": 200,
    "n_batch": 2,
    "lr": 0.001,
    "save_path": "importance_result.pth",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
args = argparse.Namespace(**args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class ArrangedData(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        assert len(self.inputs) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, index):
        return(
            torch.tensor(self.inputs[index]),
            torch.tensor(self.labels[index]),
        )

    def collate_fn(self, batch):
        inputs, labels = list(zip(*batch))

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        labels = torch.stack(labels)

        batch = [
            inputs,
            labels,
        ]

        return batch


# implementing class for importance prediction
class PredictImportance(torch.nn.Module):
    def __init__(self, n_vocab):
        super().__init__()
        self.embed = torch.nn.Embedding(n_vocab, 100)
        self.rnn = torch.nn.RNN(100, 256)
        self.linear = torch.nn.Linear(256, 1)

    def forward(self, inputs):
        embeded = self.embed(inputs)
        embeded = torch.transpose(embeded, 0, 1)
        output, hidden = self.rnn(embeded)
        logits = self.linear(hidden.squeeze(0))
        return logits


# function for estimating accuracy
def cal_accuracy(logits, labels):
    rounded_preds = torch.round(torch.sigmoid(logits))
    correct = (rounded_preds == labels).float()
    acc = correct.sum() / len(correct)
    return acc


# function for proceeding learning
def learn_per_ep(args, model, loader, loss_fn, optimizer):
    model.train()
    losses, access = [], []
    for batch in loader:
        optimizer.zero_grad()
        inputs, labels = map(lambda v: v.to(args.device), batch)
        logits = model(inputs).squeeze(1)
        loss = loss_fn(logits, labels.float())
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        losses.append(loss_val)
        acc_val = cal_accuracy(logits, labels.float())
        access.append(acc_val)

    return np.mean(losses), np.mean(access)


# function for evaluating
def estimate_ep(args, model, loader, loss_fn):
    model.eval()
    losses, access = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = map(lambda v: v.to(args.device), batch)
            logits = model(inputs).squeeze(1)
            loss = loss_fn(logits, labels.float())
            loss_val = loss.item()
            losses.append(loss_val)
            acc_val = cal_accuracy(logits, labels)
            access.append(acc_val)

    return np.mean(losses), np.mean(access)


def plotting_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], "b-", label="train_loss")
    plt.plot(history["valid_loss"], "r--", label="valid_loss")
    plt.xlabel("epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], "b-", label="train_acc")
    plt.plot(history["valid_acc"], "r--", label="valid_acc")
    plt.xlabel("epoch")
    plt.legend()

    plt.show()


with open('word_to_id.pkl', 'rb') as f:
    word_to_id = pickle.load(f)

with open('id_to_word.pkl', 'rb') as f:
    id_to_word = pickle.load(f)


with open('raw_inputs.pkl', 'rb') as f:
    raw_inputs = pickle.load(f)

with open('raw_inputs_valid.pkl', 'rb') as f:
    raw_inputs_valid = pickle.load(f)

with open('raw_labels.pkl', 'rb') as f:
    raw_labels = pickle.load(f)

with open('raw_labels_valid.pkl', 'rb') as f:
    raw_labels_valid = pickle.load(f)

# ============================================
# inputs,labels for learning
inputs = []
for s in raw_inputs:
    inputs.append([word_to_id[w] for w in s.split()])

labels = raw_labels
# ============================================
# inputs,labels for verifying
inputs_valid = []
for s in raw_inputs_valid:
    inputs_valid.append([word_to_id[w] for w in s.split()])

labels_valid = raw_labels_valid
# ============================================

# data for learning
dataset = ArrangedData(inputs, labels)
sampler = torch.utils.data.RandomSampler(dataset)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.n_batch, sampler=sampler, collate_fn=dataset.collate_fn)

# data for verifying
dataset = ArrangedData(inputs_valid, labels_valid)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=args.n_batch, sampler=None, collate_fn=dataset.collate_fn)

# data for testing
# dataset = ArrangedData(inputs, labels)
# test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.n_batch, sampler=None, collate_fn=dataset.collate_fn)


# making model for learning
model = PredictImportance(len(word_to_id))
model.to(args.device)

# setting loss and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# recording process in history, saving the best accuracy in max_acc
records = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
max_acc = 0


# starting learning
for e in range(args.n_epoch):
    train_loss, train_acc = learn_per_ep(args, model, train_loader, loss_fn, optimizer)
    valid_loss, valid_acc = estimate_ep(args, model, valid_loader, loss_fn)

    records["train_loss"].append(train_loss)
    records["train_acc"].append(train_acc)
    records["valid_loss"].append(valid_loss)
    records["valid_acc"].append(valid_acc)

    print(f"epoch: {e + 1:3d}, train_loss: {train_loss: .5f}, train_acc: {train_acc: .5f}, valid_loss: {valid_loss: .5f}, valid_acc: {valid_acc: .5f}")

    if max_acc < valid_acc:
        max_acc = valid_acc
        torch.save(
            {"state_dict": model.state_dict(), "valid_acc": valid_acc},
            args.save_path,
        )


# plotting learning results
plotting_history(records)


print("done\n")
