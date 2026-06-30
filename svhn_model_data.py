import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random
import os
from graphdisambiguation import GraphDisambiguation, create_UMAP_graph_and_embedding
from helpers import save_for_drawing, render_images
import networkx as nx

import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SVHNCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 64, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 512),
            nn.ReLU()
        )

        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return self.output(x)


def custom_init(layer):
    if isinstance(layer, nn.Conv2d):
        k = layer.kernel_size[0]
        in_c = layer.in_channels
        out_c = layer.out_channels
        s = (6.0 / (k*k*in_c + k*k*out_c)) ** 0.5
        nn.init.uniform_(layer.weight, -s, s)
        nn.init.constant_(layer.bias, 0)

    if isinstance(layer, nn.Linear):
        n_in = layer.in_features
        n_out = layer.out_features
        s = (6.0 / (n_in + n_out)) ** 0.5
        nn.init.uniform_(layer.weight, -s, s)
        nn.init.constant_(layer.bias, 0)

if __name__ == "__main__":
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.ToTensor()

    full_train_set = torchvision.datasets.SVHN(
        root="./data", split='train', download=True, transform=transform
    )
    test_set = torchvision.datasets.SVHN(
        root="./data", split='test', download=True, transform=transform
    )

    def fix_svhn_labels(dataset):
        labels = dataset.labels
        labels = np.where(labels == 10, 0, labels)
        dataset.labels = labels

    fix_svhn_labels(full_train_set)
    fix_svhn_labels(test_set)

    train_size = int(0.9 * len(full_train_set))
    val_size = len(full_train_set) - train_size

    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    model = SVHNCNN().to(device)
    model.apply(custom_init)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    def train_epoch():
        model.train()
        correct, total = 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).long()

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return correct / total

    def eval_epoch(loader):
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device).long()
                out = model(X)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total


    EPOCHS = 20
    for epoch in range(EPOCHS):
        train_acc = train_epoch()
        val_acc = eval_epoch(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

    final_test_acc = eval_epoch(test_loader)
    print("FINAL TEST ACCURACY:", final_test_acc)

    # Save model
    torch.save(model.state_dict(), "svhn_cnn_model.pt")
    print("Model saved.")

    model = SVHNCNN().to(device)
    model.load_state_dict(torch.load("svhn_cnn_model.pt", map_location=device))
    model.eval()

    print("Model loaded.")


    transform = transforms.ToTensor()

    test_set = torchvision.datasets.SVHN(
        root="./data",
        split="test",
        download=True,
        transform=transform
    )

    test_set.labels = np.where(test_set.labels == 10, 0, test_set.labels)

    np.random.seed(42)
    subset_size = 5000
    indices = np.random.choice(len(test_set), size=subset_size, replace=False)

    subset = Subset(test_set, indices)
    subset_loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)

    activations_list = []
    preds_list = []
    probs_list = []
    labels_list = []
    images_list = []


    def hook_fn(module, input, output):
        activations_list.append(output.detach().cpu())

    handle = model.classifier[3].register_forward_hook(hook_fn)

    #Inference
    with torch.no_grad():
        for X, y in subset_loader:
            images_list.append(X.cpu())               # (N,3,32,32)
            labels_list.append(y.clone())             # raw labels

            X = X.to(device)
            y = y.to(device).long()

            logits = model(X)
            preds = logits.argmax(dim=1).cpu()
            probs = torch.softmax(logits, dim=1).cpu()

            preds_list.append(preds)
            probs_list.append(probs)

    handle.remove()


    images = torch.cat(images_list).numpy()          # (5000, 3, 32, 32)
    labels = torch.cat(labels_list).numpy()          # (5000,)
    preds = torch.cat(preds_list).numpy()            # (5000,)
    probs = torch.cat(probs_list).numpy()            # (5000,10)
    activations = torch.cat(activations_list).numpy()  # (5000,512)

    print("Shapes:")
    print("images:", images.shape)
    print("labels:", labels.shape)
    print("preds:", preds.shape)
    print("probs:", probs.shape)
    print("activations:", activations.shape)

    SAVE_DIR = "data/svhn_5000_latent"
    os.makedirs(SAVE_DIR, exist_ok=True)

    np.save("data/svhn_5000_latent/images.npy", images)
    np.save("data/svhn_5000_latent/y.npy", labels)
    np.save("data/svhn_5000_latent/preds.npy", preds)
    np.save("data/svhn_5000_latent/probs_5000.npy", probs)
    np.save("data/svhn_5000_latent/X.npy", activations)

    print("Saved 5000-sample subset!")

    