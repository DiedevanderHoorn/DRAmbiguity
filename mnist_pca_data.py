import os
import random
import numpy as np
import torch
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import joblib


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load full MNIST test set
transform = transforms.ToTensor()
mnist_test = datasets.MNIST(
    root="./data",
    train=False,          #only test set
    download=True,
    transform=transform
)

# Extract data
X = np.stack([img.view(-1).numpy() for img, _ in mnist_test])  # (10000, 784)
y = np.array([label for _, label in mnist_test])               # (10000,)
images = np.stack([img.squeeze().numpy() for img, _ in mnist_test])  # (10000, 28, 28)
indices = np.arange(len(mnist_test))                           # (10000,)

print("Loaded:", X.shape, y.shape, images.shape, indices.shape)

pca = PCA(n_components=200, random_state=SEED)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_pca = pca.fit_transform(X)

print("After PCA:", X_pca.shape)

save_dir = "data/mnist_testset_pca_200_norm"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "X.npy"), X_pca)
np.save(os.path.join(save_dir, "y.npy"), y)
np.save(os.path.join(save_dir, "images.npy"), images)
np.save(os.path.join(save_dir, "indices.npy"), indices)
joblib.dump(pca, os.path.join(save_dir,"pca.pkl"))

print("Saved all files to:", save_dir)

