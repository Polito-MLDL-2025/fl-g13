#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fl_g13.config import RAW_DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from collections import Counter


# ## Load data

transform = transforms.Compose([
    transforms.ToTensor()
])
cifar100_train = datasets.CIFAR100(root=RAW_DATA_DIR, train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100(root=RAW_DATA_DIR, train=False, download=True, transform=transform)


# ## Dataset overview

print(f"Training samples: {len(cifar100_train)}")
print(f"Test samples: {len(cifar100_test)}")
print(f"Classes: {len(cifar100_train.classes)}")
print(f"First 5 classes: {cifar100_train.classes[:5]}")


# ## Display a few sample images

def show_samples(dataset, n=10):
    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i in range(n):
        img, label = dataset[i]
        axes[i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axes[i].axis('off')
        axes[i].set_title(dataset.classes[label], fontsize=8)
    plt.tight_layout()
    plt.show()

show_samples(cifar100_train)


# ## Class distribution

def get_class_distribution(dataset):
    labels = [label for _, label in dataset]
    return Counter(labels)

train_dist = get_class_distribution(cifar100_train)
test_dist = get_class_distribution(cifar100_test)


# Plot distribution
def plot_distribution(distribution, title):
    labels = list(range(100))
    counts = [distribution[l] for l in labels]
    plt.figure(figsize=(12, 4))
    plt.bar(labels, counts)
    plt.title(title)
    plt.xlabel("Class ID")
    plt.ylabel("Frequency")
    plt.show()

plot_distribution(train_dist, "Class Distribution (Train)")
plot_distribution(test_dist, "Class Distribution (Test)")


# ## Train, val split

from fl_g13.dataset import iid_sharding, non_iid_sharding, train_test_split, check_subset_distribution


### train val split
train_dataset, val_dataset = train_test_split(cifar100_train,train_ratio=0.8)


# **Check distribution**

check_subset_distribution(val_dataset)


# ## I.I.D Sharding Split

## k client
k =10
clients_dataset= iid_sharding(cifar100_train,k)


check_subset_distribution(clients_dataset[0])


# ## Non I.I.D Sharding Split

## k client , nc = 2
k =10
nc = 2 
non_iid_clients_dataset= non_iid_sharding(cifar100_train,k,keep_random=nc)


check_subset_distribution(non_iid_clients_dataset[0])

