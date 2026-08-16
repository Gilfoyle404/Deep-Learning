# train_mnist_tpc.py
#
# What this script does:
# ----------------------
# - Trains a simple fully connected network on MNIST.
# - Runs on HPU ("hpu" device) when available.
# - Uses TPCReLU / TPCLinear from custom_ops_tpc.py if import works.
#
# IMPORTANT:
# ----------
# Even when TPCReLU/TPCLinear are used, they internally call standard
# PyTorch ops (clamp, matmul) which are then compiled by Habana's
# backend to its own internal TPC kernels.
#
# This script DOES NOT:
# - Directly call your hand-written TPC .c kernels
#   (relu_fwd.c, relu_bckd.c, linear_fwd.c, linear_bckd.c).
# - Use Habana's CustomOp API (no GUID, no registerUserCustomOp).
#
# It is a good, HPU-optimized baseline, but not a "true" custom TPC
# kernel integration.

import os
os.environ['PT_HPU_LAZY_MODE'] = '1'  # Enable lazy execution on HPU

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import habana_frameworks.torch as htorch  # noqa: F401 (initializes HPU backend)

# Try to import our "TPC" wrappers (which are actually HPU-optimized PyTorch ops)
try:
    from custom_ops_tpc import TPCReLU, TPCLinear
    print("✓ Using TPCReLU / TPCLinear wrappers (PyTorch ops on HPU)")
    USE_TPC = True
except Exception as e:
    print(f"⚠ Could not import custom_ops_tpc: {e}")
    print("  Falling back to standard nn.ReLU / nn.Linear")
    USE_TPC = False
    TPCReLU = nn.ReLU
    TPCLinear = nn.Linear


class MNISTNet(nn.Module):
    """
    Simple fully-connected MNIST network.

    Layers:
    - fc1: 784 -> 128
    - relu1
    - fc2: 128 -> 64
    - relu2
    - fc3: 64 -> 10

    Note: The actual implementation of TPCLinear/TPCReLU is either:
    - our wrappers around standard PyTorch ops (on HPU), or
    - plain nn.Linear / nn.ReLU, depending on USE_TPC.
    """

    def __init__(self):
        super().__init__()

        self.fc1 = TPCLinear(784, 128)
        self.relu1 = TPCReLU()
        self.fc2 = TPCLinear(128, 64)
        self.relu2 = TPCReLU()
        self.fc3 = TPCLinear(64, 10)

    def forward(self, x):
        # MNIST images: [batch, 1, 28, 28]
        x = x.view(-1, 784)  # flatten to [batch, 784]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    One training epoch:
    - Loops over train_loader
    - Computes loss, backward, optimizer step
    - Prints running loss and accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy on this running set
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total_samples += target.size(0)

        if batch_idx % 100 == 0:
            running_acc = 100.0 * correct / total_samples
            print(
                f"  Batch {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.4f} "
                f"Acc: {running_acc:.2f}%"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total_samples

    print(f"\nEpoch {epoch} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """
    Evaluation:
    - Runs model in eval mode on test_loader
    - Returns average loss and overall accuracy
    """
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f"Test - Loss: {test_loss:.4f}, Acc: {accuracy:.2f}%\n")
    return test_loss, accuracy


def main():
    print("=" * 70)
    print("MNIST Training with HPU-Optimized PyTorch Ops (TPC wrappers)")
    print("=" * 70)

    # Choose device: HPU if available, otherwise CPU
    device = torch.device("hpu" if torch.hpu.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "hpu":
        print(f"HPU: {torch.hpu.get_device_name(0)}")

    # Data preprocessing: standard MNIST normalization
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Download / load datasets
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"\nDataset sizes: {len(train_dataset)} train, {len(test_dataset)} test")

    # Build model
    model = MNISTNet().to(device)
    print(f"\nModel:\n{model}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Main training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70 + "\n")

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 70)
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)

    # Save trained weights
    torch.save(model.state_dict(), "mnist_tpc_kernels.pth")
    print("\nModel saved to: mnist_tpc_kernels.pth")


if __name__ == "__main__":
    main()
