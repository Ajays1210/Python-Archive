"""
MNIST Digit Classifier
A CNN-based classifier for handwritten digits (0-9)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main():
    print("MNIST Digit Classifier")
    print("=" * 40)

    torch.manual_seed(42)

    # 1. Load dataset
    print("\n1. Loading MNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1000, num_workers=0)

    print(f"   Training: {len(train_data)} images")
    print(f"   Testing:  {len(test_data)} images")

    # 2. Build CNN model
    print("\n2. Building CNN model...")

    class DigitCNN(nn.Module):
        def __init__(self):
            super(DigitCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = DigitCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 3. Train model
    print("\n3. Training model (3 epochs)...")

    for epoch in range(3):
        model.train()
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"   Epoch {epoch + 1}: {accuracy:.2f}% accuracy")

    # 4. Test model
    print("\n4. Testing on unseen data...")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"   Test Accuracy: {test_accuracy:.2f}%")

    # 5. Make predictions and show examples
    print("\n5. Making sample predictions...")

    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:5], labels[:5]

    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)

    # Confidence scores
    print("\n6. Confidence scores:")
    confidences = []
    for i in range(5):
        conf = probs[i][predictions[i]].item() * 100
        confidences.append(conf)
        print(f"   Image {i}: {predictions[i].item()} ({conf:.1f}% confidence)")

    # 6. Visualize results
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))

    for i in range(5):
        img = images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')

        pred = predictions[i].item()
        true = labels[i].item()
        conf = confidences[i]

        color = 'green' if pred == true else 'red'
        axes[i].set_title(f'Pred: {pred}\nTrue: {true}\n({conf:.0f}%)', color=color)
        axes[i].axis('off')

    plt.suptitle(f'MNIST Predictions (Accuracy: {test_accuracy:.2f}%)', fontsize=14)
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=100)
    plt.show()

    # 7. Save model
    print("\n7. Saving model...")
    torch.save(model.state_dict(), 'model.pth')
    print("   Saved as 'model.pth'")

    # 8. Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Files saved: predictions.png, model.pth")
    print("=" * 40)

if __name__ == "__main__":
    main()