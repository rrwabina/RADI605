import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def display_images_from_loader(loader, class_names = ['Benign', 'Malignant'], nrow = 6):
    for images, labels in loader:
        break

    print('Label:', labels.numpy())
    print('Class: ', *np.array([class_names[i] for i in labels]))

    im = make_grid(images, nrow=nrow)
    plt.figure(figsize=(15, 8))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()

def count_classes_in_loader(loader):
    class_counts = defaultdict(int)
    for _, targets in loader:
        for target in targets:
            class_counts[target.item()] += 1
    for i in range(len(loader.dataset.classes)):
        print(f'Class {loader.dataset.classes[i]}: {class_counts[i]}')

def scrap_train():
    model_vgg19 = torchvision.models.vgg19(weights = False).to(device)
    learning_rate = 1e-2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_vgg19.parameters(), lr = learning_rate, eps = 10e-06)

    train_losses = []
    val_losses = []
    for epoch in range(10):
        running_loss = 0.0
        model_vgg19.train()
        for idx, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model_vgg19(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx == 0:
                print('Epoch %2d | Iteration %2d Train Loss: %.5f' % (epoch + 1, idx + 1, running_loss / 100))
                running_loss = 0.0    
            elif idx == 5:
                break
            train_losses.append(loss.item())

        model_vgg19.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_data in valid_loader:
                val_inputs, val_labels = val_data
                val_outputs = model_vgg19(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
            val_losses.append(val_loss / len(valid_loader))
            print('Epoch %2d | Validation Loss: %.5f' % (epoch + 1, val_loss / len(valid_loader)))

