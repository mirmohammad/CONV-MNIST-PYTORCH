import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from MNIST import MNIST
from MyModel import MyModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tqdm.write('CUDA is not available!' if not torch.cuda.is_available() else 'CUDA is available!')
tqdm.write('')

classes = 10

batch_size = 100

epochs = 10
learning_rate = 1e-2

train_dataset = MNIST('mnist.pkl.gz', split='train', transform=transforms.Compose([transforms.ToTensor()]))
valid_dataset = MNIST('mnist.pkl.gz', split='valid', transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = MNIST('mnist.pkl.gz', split='test', transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MyModel(num_classes=classes)

model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
total_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
tqdm.write('Number of parameters : {}'.format(total_params))
tqdm.write('Number of trainable parameters : {}'.format(total_grad_params))


def train(_epoch):
    model.train()
    num_images = 0.
    running_loss = 0.
    running_accuracy = 0.
    monitor = tqdm(train_loader, desc='Training')
    for i, (train_images, train_labels) in enumerate(monitor):
        train_images, train_labels = train_images.to(device), train_labels.to(device)

        outputs = model(train_images)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, train_labels)

        num_images += train_images.size(0)
        running_loss += loss.item() * train_images.size(0)
        running_accuracy += (preds == train_labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        monitor.set_postfix(epoch=_epoch, loss=running_loss / num_images, accuracy=running_accuracy / num_images)

    epoch_loss = running_loss / num_images
    epoch_accuracy = running_accuracy / num_images

    return epoch_loss, epoch_accuracy


def valid(_epoch):
    model.eval()
    with torch.no_grad():
        num_images = 0.
        running_loss = 0.
        running_accuracy = 0.
        monitor = tqdm(valid_loader, desc='Validating')
        for i, (valid_images, valid_labels) in enumerate(monitor):
            valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)

            outputs = model(valid_images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, valid_labels)

            num_images += valid_images.size(0)
            running_loss += loss.item() * valid_images.size(0)
            running_accuracy += (preds == valid_labels).sum().item()

            monitor.set_postfix(epoch=_epoch, loss=running_loss / num_images, accuracy=running_accuracy / num_images)

        epoch_loss = running_loss / num_images
        epoch_accuracy = running_accuracy / num_images

    return epoch_loss, epoch_accuracy


def test(_epoch):
    model.eval()
    with torch.no_grad():
        num_images = 0.
        running_loss = 0.
        running_accuracy = 0.
        monitor = tqdm(test_loader, desc='Testing')
        for i, (test_images, test_labels) in enumerate(monitor):
            test_images, test_labels = test_images.to(device), test_labels.to(device)

            outputs = model(test_images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, test_labels)

            num_images += test_images.size(0)
            running_loss += loss.item() * test_images.size(0)
            running_accuracy += (preds == test_labels).sum().item()

            monitor.set_postfix(epoch=_epoch, loss=running_loss / num_images, accuracy=running_accuracy / num_images)

        epoch_loss = running_loss / num_images
        epoch_accuracy = running_accuracy / num_images

    return epoch_loss, epoch_accuracy


def log_results(file_name, losses, accuracies):
    file = open(file_name, 'w+')
    for x, y in zip(losses, accuracies):
        file.write('{:.3f},'.format(x))
        file.write('{:.3f}'.format(y))
        file.write('\n')
    file.close()


if __name__ == '__main__':

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(epochs):
        train_loss, train_accuracy = train(epoch + 1)
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        valid_loss, valid_accuracy = valid(epoch + 1)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_accuracy)

        test_loss, test_accuracy = test(epoch + 1)
        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

        tqdm.write('')

    log_results('CNN_TRAIN' + '.csv', train_losses, train_accs)
    log_results('CNN_VALID' + '.csv', valid_losses, valid_accs)
    log_results('CNN_TEST' + '.csv', test_losses, test_accs)
