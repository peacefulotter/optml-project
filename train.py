
import torch

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Training:
    @staticmethod
    def get_validation_loss(model, dataloader, crit, device):
        validation_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = crit(outputs, labels)
            validation_loss += loss
        return validation_loss
    
    @staticmethod
    def save_model(model, timestamp, epoch, name):
        model_path = f'./model/model_{timestamp}_{name}_{epoch}'
        torch.save(model.state_dict(), model_path)
    
    @staticmethod
    def train_one_epoch(model, dataloader, optim, crit, writer, epoch, device):
        running_loss = 0.
        last_loss = 0.

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optim.zero_grad()
            outputs = model(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(dataloader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    @staticmethod
    def train(model, optim, crit, training_dl, validation_dl, epochs, name, device):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(f'runs/training_{name}_{timestamp}')

        best_loss = 1_000_000

        losses = torch.empty((2, epochs))
        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch))

            training_loss = Training.train_one_epoch(model, training_dl, optim, crit, writer, epoch, device)
            validation_loss = Training.get_validation_loss(model, validation_dl, crit, device)

            losses[0][epoch] = training_loss
            losses[1][epoch] = validation_loss
            writer.add_scalars('Training vs. Validation Loss', { 'train' : training_loss, 'validation' : validation_loss }, epoch)
            writer.flush()

            if validation_loss < best_loss:
                best_loss = validation_loss
                Training.save_model(model, timestamp, epoch, name)
                
        return losses

    

if __name__ == '__main__':
    from torchvision import datasets
    import torchvision.transforms as transforms

    from torch.optim import SGD
    from torch.nn import CrossEntropyLoss
    from torch.utils.data import DataLoader

    from classifier import GarmentClassifier

    lr = 0.0001
    momentum = 0.9
    batch_size = 4
    epochs = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device', device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def get_loader(train: bool, shuffle: bool):
        set = datasets.FashionMNIST(root='./data', train=train, transform=transform, download=True)
        return DataLoader(set, batch_size=batch_size, shuffle=shuffle)
   
    training_loader = get_loader(train=True, shuffle=True)
    validation_loader = get_loader(train=False, shuffle=False)

    model = GarmentClassifier()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = CrossEntropyLoss()
    losses = Training.train(model, optimizer, criterion, training_loader, validation_loader, epochs=epochs, name='train_test', device=device)