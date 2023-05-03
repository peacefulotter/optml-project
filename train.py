
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Training:
    @staticmethod
    def get_validation_loss(model, dataloader, crit):
        validation_loss = 0.0
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs)
            loss = crit(outputs, labels)
            validation_loss += loss
        return validation_loss
    
    @staticmethod
    def save_model(model, timestamp, epoch):
        model_path = './model/model_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)
    
    @staticmethod
    def train_one_epoch(model, dataloader, optim, crit, writer, epoch):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(dataloader):
            inputs, labels = data
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
    def train(model, optim, crit, training_dl, validation_dl, epochs):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        writer = SummaryWriter('runs/training_{}'.format(timestamp))

        best_loss = 1_000_000

        losses = torch.empty((epochs, 2))
        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch))

            training_loss = Training.train_one_epoch(model, training_dl, optim, crit, writer, epoch)
            validation_loss = Training.get_validation_loss(model, validation_dl, crit)

            losses[epoch] = torch.Tensor([training_loss, validation_loss])
            writer.add_scalars('Training vs. Validation Loss', { 'train' : training_loss, 'validation' : validation_loss }, epoch)
            writer.flush()

            if validation_loss < best_loss:
                best_loss = validation_loss
                Training.save_model(model, timestamp, epoch)
                
        return losses

    

if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms

    from torchvision import datasets
    from torch.utils.data import DataLoader

    from classifier import GarmentClassifier

    lr = 0.0001
    momentum = 0.9
    batch_size = 4

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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()
    training = Training.train(model, optimizer, criterion, training_loader, validation_loader, epochs=5)