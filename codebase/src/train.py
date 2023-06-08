
import torch

# from torch.utils.tensorboard import SummaryWriter
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
        model_path = f'./models/model_{timestamp}_{name}_{epoch}'
        torch.save(model.state_dict(), model_path)
    
    @staticmethod
    def train_one_epoch(model, dataloader, optim, crit, writer, epoch, device, steps=1000):
        dl_len = len(dataloader)
        running_loss = 0.
        losses = torch.empty(dl_len // steps)

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optim.zero_grad()
            outputs = model(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            if i % steps == steps - 1:
                last_loss = running_loss / steps # loss per batch
                losses[i // steps] = last_loss
                print(f'epoch={epoch}, step={i // steps}, batch={i + 1}, loss={last_loss}')
                tb_x = epoch * dl_len + i + 1
                # writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return losses
    
    @staticmethod
    def train(model, optim, crit, tr_dl, ev_dl, epochs, name, device, steps=1000):
        """
        model: nn.Module
        optim: Optimizer
        crit: Criterion
        td_dl: Dataloader (training)
        ev_dl: Dataload (evaluation)
        epochs: int
        name: str
        device: str (cuda / cpu)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = None # SummaryWriter(f'runs/training_{name}_{timestamp}')

        best_loss = 1_000_000

        batch_loss_size = len(tr_dl) // steps
        tr_losses = torch.empty(epochs * batch_loss_size)
        ev_losses = torch.empty(epochs)
        for epoch in range(epochs):
            print(f'{name} - epoch: {epoch}')

            tr_loss = Training.train_one_epoch(model, tr_dl, optim, crit, writer, epoch, device, steps)
            ev_loss = Training.get_validation_loss(model, ev_dl, crit, device)

            tr_losses[epoch * batch_loss_size:(epoch + 1) * batch_loss_size] = tr_loss
            ev_losses[epoch] = ev_loss
            # writer.add_scalars('Training vs. Validation Loss', { 'train' : training_loss, 'validation' : validation_loss }, epoch)
            # writer.flush()

            if ev_loss < best_loss:
                best_loss = ev_loss
                Training.save_model(model, timestamp, epoch, name)
                
        return tr_losses, ev_losses

    

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