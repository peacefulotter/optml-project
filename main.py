import torch
import torchvision.transforms as transforms

from torchvision import datasets
from torch.optim import SGD, Adam, AdamW, Adagrad, Adadelta, SparseAdam, ASGD, RAdam, NAdam, Adamax
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from src.classifier import GarmentClassifier
from src.train import Training
from src.plots import plot_losses

# TODO: Need to define what a "FAIR" comparison is between optimizers
# TODO: Need to select the ones that are "made to fit" for the problem

def get_loader(train: bool, shuffle: bool):
    set = datasets.FashionMNIST(root='./data', train=train, transform=transform, download=True)
    return DataLoader(set, batch_size=batch_size, shuffle=shuffle)

def get_model_optim(Optimizer, device, **kwargs):
    model = GarmentClassifier() # Network.from_sequential()
    model.to(device)

    import inspect
    args = inspect.getfullargspec(Optimizer).args
     # TODO: add kwargs to optimizers
    if 'momentum' in args:
        optim = Optimizer(model.parameters(), lr=lr, momentum=momentum) 
    else:
        optim = Optimizer(model.parameters(), lr=lr)
    return model, optim

if __name__ == '__main__':

    from lion.lion import Lion
    from Sophia.sophia import SophiaG

    lr = 0.0001
    momentum = 0.9
    batch_size = 4 # default batch_size to 4 but GPU can handle way more - nevertheless, an increase of batch size isn't faster somehow
    epochs = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device', device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training_loader = get_loader(train=True, shuffle=True)
    validation_loader = get_loader(train=False, shuffle=False)

    optimizers = [SophiaG, Lion, Adam, AdamW]
    models_optimizers = [get_model_optim(optim, device) for optim in optimizers]
    criterion = CrossEntropyLoss()

    for model, optim in models_optimizers:   
        optim_name = type(optim).__name__
        losses = Training.train(model, optim, criterion, training_loader, validation_loader, epochs=epochs, name=optim_name, device=device)
        with torch.no_grad():
            plot_losses(losses, name=optim_name, save=True)