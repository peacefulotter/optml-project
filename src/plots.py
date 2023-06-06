
import matplotlib.pyplot as plt

def plot_losses(losses, name, save=False):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))
    axes[0].plot(range(losses.shape[1]), losses[0])
    axes[1].plot(range(losses.shape[1]), losses[1])
    plt.tight_layout()
    if save:
        plt.savefig( f'plots/{name}.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # Test plot_losses
    import torch
    losses = torch.rand(size=(2, 100))
    plot_losses(losses, name='test', save=True)

