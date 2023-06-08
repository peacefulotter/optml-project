
import matplotlib.pyplot as plt

def plot_losses(losses, name, save=False):
    _, axes = plt.subplots(nrows=1, ncols=len(losses), figsize=(10,3))
    for i, loss in enumerate(losses): 
        axes[i].plot(range(loss.shape[0]), loss)
    plt.tight_layout()
    if save:
        plt.savefig( f'plots/{name}.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # Test plot_losses
    import torch
    losses = torch.rand(size=(2, 100))
    plot_losses(losses, name='test', save=True)

