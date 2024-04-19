import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from .utils import scale2range


def visualise_samples(x, scale=False):
    while len(x.shape) < 4:
        x = x.unsqueeze(0)
    if scale:
        x = scale2range(x, [0, 1])
    fig = plt.figure()
    grid = make_grid(x).permute(1, 2, 0).cpu()
    plt.imshow(grid)
    return fig
