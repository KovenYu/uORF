import matplotlib.figure
import matplotlib.pyplot as plt
import torch

from torch import optim

def make_average_gradient_plot(named_parameter_chain) -> matplotlib.figure.Figure:
    layer_names = []
    avg_grads = []

    for n, p in named_parameter_chain:
        if p.grad is not None and "bias" not in n:
            with torch.no_grad():
                layer_names.append(n)
                avg_grads.append(p.grad.abs().mean().cpu().item())

    fig, ax = plt.subplots()
    ax.barh(layer_names, avg_grads)
    for i in ax.patches:
        ax.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')

    ax.set_title('Average gradient of layers')
    ax.set_xlabel('Average gradient')

    return fig
    