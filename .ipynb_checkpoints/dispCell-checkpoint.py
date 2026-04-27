import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def interactive_slice(cell):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    idx = cell.shape[2] // 2
    img = ax.imshow(cell[:, :, idx], cmap='viridis')
    ax.set_title(f'z = {idx}')
    ax.axis('off')

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, cell.shape[2]-1,
                    valinit=idx, valstep=1)

    def update(val):
        i = int(slider.val)
        img.set_data(cell[:, :, i])
        ax.set_title(f'z = {i}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# 実行
cell = np.load("cell.npy")
interactive_slice(cell)
