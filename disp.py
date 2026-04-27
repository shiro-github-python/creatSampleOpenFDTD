import numpy as np
import napari

cell = np.load('cell.npy')

viewer = napari.Viewer()
viewer.add_labels(cell)
napari.run()

