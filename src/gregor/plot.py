import matplotlib.pyplot as plt



def plot_vector(vector, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    vector.plot(ax=ax, **kwargs)
    return ax


def plot_raster(raster, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(raster, **kwargs)
    return ax