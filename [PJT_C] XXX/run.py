def run_model(model):
    model.run(5000)

if __name__ == "__main__":

    import matplotlib.colors as mcolors
    import matplotlib.animation as ma
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from itertools import product
    import pandas as pd
    import numpy as np
    import numba as nb
    import imageio
    import os
    import shutil

    randomSeed = 100

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
    )

    @nb.njit
    def colors_idx(phaseTheta):
        return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

    import seaborn as sns

    sns.set_theme(
        style="ticks", 
        font_scale=1.1, rc={
        'figure.figsize': (6, 5),
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.color': '#dddddd',
        'grid.linewidth': 0.5,
        "lines.linewidth": 1.5,
        'text.color': '#000000',
        'figure.titleweight': "bold",
        'xtick.color': '#000000',
        'ytick.color': '#000000'
    })


    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

    from main import *
    from multiprocessing import Pool

    Fs = np.linspace(0, 5, 30)
    Ks = np.sort(np.concatenate([np.linspace(-1, 1, 30), [0]]))

    models = [
        MobileDrive(agentsNum=500, K=K, J=1, F=F, dt=0.01,
                    tqdm=False, savePath="./data", randomSeed=10, overWrite=True) 
        for F in Fs
        for K in Ks
    ] + [
        # MobileDrive(agentsNum=500, K=K, J=0.1, F=F, dt=0.02,
        #              tqdm=False, savePath="./data", randomSeed=10, overWrite=False) 
        # for F in Fs
        # for K in Ks
    ]

    with Pool(8) as p:
        # p.map(run_model, models)

        p.map(
            run_model,
            tqdm(models, desc="run models", total=len(models))
        )