import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from itertools import product
from typing import List
import pandas as pd
import numpy as np
import numba as nb
import imageio
import sys
import os
import shutil

randomSeed = 10

if "ipykernel_launcher.py" in sys.argv[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

import seaborn as sns

sns.set_theme(font_scale=1.1, rc={
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
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class MobileDrive(Swarmalators2D):
    def __init__(self, agentsNum: int, K: float, J: float, F: float,
                 randomSeed: int = 100, dt: float = 0.1, tqdm: bool = True, 
                 savePath: str = None, shotsnaps: int = 10, overWrite: bool = False) -> None:
        super().__init__(agentsNum=agentsNum, dt=dt, K=K, randomSeed=randomSeed, tqdm=tqdm, 
                         savePath=savePath, shotsnaps=shotsnaps, overWrite=overWrite)
        self.J = J
        self.F = F
        self.driveThateVelocityOmega = 0.5 * np.pi
        self.driveAngularVelocityW = 0.5
        self.druveRadiusR = 0.5
        self.drivePosition = np.array([0.5, 0])
        self.one = np.ones((agentsNum, agentsNum))

    @property
    def Fatt(self) -> np.ndarray:
        """
        Effect of phase similarity on spatial attraction: 1 + J * cos(theta_j - theta_i)
        """
        return 1 + self.J * np.cos(self.temp["deltaTheta"])

    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion: 1"""
        return self.one
    
    @property
    def Iatt(self) -> np.ndarray:
        """Spatial attraction: (x_j - x_i) / |x_j - x_i|"""
        return self.div_distance_power(numerator=self.temp["deltaX"], power=1)
    
    @property
    def Irep(self) -> np.ndarray:
        """Spatial repulsion: (x_j - x_i) / |x_j - x_i| ^ 2"""
        return self.div_distance_power(numerator=self.temp["deltaX"], power=2)
    
    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity: 0"""
        return 0
    
    @property
    def omega(self) -> np.ndarray:
        """Natural frequency: 0"""
        return 0
    
    @property
    def H(self) -> np.ndarray:
        """Phase interaction: sin(theta_j - theta_i)"""
        return np.sin(self.deltaTheta)
    
    @property
    def G(self) -> np.ndarray:
        """Effect of spatial similarity on phase couplings: 1 / |x_i - x_j|"""
        return self.div_distance_power(numerator=self.one, power=1, dim=1)
    
    @property
    def P(self) -> np.ndarray:
        """External drive: F*cos(omega*t - theta_i) / |x_0 - x_i|"""
        t = self.counts * self.dt
        self.drivePosition = np.array([
            self.druveRadiusR * np.cos(self.driveAngularVelocityW * t),
            self.druveRadiusR * np.sin(self.driveAngularVelocityW * t)
        ])
        disance = self.distance_x(((self.positionX - self.drivePosition)[:, np.newaxis]))[:, 0]
        
        return self.F * np.cos(self.driveThateVelocityOmega * t - self.phaseTheta) / disance

    def update(self) -> None:
        self.update_temp()
        self.positionX, self.phaseTheta = self._update(
            self.positionX, self.phaseTheta,
            self.velocity, self.omega,
            self.Iatt, self.Irep,
            self.Fatt, self.Frep,
            self.H, self.G, self.P,
            self.K, self.dt, 
        )
        self.counts += 1

    @staticmethod
    @nb.njit
    def _update(
        positionX: np.ndarray, phaseTheta: np.ndarray,
        velocity: np.ndarray, omega: np.ndarray,
        Iatt: np.ndarray, Irep: np.ndarray,
        Fatt: np.ndarray, Frep: np.ndarray,
        H: np.ndarray, G: np.ndarray, P: np.ndarray,
        K: float, dt: float
    ):
        dim = positionX.shape[0]
        pointX = velocity + np.sum(
            Iatt * Fatt.reshape((dim, dim, 1)) - Irep * Frep.reshape((dim, dim, 1)),
            axis=1
        ) / dim
        positionX += pointX * dt
        pointTheta = omega + K * np.sum(H * G, axis=1) / dim + P
        phaseTheta = np.mod(phaseTheta + pointTheta * dt, 2 * np.pi)
        return positionX, phaseTheta

    def plot(self, ax: plt.Axes = None) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(self.drivePosition[0], self.drivePosition[1], c='r', s=100, marker='x', label="Drive")
        sc = ax.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)

        cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
        cbar.ax.set_ylim(0, 2*np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])

    def __str__(self) -> str:
        return f"MobileDrive_a{self.agentsNum}_K{self.K:.2f}_J{self.J:.2f}_F{self.F:.2f}"