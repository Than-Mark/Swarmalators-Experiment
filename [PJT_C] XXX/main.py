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
        return np.sin(self.temp["deltaTheta"])
    
    @property
    def G(self) -> np.ndarray:
        """Effect of spatial similarity on phase couplings: 1 / |x_i - x_j|"""
        return self.div_distance_power(numerator=self.one, power=1, dim=1)
    
    @property
    def P(self) -> np.ndarray:
        """External drive: F*cos(omega*t - theta_i) / |x_0 - x_i|"""
        t = self.counts * self.dt
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
        t = self.counts * self.dt
        self.drivePosition = np.array([
            self.druveRadiusR * np.cos(self.driveAngularVelocityW * t),
            self.druveRadiusR * np.sin(self.driveAngularVelocityW * t)
        ])
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

        ax.scatter(self.drivePosition[0], self.drivePosition[1], c='r', s=100, marker='x', zorder=10)
        sc = ax.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)

        cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
        cbar.ax.set_ylim(0, 2*np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])

    def __str__(self) -> str:
        return f"MobileDrive_a{self.agentsNum}_K{self.K:.2f}_J{self.J:.2f}_F{self.F:.2f}"
    

class StateAnalysis:
    def __init__(self, model: MobileDrive, lookIndex: int = -1, showTqdm: bool = False):
        self.model = model
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm
        
        targetPath = f"{self.model.savePath}/{self.model}.h5"
        totalPositionX = pd.read_hdf(targetPath, key="positionX")
        totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
        
        TNum = totalPositionX.shape[0] // self.model.agentsNum
        self.TNum = TNum
        self.tRange = np.arange(0, (TNum - 1) * model.shotsnaps, model.shotsnaps) * self.model.dt
        self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
        self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)
        self.totalDrivePosition = np.array([
            model.druveRadiusR * np.cos(model.driveAngularVelocityW * self.tRange),
            model.druveRadiusR * np.sin(model.driveAngularVelocityW * self.tRange)
        ]).transpose(1, 0)
        self.totalDrivePhaseTheta = np.mod(model.driveThateVelocityOmega * self.tRange, 2 * np.pi)

        self.centersValue = None
        self.classesValue = None

        if self.showTqdm:
            self.iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            self.iterObject = range(1, self.totalPhaseTheta.shape[0])
            
    @staticmethod
    def calc_order_parameter_R(model: MobileDrive) -> float:
        return np.abs(np.sum(np.exp(1j * model.phaseTheta))) / model.agentsNum
    
    @staticmethod
    def calc_order_parameter_S(model: MobileDrive) -> float:
        phi = np.arctan2(model.positionX[:, 1], model.positionX[:, 0])
        Sadd = np.abs(np.sum(np.exp(1j * (phi + model.phaseTheta)))) / model.agentsNum
        Ssub = np.abs(np.sum(np.exp(1j * (phi - model.phaseTheta)))) / model.agentsNum
        return np.max([Sadd, Ssub])
    
    @staticmethod
    @nb.njit
    def _calc_pointX(Iatt, Irep, Fatt, Frep):
        dim = Iatt.shape[0]
        return np.sum(
            Iatt * Fatt.reshape((dim, dim, 1)) - Irep * Frep.reshape((dim, dim, 1)),
            axis=1
        ) / dim

    @staticmethod
    def calc_order_parameter_Vp(model: MobileDrive) -> float:
        Iatt, Irep, Fatt, Frep = model.Iatt, model.Irep, model.Fatt, model.Frep
        pointX = StateAnalysis._calc_pointX(Iatt, Irep, Fatt, Frep)
        phi = np.arctan2(pointX[:, 1], pointX[:, 0])
        return np.abs(np.sum(np.exp(1j * phi))) / model.agentsNum
    
    @staticmethod
    def calc_order_parameter_Ptr(model: MobileDrive) -> float:
        K, H, G, P = model.K, model.H, model.G, model.P
        dim = P.shape[0]
        pointTheta = K * np.sum(H * G, axis=1) / dim + P
        Ntr = np.abs(pointTheta - model.driveThateVelocityOmega) < 0.2 / model.dt * 0.1
        return Ntr.sum() / model.agentsNum
    
    @staticmethod
    def plot_last_state(model: MobileDrive, ax: plt.Axes, withColorBar: bool =True, s: float = 50, driveS: float = 100):
        t = model.counts * model.dt
        model.drivePosition = np.array([
            np.cos(model.driveThateVelocityOmega * t) * model.druveRadiusR,
            np.sin(model.driveThateVelocityOmega * t) * model.druveRadiusR
        ])
        ax.scatter(model.drivePosition[0], model.drivePosition[1], color="white", s=driveS, marker='o', edgecolors='k', zorder=10)
        sc = ax.scatter(model.positionX[:, 0], model.positionX[:, 1], s=s,
                    c=model.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
        driveCircle = plt.Circle((0, 0), model.druveRadiusR, color='black', fill=False, lw=2, linestyle='--')
        ax.add_artist(driveCircle)
        if withColorBar:
            cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
            cbar.ax.set_ylim(0, 2*np.pi)
            cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])