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

if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class SelfPropSwarmalator(Swarmalators2D):
    def __init__(self, K: float, J: float, 
                 distanceD0: float, boundaryLength: float = 10, 
                 agentsNum: int=1000, dt: float=0.01, speedV: float=3,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10, 
                 distribute: str = "uniform", randomSeed: int = 10, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.vecAnglePhi = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.speedV = speedV
        self.distanceD0 = distanceD0
        if distribute == "uniform":
            self.freqOmega = np.concatenate([
                np.random.uniform(1, 3, size=agentsNum // 2),
                np.random.uniform(-3, -1, size=agentsNum // 2)
            ])
        elif distribute == "normal":
            self.freqOmega = np.concatenate([
                np.random.normal(loc=3, scale=0.5, size=agentsNum // 2),
                np.random.normal(loc=-3, scale=0.5, size=agentsNum // 2)
            ])
        else:
            assert False, "Invalid distribution type"
        if agentsNum == 2:
            self.freqOmega = np.array([3, -3])

        self.distribute = distribute
        self.K = K
        self.J = J
        self.tqdm = tqdm
        self.savePath = savePath
        self.temp = np.zeros(agentsNum)
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.randomSeed = randomSeed
        self.overWrite = overWrite

        self.temp = {
            "deltaTheta": self.deltaTheta,
            "deltaPhi": self.deltaPhi
        }
        self.temp = {
            "dotTheta": self.dotTheta,
            "dotPhi": self.dotPhi
        }

    def plot(self, ax: plt.Axes = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        plt.quiver(
            self.positionX[:self.agentsNum // 2, 0], self.positionX[:self.agentsNum // 2, 1],
            np.cos(self.phaseTheta[:self.agentsNum // 2]), np.sin(self.phaseTheta[:self.agentsNum // 2]), color='tomato'
        )
        plt.quiver(
            self.positionX[self.agentsNum // 2:, 0], self.positionX[self.agentsNum // 2:, 1],
            np.cos(self.phaseTheta[self.agentsNum // 2:]), np.sin(self.phaseTheta[self.agentsNum // 2:]), color='dodgerblue'
        )
        plt.xlim(0, self.boundaryLength)
        plt.ylim(0, self.boundaryLength)

    @property
    def G(self):
        return self.distance_x(self.deltaX) <= self.distanceD0

    @property
    def deltaX(self) -> np.ndarray:
        return self._delta_x(self.positionX, self.positionX[:, np.newaxis], 
                             self.boundaryLength, self.halfBoundaryLength)

    @staticmethod
    @nb.njit
    def _delta_x(positionX: np.ndarray, others: np.ndarray,
                 boundaryLength: float, halfBoundaryLength: float) -> np.ndarray:
        subX = positionX - others
        return positionX - (
            others * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) + 
            (others - boundaryLength) * (subX < -halfBoundaryLength) + 
            (others + boundaryLength) * (subX > halfBoundaryLength)
        )

    @property
    def dotTheta(self):
        return self._dotTheta(self.freqOmega, 
                              self.temp["deltaTheta"], self.temp["deltaPhi"],
                              self.K, self.G)

    @staticmethod
    @nb.njit
    def _dotTheta(freqOmega: np.ndarray, deltaTheta: np.ndarray, deltaPhi: np.ndarray, 
                  K: float, G: np.ndarray):
        return freqOmega + K * np.sum(G * np.sin(deltaTheta) * np.cos(deltaPhi), axis=0) / deltaTheta.shape[0]

    @property
    def dotPhi(self) -> np.ndarray:
        return self._dotPhi(self.temp["deltaPhi"], self.temp["deltaTheta"], self.J, self.G)

    @staticmethod
    @nb.njit
    def _dotPhi(deltaPhi: np.ndarray, deltaTheta: np.ndarray, J: float, G: np.ndarray):
        return J * np.sum(G * np.sin(deltaPhi) * np.cos(deltaTheta), axis=0) / deltaPhi.shape[0]

    @property
    def deltaPhi(self) -> np.ndarray:
        """Phase difference between agents"""
        return self.vecAnglePhi - self.vecAnglePhi[:, np.newaxis]

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="dotTheta", value=pd.DataFrame(self.temp["dotTheta"]))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="dotPhi", value=pd.DataFrame(self.temp["dotPhi"]))
            self.store.append(key="vecAnglePhi", value=pd.DataFrame(self.vecAnglePhi))

    def update_temp(self):
        self.temp["deltaTheta"] = self.deltaTheta
        self.temp["deltaPhi"] = self.deltaPhi
        self.temp["dotTheta"] = self.dotTheta
        self.temp["dotPhi"] = self.dotPhi

    def update(self):
        self.positionX[:, 0] += self.speedV * np.cos(self.vecAnglePhi) * self.dt
        self.positionX[:, 1] += self.speedV * np.sin(self.vecAnglePhi) * self.dt
        self.positionX = np.mod(self.positionX, self.boundaryLength)

        self.phaseTheta += self.temp["dotTheta"] * self.dt
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

        self.vecAnglePhi += self.temp["dotPhi"] * self.dt
        self.vecAnglePhi = np.mod(self.vecAnglePhi + np.pi, 2 * np.pi) - np.pi

    def __str__(self) -> str:
        return (
            f"SelfPropSwarmalator_{self.distribute}_"
            f"{self.K:.3f}_{self.J:.3f}"
            f"{self.distanceD0:.2f}_{self.randomSeed}"
        )
        
    def close(self):
        if self.store is not None:
            self.store.close()