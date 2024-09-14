import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from typing import List, Tuple
from itertools import product
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
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class PatternFormation(Swarmalators2D):
    def __init__(self, strengthLambda: float, alpha: float, boundaryLength: float = 10, 
                 productRateK0: float = 1, decayRateKd: float = 1, c0: float = 5, 
                 chemotacticStrengthBetaR: float = 1, diffusionRateDc: float = 1, 
                 epsilon: float = 10, cellNumInLine: int = 50, 
                 typeA: str = "distanceWgt", agentsNum: int=1000, dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10, 
                 distribution: str = "uniform", randomSeed: int = 10, overWrite: bool = False) -> None:
        assert distribution in ["uniform"]
        assert typeA in ["distanceWgt"]

        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.c = np.random.rand(cellNumInLine, cellNumInLine)
        self.cellNumInLine = cellNumInLine
        self.cPosition = np.array(list(product(np.linspace(0, boundaryLength, cellNumInLine), repeat=2)))
        self.dx = boundaryLength / (cellNumInLine - 1)
        self.agentsNum = agentsNum
        self.productRateK0 = productRateK0
        self.decayRateKd = decayRateKd
        self.diffusionRateDc = diffusionRateDc
        self.chemotacticStrengthBetaR = chemotacticStrengthBetaR
        self.c0 = c0
        self.epsilon = epsilon
        self.dt = dt
        self.speedV = 3
        self.alpha = alpha
        if distribution == "uniform":
            self.omegaTheta = np.concatenate([
                np.random.uniform(1, 3, size=agentsNum // 2),
                np.random.uniform(-3, -1, size=agentsNum // 2)
            ])
        elif distribution == "normal":
            self.omegaTheta = np.concatenate([
                np.random.normal(loc=3, scale=0.5, size=agentsNum // 2),
                np.random.normal(loc=-3, scale=0.5, size=agentsNum // 2)
            ])

        self.typeA = typeA
        self.distribution = distribution
        self.strengthLambda = strengthLambda
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.randomSeed = randomSeed
        self.overWrite = overWrite

        self.tempPointTheta = np.zeros(agentsNum)
        self.tempPointC = np.zeros((cellNumInLine, cellNumInLine))
        self.tempDistanceCX: np.ndarray = self.distance_x(self.deltaCX)

    def plot(self, ax: plt.Axes = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.quiver(
            self.positionX[:self.agentsNum // 2, 0], self.positionX[:self.agentsNum // 2, 1],
            np.cos(self.phaseTheta[:self.agentsNum // 2]), np.sin(self.phaseTheta[:self.agentsNum // 2]), color='tomato'
        )
        ax.quiver(
            self.positionX[self.agentsNum // 2:, 0], self.positionX[self.agentsNum // 2:, 1],
            np.cos(self.phaseTheta[self.agentsNum // 2:]), np.sin(self.phaseTheta[self.agentsNum // 2:]), color='dodgerblue'
        )
        ax.set_xlim(0, self.boundaryLength)
        ax.set_ylim(0, self.boundaryLength)

    def plot_field(self, ax: plt.Axes = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.contourf(self.c, cmap='viridis', levels=50)

    @property
    def A(self):
        if self.typeA == "heaviside":
            return self.distance_x(self.deltaX) <= self.alpha
        elif self.typeA == "distanceWgt":
            return np.exp(-self.distance_x(self.deltaX) / self.alpha)

    @property
    def productC(self):
        if self.typeA == "heaviside":
            value = (self.tempDistanceCX <= self.alpha).mean(axis=0) * self.productRateK0
        elif self.typeA == "distanceWgt":
            value = np.exp(-self.tempDistanceCX / self.alpha).mean(axis=0) * self.productRateK0
        return self._reshape_product_c(value, self.cellNumInLine)

    @staticmethod
    @nb.njit
    def _reshape_product_c(cPosition: np.ndarray, cellNumInLine: int):
        return np.reshape(cPosition, (cellNumInLine, cellNumInLine))

    @property
    def decayC(self):
        return self.c * self.decayRateKd
    
    @property
    def nabla2C(self):
        center = -self.c
        direct_neighbors = 0.20 * (
            np.roll(self.c, 1, axis=0)
            + np.roll(self.c, -1, axis=0)
            + np.roll(self.c, 1, axis=1)
            + np.roll(self.c, -1, axis=1)
        )
        diagonal_neighbors = 0.05 * (
            np.roll(np.roll(self.c, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.c, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.c, -1, axis=0), -1, axis=1)
            + np.roll(np.roll(self.c, 1, axis=0), -1, axis=1)
        )

        out_array = center + direct_neighbors + diagonal_neighbors
        return out_array / (self.dx ** 2)
    
    @property
    def nablaC(self):
        return np.array([
            (np.roll(self.c, 1, axis=1) - np.roll(self.c, -1, axis=1)) / (2 * self.dx), 
            (np.roll(self.c, 1, axis=0) - np.roll(self.c, -1, axis=0)) / (2 * self.dx)
        ]).transpose(1, 2, 0)

    @property
    def diffusionC(self):
        return self.diffusionRateDc * self.nabla2C

    @property
    def growthLimitC(self):
        return self.epsilon * (self.c0 - self.c) ** 3        

    @property
    def deltaCX(self):
        return self._delta_x(self.cPosition, self.positionX[:, np.newaxis], 
                             self.boundaryLength, self.halfBoundaryLength)

    @property
    def deltaX(self) -> np.ndarray:
        return self._delta_x(self.positionX, self.positionX[:, np.newaxis], 
                             self.boundaryLength, self.halfBoundaryLength)

    @staticmethod
    @nb.njit
    def _delta_x(positionX: np.ndarray, others: np.ndarray,
                 boundaryLength: float, halfBoundaryLength: float) -> np.ndarray:
        subX = positionX - others
        return (
            subX * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) +
            (subX + boundaryLength) * (subX < -halfBoundaryLength) +
            (subX - boundaryLength) * (subX > halfBoundaryLength)
        )

    @property
    def chemotactic(self):
        gradC = self.nablaC.reshape(-1, 2)
        localGradC = gradC[self.tempDistanceCX.argmin(axis=1)]
        return self.chemotacticStrengthBetaR * (
            np.cos(self.phaseTheta) * localGradC[:, 1] - 
            np.sin(self.phaseTheta) * localGradC[:, 0]
        )

    @property
    def pointTheta(self):
        return self._pointTheta(self.phaseTheta, self.omegaTheta, self.chemotactic, 
                                self.strengthLambda, self.A)

    @staticmethod
    @nb.njit
    def _pointTheta(phaseTheta: np.ndarray, omegaTheta: np.ndarray, 
                    chemotactic: np.ndarray, strengthLambda: float, 
                    A: np.ndarray):
        adjMatrixTheta = (
            np.repeat(phaseTheta, phaseTheta.shape[0])
            .reshape(phaseTheta.shape[0], phaseTheta.shape[0])
        )
        k1 = omegaTheta + chemotactic + strengthLambda * np.sum(A * np.sin(
            adjMatrixTheta - phaseTheta
        ), axis=0)
        return k1
    
    @property
    def pointC(self):
        return self.productC - self.decayC + self.diffusionC + self.growthLimitC

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="pointTheta", value=pd.DataFrame(self.tempPointTheta))
            self.store.append(key="c", value=pd.DataFrame(self.c))
            self.store.append(key="pointC", value=pd.DataFrame(self.tempPointC))

    def update(self):
        self.tempDistanceCX = self.distance_x(self.deltaCX)
        self.positionX[:, 0] += self.speedV * np.cos(self.phaseTheta) * self.dt
        self.positionX[:, 1] += self.speedV * np.sin(self.phaseTheta) * self.dt
        self.positionX = np.mod(self.positionX, self.boundaryLength)
        self.tempPointTheta = self.pointTheta
        self.tempPointC = self.pointC
        self.phaseTheta += self.tempPointTheta * self.dt
        self.c += self.tempPointC * self.dt
        self.c[self.c < 0] = 0
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def __str__(self) -> str:
        
        name =  (
            f"PF_K{self.strengthLambda:.3f}_a{self.alpha:.2f}"
            f"_b{self.chemotacticStrengthBetaR:.1f}"
            f"_r{self.randomSeed}"
        )
        
        return name

    def close(self):
        if self.store is not None:
            self.store.close()


class GSPatternFormation(PatternFormation):
    def __init__(self, strengthLambda: float, alpha: float, boundaryLength: float = 10, 
                 productRateK0: float = 1, decayRateKd: float = 1, c0: float = 5, 
                 chemoBetaU: float = 1, chemoBetaV: float = 1, 
                 diffusionRateDc: float = 1, epsilon: float = 10, cellNumInLine: int = 50, 
                 typeA: str = "distanceWgt", agentsNum: int=1000, dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10, 
                 distribution: str = "uniform", randomSeed: int = 10, overWrite: bool = False) -> None:
        
        super().__init__(
            strengthLambda, alpha, boundaryLength, productRateK0, decayRateKd, c0, 
            0, diffusionRateDc, epsilon, cellNumInLine, typeA, 
            agentsNum, dt, tqdm, savePath, shotsnaps, distribution, randomSeed, overWrite
        )
        self.chemoBetaU = chemoBetaU
        self.chemoBetaV = chemoBetaV
        self.halfAgentsNum = agentsNum // 2
        self.u = np.random.rand(cellNumInLine, cellNumInLine)
        self.v = np.random.rand(cellNumInLine, cellNumInLine)
        
    @property
    def nablaU(self):
        return np.array([
            (np.roll(self.u, 1, axis=1) - np.roll(self.u, -1, axis=1)) / (2 * self.dx), 
            (np.roll(self.u, 1, axis=0) - np.roll(self.u, -1, axis=0)) / (2 * self.dx)
        ]).transpose(1, 2, 0)
    
    @property
    def nablaV(self):
        return np.array([
            (np.roll(self.v, 1, axis=1) - np.roll(self.v, -1, axis=1)) / (2 * self.dx), 
            (np.roll(self.v, 1, axis=0) - np.roll(self.v, -1, axis=0)) / (2 * self.dx)
        ]).transpose(1, 2, 0)

    @property
    def productU(self):
        if self.typeA == "heaviside":
            value = (self.tempDistanceCX <= self.alpha)[:self.halfAgentsNum].mean(axis=0) * self.productRateK0
        elif self.typeA == "distanceWgt":
            value = np.exp(-self.tempDistanceCX / self.alpha)[:self.halfAgentsNum].mean(axis=0) * self.productRateK0
        return self._reshape_product_c(value, self.cellNumInLine)
    
    @property
    def productV(self):
        if self.typeA == "heaviside":
            value = (self.tempDistanceCX <= self.alpha)[self.halfAgentsNum:].mean(axis=0) * self.productRateK0
        elif self.typeA == "distanceWgt":
            value = np.exp(-self.tempDistanceCX / self.alpha)[self.halfAgentsNum:].mean(axis=0) * self.productRateK0
        return self._reshape_product_c(value, self.cellNumInLine)

    @property
    def chemotactic(self):
        idxs = (self.positionX / self.dx).round().astype(int)
        localGradU = self.nablaU[idxs[:, 0], idxs[:, 1]]
        localGradV = self.nablaV[idxs[:, 0], idxs[:, 1]]
        return self.chemoBetaU * (
            np.cos(self.phaseTheta) * localGradU[:, 1] - 
            np.sin(self.phaseTheta) * localGradU[:, 0]
        ) + self.chemoBetaV * (
            np.cos(self.phaseTheta) * localGradV[:, 1] -
            np.sin(self.phaseTheta) * localGradV[:, 0]
        )
    
    @property
    def nabla2U(self):
        center = -self.u
        direct_neighbors = 0.20 * (
            np.roll(self.u, 1, axis=0)
            + np.roll(self.u, -1, axis=0)
            + np.roll(self.u, 1, axis=1)
            + np.roll(self.u, -1, axis=1)
        )
        diagonal_neighbors = 0.05 * (
            np.roll(np.roll(self.u, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.u, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.u, -1, axis=0), -1, axis=1)
            + np.roll(np.roll(self.u, 1, axis=0), -1, axis=1)
        )

        out_array = center + direct_neighbors + diagonal_neighbors
        return out_array / (self.dx ** 2)
    
    @property
    def nabla2V(self):
        center = -self.v
        direct_neighbors = 0.20 * (
            np.roll(self.v, 1, axis=0)
            + np.roll(self.v, -1, axis=0)
            + np.roll(self.v, 1, axis=1)
            + np.roll(self.v, -1, axis=1)
        )
        diagonal_neighbors = 0.05 * (
            np.roll(np.roll(self.v, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.v, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.v, -1, axis=0), -1, axis=1)
            + np.roll(np.roll(self.v, 1, axis=0), -1, axis=1)
        )

        out_array = center + direct_neighbors + diagonal_neighbors
        return out_array / (self.dx ** 2)
    
    @property
    def diffusionU(self):
        return self.diffusionRateDc * self.nabla2U
    
    @property
    def diffusionV(self):
        return self.diffusionRateDc * self.nabla2V
    
    @property
    def pointU(self):
        return (
            self.productU 
            - self.u * self.v ** 2 
            - self.u * self.decayRateKd 
            + self.diffusionU
            + self.epsilon * (self.c0 - self.u) ** 3
        )
    
    @property
    def pointV(self):
        return (
            self.productV 
            + self.u * self.v ** 2 
            - self.v * self.decayRateKd 
            + self.diffusionV
            + self.epsilon * (self.c0 - self.v) ** 3
        )

    def update(self):
        self.tempDistanceCX = self.distance_x(self.deltaCX)
        self.positionX[:, 0] += self.speedV * np.cos(self.phaseTheta) * self.dt
        self.positionX[:, 1] += self.speedV * np.sin(self.phaseTheta) * self.dt
        self.positionX = np.mod(self.positionX, self.boundaryLength)
        self.tempPointTheta = self.pointTheta
        self.tempPointU = self.pointU
        self.tempPointV = self.pointV
        self.phaseTheta += self.tempPointTheta * self.dt
        self.u += self.tempPointU * self.dt
        self.v += self.tempPointV * self.dt
        self.u[self.u < 0] = 0
        self.v[self.v < 0] = 0
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="pointTheta", value=pd.DataFrame(self.tempPointTheta))
            self.store.append(key="u", value=pd.DataFrame(self.u))
            self.store.append(key="v", value=pd.DataFrame(self.v))

    def __str__(self) -> str:
            
            name =  (
                f"GSPF_K{self.strengthLambda:.3f}_a{self.alpha:.2f}"
                f"_b{self.chemotacticStrengthBetaR:.1f}"
                f"_r{self.randomSeed}"
            )
            
            return name


class StateAnalysis:
    def __init__(self, model: PatternFormation = None, classDistance: float = 2, 
                 lookIndex: int = -1, showTqdm: bool = False):
        
        self.classDistance = classDistance
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm
        
        if model is not None:
            self.model = model
            targetPath = f"{self.model.savePath}/{self.model}.h5"
            totalPositionX = pd.read_hdf(targetPath, key="positionX")
            totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
            totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
            
            TNum = totalPositionX.shape[0] // self.model.agentsNum
            self.TNum = TNum
            self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
            self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)
            self.totalPointTheta = totalPointTheta.values.reshape(TNum, self.model.agentsNum)

            if self.showTqdm:
                self.iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
            else:
                self.iterObject = range(1, self.totalPhaseTheta.shape[0])

    def get_state(self, index: int = -1):
        positionX = self.totalPositionX[index]
        phaseTheta = self.totalPhaseTheta[index]
        pointTheta = self.totalPointTheta[index]

        return positionX, phaseTheta, pointTheta

    @staticmethod
    @nb.njit
    def _calc_centers(positionX, phaseTheta, pointTheta, speedV, dt):
        centers = np.zeros((positionX.shape[0], 2))
        centers[:, 0] = positionX[:, 0] - speedV * dt / pointTheta * np.sin(phaseTheta)
        centers[:, 1] = positionX[:, 1] + speedV * dt / pointTheta * np.cos(phaseTheta)

        return centers

    @property
    def centers(self):
        
        lastPositionX, lastPhaseTheta, lastPointTheta = self.get_state(self.lookIndex)
        
        centers = self._calc_centers(
            lastPositionX, lastPhaseTheta, lastPointTheta, self.model.speedV, self.model.dt
        )

        return np.mod(centers, self.model.boundaryLength)

    @property
    def centersNoMod(self):
            
        lastPositionX, lastPhaseTheta, lastPointTheta = self.get_state(self.lookIndex)
        
        centers = self._calc_centers(
            lastPositionX, lastPhaseTheta, lastPointTheta, self.model.speedV, self.model.dt
        )

        return centers
         
    
    def adj_distance(self, positionX, others):
        return self._adj_distance(
            positionX, others, self.model.boundaryLength, self.model.halfBoundaryLength
        )

    @staticmethod
    @nb.njit
    def _adj_distance(positionX, others, boundaryLength, halfLength):
        subX = positionX - others
        adjustOthers = (
            others * (-halfLength <= subX) * (subX <= halfLength) + 
            (others - boundaryLength) * (subX < -halfLength) + 
            (others + boundaryLength) * (subX > halfLength)
        )
        adjustSubX = positionX - adjustOthers
        return np.sqrt(np.sum(adjustSubX ** 2, axis=-1))
    
    @staticmethod
    @nb.njit
    def _calc_classes(centers, classDistance, totalDistances):
        classes = [[0]]
        classNum = 1
        nonClassifiedOsci = np.arange(1, centers.shape[0])

        for i in nonClassifiedOsci:
            newClass = True

            for classI in range(len(classes)):
                distance = classDistance
                for j in classes[classI]:
                    if totalDistances[i, j] < distance:
                        distance = totalDistances[i, j]
                if distance < classDistance:
                    classes[classI].append(i)
                    newClass = False
                    break

            if newClass:
                classNum += 1
                classes.append([i])

        newClasses = [classes[0]]

        for subClass in classes[1:]:
            newClass = True
            for newClassI in range(len(newClasses)):
                distance = classDistance
                for i in newClasses[newClassI]:
                    for j in subClass:
                        if totalDistances[i, j] < distance:
                            distance = totalDistances[i, j]
                if distance < classDistance:
                    newClasses[newClassI] += subClass
                    newClass = False
                    break

            if newClass:
                newClasses.append(subClass)
    
        return newClasses

    def get_classes_centers(self):
        centers = self.centers
        classes = self._calc_classes(
            centers, self.classDistance, self.adj_distance(centers, centers[:, np.newaxis])
        )
        return {i + 1: classes[i] for i in range(len(classes))}, centers

    def plot_spatial(self, ax: plt.Axes = None, oscis: np.ndarray = None, index: int = -1, **kwargs):
        positionX, phaseTheta, pointTheta = self.get_state(index)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        if oscis is None:
            oscis = np.arange(self.model.agentsNum)

        ax.quiver(
            positionX[oscis, 0], positionX[oscis, 1],
            np.cos(phaseTheta[oscis]), np.sin(phaseTheta[oscis]), **kwargs
        )
        ax.set_xlim(0, self.model.boundaryLength)
        ax.set_ylim(0, self.model.boundaryLength)    

    def plot_centers(self, ax: plt.Axes = None, index: int = -1):
        positionX, phaseTheta, pointTheta = self.get_state(index)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        quiverColors = ["#FF4B4E"] * 500 + ["#414CC7"] * 500
        ax.quiver(
            positionX[:, 0], positionX[:, 1],
            np.cos(phaseTheta[:]), np.sin(phaseTheta[:]), color=quiverColors, alpha=0.8
        )
        centerColors = ["#FBDD85"] * 500 + ["#80A6E2"] * 500
        centers = self.centers
        ax.scatter(centers[:, 0], centers[:, 1], color=centerColors, s=5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)    
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.grid(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.set_xlabel(r"$x$", fontsize=16)
        ax.set_ylabel(r"$y$", fontsize=16, rotation=0)
    
    def calc_order_parameter_R(self, state: Tuple[np.ndarray, np.ndarray, np.ndarray] = None):
        if state is None:
            _, phaseTheta, _ = self.get_state(self.lookIndex)
        else:
            _, phaseTheta, _ = state

        return np.abs(np.sum(np.exp(1j * phaseTheta))) / phaseTheta.size