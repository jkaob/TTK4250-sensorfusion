#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic models to be used with eg. EKF.

"""
# %%
from typing import Optional, Sequence
from typing_extensions import Final, Protocol
from dataclasses import dataclass, field

import numpy as np

# %% the dynamic models interface declaration


class DynamicModel(Protocol):
    n: int
    def f(self, x: np.ndarray, Ts: float) -> np.ndarray: ...
    def F(self, x: np.ndarray, Ts: float) -> np.ndarray: ...
    def Q(self, x: np.ndarray, Ts: float) -> np.ndarray: ...

# %%


@dataclass
class WhitenoiseAccelleration:
    """
    A white noise accelereation model also known as CV, states are position and speed.

    The model includes the discrete prediction equation f, its Jacobian F, and
    the process noise covariance Q as methods.
    """
    # noise standard deviation
    sigma: float
    dim: int = 2    # number of dimensions
    n: int = 4      # number of states

    def f(self, 
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        f = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]])@x
        return f

    def F(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """ Calculate the transition function jacobian for Ts time units at x."""
        return np.array([[1, 0, Ts, 0],[0, 1, 0, Ts],[0, 0, 1, 0],[0, 0, 0, 1]])

    def Q(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """
        Calculate the Ts time units transition Covariance. """
        Q = Ts*np.array([[Ts**2/3, 0, Ts/2, 0],[0, Ts**2/3, 0, Ts/2],[Ts/2, 0, 1, 0],[0, Ts/2, 0, 1]])*(self.sigma)**2
        return Q
