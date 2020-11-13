import numpy as np
import scipy.linalg as la
import utils
import time
import itertools

from scipy.linalg import block_diag
from typing import Tuple
from utils import rotmat2d
from JCBB import JCBB
from numpy import matlib as ml


# import line_profiler
# import atexit

# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


class EKFSLAM:
    def __init__(
        self,
        Q,
        R,
        do_asso=False,
        alphas=np.array([0.001, 0.0001]),
        sensor_offset=np.zeros(2),
        #max_range=100
    ):
        self.Q = Q
        self.R = R
        self.do_asso = do_asso
        self.alphas = alphas
        self.sensor_offset = sensor_offset
        #self.max_range = max_range

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Add the odometry u to the robot state x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray, shape = (3,)
            the predicted state
        """
        x_p, y_p, psi_p = x
        u_k, v_k, phi_k = u
        
        x_n = x_p + u_k * np.cos(psi_p) - v_k * np.sin(psi_p)
        y_n = y_p + u_k * np.sin(psi_p) + v_k * np.cos(psi_p)
        psi_n = utils.wrapToPi(psi_p + phi_k) # Wrapping heading angle between (-pi, pi)
        
        xpred = np.array([x_n, y_n, psi_n]) # eq (11.7)
        
        assert xpred.shape == (3,), "EKFSLAM.f: wrong shape for xpred"
        return xpred

    def Fx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. x.
        """
        psi = x[2]
        u_k, v_k, _ = u
        
        f_13 = -u_k * np.sin(psi) - v_k * np.cos(psi)
        f_23 = u_k * np.cos(psi) - v_k * np.sin(psi)
        Fx = np.eye(3)
        Fx[:2, 2] = f_13, f_23

        assert Fx.shape == (3, 3), "EKFSLAM.Fx: wrong shape"
        return Fx

    def Fu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to u.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. u.
        """
        psi = x[2]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        Fu = np.eye(3)
        Fu[0, :2] = cos_psi, sin_psi
        Fu[1, :2] = -sin_psi, cos_psi

        assert Fu.shape == (3, 3), "EKFSLAM.Fu: wrong shape"
        return Fu

    def predict(
        self, eta: np.ndarray, P: np.ndarray, z_odo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the robot state using the zOdo as odometry the corresponding state&map covariance.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z_odo : np.ndarray, shape=(3,)
            the measured odometry

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes= (3 + 2*#landmarks,), (3 + 2*#landmarks,)*2
            predicted mean and covariance of eta.
        """
        #print('predict')
        # check inout matrix
        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P input"
        assert np.all(
            np.linalg.eigvals(P) >= 0
        ), "EKFSLAM.predict: non-positive eigen values in P input"
        assert (
            eta.shape * 2 == P.shape
        ), "EKFSLAM.predict: input eta and P shape do not match"
        x = eta[:3]
        
        etapred = np.empty_like(eta)
        etapred[:3] = self.f(x, z_odo) # robot state prediction
        etapred[3:] = eta[3:] # landmarks: no effect
        
        Fu = self.Fu(x, None)
        Fx = self.Fx(x, z_odo)
        # evaluate covariance prediction in place to save computation
        # only robot state changes, so only rows and colums of robot state needs changing
        P_superdiag = Fx @ P[:3, 3:]
        # robot cov prediction
        P[:3, :3] = Fx @ P[:3, :3] @ Fx.T + self.Q
        P[:3, 3:] = P_superdiag # robot-map covariance prediction
        P[3:, :3] = P_superdiag.T # map-robot covariance: transpose of the above
        #P += Fx @ self.R @ Fx.T # Assuming self.R is R_t^x
        
        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P"
        assert np.all(
            np.linalg.eigvals(P) > 0
        ), "EKFSLAM.predict: non-positive eigen values"
        assert (
            etapred.shape * 2 == P.shape
        ), "EKFSLAM.predict: calculated shapes does not match"
        return etapred, P

    def h(self, eta: np.ndarray) -> np.ndarray:
        """Predict all the landmark positions in sensor frame.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks,)
            The landmarks in the sensor frame.
        """
        #print('h')
        # extract states and map
        x = eta[:3]
        ## reshape map (2, #landmarks), m[:, j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T
        Rot = rotmat2d(-x[2])
        
        # relative position of landmark to sensor on robot in world frame
        delta_m = m - x[:2, None] - Rot.T @ self.sensor_offset[:, None]
        # predicted measurements in cartesian coordinates
        zpredcart = Rot @ delta_m 
        zpred_r = la.norm(zpredcart, axis=0) # ranges
        zpred_theta = np.arctan2(zpredcart[1], zpredcart[0]) # bearings
        
        zpred = np.stack((zpred_r, zpred_theta))
        zpred = zpred.T.ravel() # stack measurements along one dimension    
        
        assert (
            zpred.ndim == 1 and zpred.shape[0] == eta.shape[0] - 3
        ), "SLAM.h: Wrong shape on zpred"
        return zpred

    def H(self, eta: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of h.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """
        # extract states and map
        x = eta[:3]
        # reshape map (2, #landmarks), m[j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T
        numM = m.shape[1]
        Rot = rotmat2d(-x[2])
        
        delta_m = m - x[:2, None]
        zc = delta_m - Rot @ self.sensor_offset[:, None]
        zpred = self.h(eta).reshape(-1, 2).T # predicted measurements
        zr = zpred[0] # ranges

        # Allocate H and set submatrices as memory views into H
        H = np.zeros((2 * numM, 3 + 2 * numM))
        Hx = H[:, :3]
        Hm = H[:, 3:]
        
        Rpihalf = rotmat2d(np.pi / 2)
        jac_z_cb = -np.eye(2, 3)
        #jac_z_cb[:, 2] = -Rpihalf @ delta_m # [-I_2 -Rph(m^i - rho_k)]
        for i in range(numM):  # But this whole loop can be vectorized
            """if zr[i] > self.max_range:
                Hx[ind] = zc[:, i]# @ jac_z_cb / zr[i]
                Hx[ind + 1] = zc[:, i]# @ Rpihalf.T @ jac_z_cb / zr[i]**2
                Hm[inds, inds] = -Hx[inds, :2]
                continue"""
            ind = 2 * i # starting position of the ith landmark into H
            inds = slice(ind, ind + 2)  # the inds slice for the ith landmark into H
            
            jac_z_cb[:, 2] = -Rpihalf @ delta_m[:, i] # > do this before loop
            Hx[ind] = zc[:, i] @ jac_z_cb / zr[i]
            Hx[ind + 1] = zc[:, i] @ Rpihalf.T @ jac_z_cb / zr[i]**2
            Hm[inds, inds] = -Hx[inds, :2]

        assert (H.shape == (2 * numM, 3 + 2 * numM)
        ), "EKFSLAM.H: Wrong shape on calculated H"
        return H

    def add_landmarks(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate new landmarks, their covariances and add them to the state.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z : np.ndarray, shape(2 * #newlandmarks,)
            A set of measurements to create landmarks for

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes=(3 + 2*(#landmarks + #newlandmarks,), (3 + 2*(#landmarks + #newlandmarks,)*2
            eta with new landmarks appended, and its covariance
        """
        #print('add_landmark')
        assert z.ndim == 1, "SLAM.add_landmarks: z must be a 1d array"
        
        n = P.shape[0]
        numLmk = z.shape[0] // 2
        lmnew = np.empty_like(z)
        Gx = np.empty((numLmk * 2, 3))
        Rall = np.zeros((numLmk * 2, numLmk * 2))
        I2 = np.eye(2) # Preallocate, used for Gx
        
        # For transforming landmark position into world frame
        sensor_offset_world = rotmat2d(eta[2]) @ self.sensor_offset 
        sensor_offset_world_der = rotmat2d(eta[2] + np.pi / 2) @ self.sensor_offset # For Gx

        for j in range(numLmk):
            ind = 2 * j
            inds = slice(ind, ind + 2)
            
            zj = z[inds]
            rot = rotmat2d(zj[1] + eta[2]) # psi + psi
            # calculate position of new landmark in world frame
            lmnew[inds] = zj[0] * rot[:, 0] + eta[:2] + sensor_offset_world

            Gx[inds, :2] = I2
            Gx[inds, 2] = zj[0] * rot[:, 1] + sensor_offset_world_der #.reshape(-1, 1) #+ sensor_offset_world_der
            Gz = rot @ np.diag([1, zj[0]])
            Rall[inds, inds] = Gz @ self.R @ Gz.T # pol2cart on measurement cov

        assert len(lmnew) % 2 == 0, "SLAM.add_landmark: lmnew not even length"
        # append new landmarks to state vector
        #etaadded = np.append(eta, lmnew)
        etaadded = np.array([*eta, *lmnew])
        Padded = la.block_diag(P, Gx @ P[:3, :3] @ Gx.T + Rall)
        Padded[n:, :n] = Gx @ P[:3, :]
        Padded[:n, n:] = Padded[n:, :n].T

        assert (
            etaadded.shape * 2 == Padded.shape
        ), "EKFSLAM.add_landmarks: calculated eta and P has wrong shape"
        assert np.allclose(
            Padded, Padded.T
        ), "EKFSLAM.add_landmarks: Padded not symmetric"
        assert np.all(
            np.linalg.eigvals(Padded) >= 0
        ), "EKFSLAM.add_landmarks: Padded not PSD"

        return etaadded, Padded

    def associate(
        self, z: np.ndarray, zpred: np.ndarray, H: np.ndarray, S: np.ndarray,
    ):  # -> Tuple[*((np.ndarray,) * 5)]:
        """Associate landmarks and measurements, and extract correct matrices for these.

        Parameters
        ----------
        z : np.ndarray,
            The measurements all in one vector
        zpred : np.ndarray
            Predicted measurements in one vector
        H : np.ndarray
            The measurement Jacobian matrix related to zpred
        S : np.ndarray
            The innovation covariance related to zpred

        Returns
        -------
        Tuple[*((np.ndarray,) * 5)]
            The extracted measurements, the corresponding zpred, H, S and the associations.

        Note
        ----
        See the associations are calculated  using JCBB. See this function for documentation
        of the returned association and the association procedure.
        """
        #print('associate')

        if self.do_asso:
            # Associate
            a = JCBB(z, zpred, S, self.alphas[0], self.alphas[1])

            # Extract associated measurements
            zinds = np.empty_like(z, dtype=bool)
            zinds[::2] = a > -1  # -1 means no association
            zinds[1::2] = zinds[::2]
            zass = z[zinds]

            # extract and rearange predicted measurements and cov
            zbarinds = np.empty_like(zass, dtype=int)
            zbarinds[::2] = 2 * a[a > -1]
            zbarinds[1::2] = 2 * a[a > -1] + 1

            zpredass = zpred[zbarinds]
            Sass = S[zbarinds][:, zbarinds]
            Hass = H[zbarinds]

            assert zpredass.shape == zass.shape
            assert Sass.shape == zpredass.shape * 2
            assert Hass.shape[0] == zpredass.shape[0]

            return zass, zpredass, Hass, Sass, a
        else:
            # should one do something her
            pass

    def update(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Update eta and P with z, associating landmarks and adding new ones.

        Parameters
        ----------
        eta : np.ndarray
            [description]
        P : np.ndarray
            [description]
        z : np.ndarray, shape=(#detections, 2)
            [description]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, np.ndarray]
            [description]
        """
        #print('update')
        
        numLmk = (eta.size - 3) // 2
        assert (len(eta) - 3) % 2 == 0, "EKFSLAM.update: landmark length not even"

        if numLmk > 0:
            # Prediction and innovation covariance
            zpred = self.h(eta)
            H = self.H(eta)
            
            #R_lg = la.block_diag(*[self.R] * numLmk)
            R_lg =  np.diag(
                np.diagonal(ml.repmat(self.R, numLmk, numLmk))
            )
            S = H @ P @ H.T + R_lg
            assert (
                S.shape == zpred.shape * 2
            ), "EKFSLAM.update: wrong shape on either S or zpred"
            z = z.ravel()  # 2D -> flat

            # Perform data association
            za, zpred, Ha, Sa, a = self.associate(z, zpred, H, S)

            # No association could be made, so skip update
            if za.shape[0] == 0:
                etaupd = eta
                Pupd = P
                NIS = 1 # TODO: beware this one when analysing consistency.
            else:
                # Create the associated innovation
                v = za.ravel() - zpred  # za: 2D -> flat
                v[1::2] = utils.wrapToPi(v[1::2])
                
                # Kalman mean update
                chol_inv = np.linalg.inv(np.linalg.cholesky(Sa))
                Sa_inv = np.dot(chol_inv.T, chol_inv)
                #S_cho_factors = la.cho_factor(Sa)
                #W = la.cho_solve(S_cho_factors, Ha @ P).T
                W = P @ Ha.T @ Sa_inv # Kalman gain using S_cho_factors
                #W = la.solve(Sa.T, Ha @ P.T).T
                etaupd = eta + W @ v # Kalman update

                # Kalman cov update: use Joseph form for stability
                #R_lg = la.block_diag(*[self.R] * (len(za)//2))
                jo = -W @ Ha
                jo[np.diag_indices(jo.shape[0])] += 1
                
                #Pupd = jo @ P @ jo.T + W @ R_lg @ W.T
                Pupd = jo @ P #@ jo.T + W @ R_lg @ W.T # Kalman update. Main workload on VP
                # calculate NIS using chol factors
                NIS = v.T @ Sa_inv @ v 
                #NIS = v.T @ la.cho_solve(S_cho_factors, v)
                
                # When tested, remove for speed
                assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd not symmetric"
                assert np.all(
                    np.linalg.eigvals(Pupd) > 0
                ), "EKFSLAM.update: Pupd not positive definite"

        else:  # All measurements are new landmarks,
            a = np.full(z.shape[0], -1)
            z = z.flatten()
            NIS = 1 # beware this one, you can change the value to for instance 1
            etaupd = eta
            Pupd = P
            #print('all new')
            
        # Create new landmarks if any is available
        if self.do_asso:
            is_new_lmk = a == -1
            if np.any(is_new_lmk):
                z_new_inds = np.empty_like(z, dtype=bool)
                z_new_inds[::2] = is_new_lmk
                z_new_inds[1::2] = is_new_lmk
                z_new = z[z_new_inds]
                etaupd, Pupd = self.add_landmarks(etaupd, Pupd, z_new)

        assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd must be symmetric"
        assert np.all(np.linalg.eigvals(Pupd) >= 0), "EKFSLAM.update: Pupd must be PSD"

        return etaupd, Pupd, NIS, a

    @classmethod
    def NEESes(cls, x: np.ndarray, P: np.ndarray, x_gt: np.ndarray,) -> np.ndarray:
        """Calculates the total NEES and the NEES for the substates
        Args:
            x (np.ndarray): The estimate
            P (np.ndarray): The state covariance
            x_gt (np.ndarray): The ground truth
        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties
        Returns:
            np.ndarray: NEES for [all, position, heading], shape (3,)
        """
        assert x.shape == (3,), f"EKFSLAM.NEES: x shape incorrect {x.shape}"
        assert P.shape == (3, 3), f"EKFSLAM.NEES: P shape incorrect {P.shape}"
        assert x_gt.shape == (3,), f"EKFSLAM.NEES: x_gt shape incorrect {x_gt.shape}"

        d_x = x - x_gt
        d_x[2] = utils.wrapToPi(d_x[2])
        assert (
            -np.pi <= d_x[2] <= np.pi
        ), "EKFSLAM.NEES: error heading must be between (-pi, pi)"

        d_p = d_x[0:2]
        P_p = P[0:2, 0:2]
        assert d_p.shape == (2,), "EKFSLAM.NEES: d_p must be 2 long"
        d_heading = d_x[2]  # Note: scalar
        assert np.ndim(d_heading) == 0, "EKFSLAM.NEES: d_heading must be scalar"
        P_heading = P[2, 2]  # Note: scalar
        assert np.ndim(P_heading) == 0, "EKFSLAM.NEES: P_heading must be scalar"

        # NB: Needs to handle both vectors and scalars! Additionally, must handle division by zero
        NEES_all = d_x @ (np.linalg.solve(P, d_x))
        NEES_pos = d_p @ (np.linalg.solve(P_p, d_p))
        try:
            NEES_heading = d_heading ** 2 / P_heading
        except ZeroDivisionError:
            NEES_heading = 1.0 # TODO: beware

        NEESes = np.array([NEES_all, NEES_pos, NEES_heading])
        NEESes[np.isnan(NEESes)] = 1.0  # We may divide by zero, # TODO: beware

        assert np.all(NEESes >= 0), "ESKF.NEES: one or more negative NEESes"
        return NEESes

