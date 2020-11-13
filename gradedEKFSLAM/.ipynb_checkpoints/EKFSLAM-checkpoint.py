from typing import Tuple
from scipy.linalg import block_diag
import numpy as np
import scipy.linalg as la
import utils
from utils import rotmat2d
from JCBB import JCBB

from sympy import Matrix, init_printing
init_printing()

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
        Q_process = None
    ):

        self.Q = Q
        self.R = R
        self.do_asso = do_asso
        self.alphas = alphas
        self.sensor_offset = sensor_offset

        self.Q_process = Q_process if Q_process else np.empty_like(Q)


    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Add the odometry u to the robot state x.
        Parameters
         x : np.ndarray, shape=(3,) the robot state
         u : np.ndarray, shape=(3,) the odometry
        Returns:
         np.ndarray, shape = (3,) the predicted state
        """
        # eq (11.7). Should wrap heading angle between (-pi, pi), see utils.wrapToPi
        #psi   = wrapToPi(x[2])
        xpred = np.array([  x[0] + u[0]*np.cos(x[2]) - u[1]*np.sin(x[2]),
                            x[1] + u[0]*np.sin(x[2]) + u[1]*np.cos(x[2]),
                            utils.wrapToPi(x[2] + u[2])])

        assert xpred.shape == (3,), "EKFSLAM.f: wrong shape for xpred"
        return xpred

    def Fx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to x.
        Parameters
         x : np.ndarray, shape=(3,) the robot state
         u : np.ndarray, shape=(3,) the odometry
        Returns:
         np.ndarray. The Jacobian of f wrt. x.
        """
        psi = x[2]
        Fx  = np.array([[1, 0, -u[0]*np.sin(psi) - u[1]*np.cos(psi) ] ,
                        [0, 1,  u[0]*np.cos(psi) - u[1]*np.sin(psi) ] ,
                        [0, 0,  1 ]])

        assert Fx.shape == (3, 3), "EKFSLAM.Fx: wrong shape"
        return Fx

    def Fu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to u.
        Parameters
         x : np.ndarray, shape=(3,)the robot state
         u : np.ndarray, shape=(3,)the odometry
        Returns
         np.ndarray The Jacobian of f wrt. u.
        """
        psi = x[2]
        Fu  = np.array([[np.cos(psi), -np.sin(psi), 0] ,
                        [np.sin(psi),  np.cos(psi), 0] ,
                        [0,             0,          1] ])

        assert Fu.shape == (3, 3), "EKFSLAM.Fu: wrong shape"
        return Fu

    def predict(
        self, eta: np.ndarray, P: np.ndarray, z_odo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the robot state using the zOdo as odometry the corresponding state&map covariance.

        Parameters
         eta   : np.ndarray, shape=(3 + 2*#landmarks,) :the robot state and map concatenated
         P     : np.ndarray, shape=(3 + 2*#landmarks,)*2 :the covariance of eta
         z_odo : np.ndarray, shape=(3,) :the measured odometry

        Returns:
         Tuple[np.ndarray, np.ndarray], shapes= (3 + 2*#landmarks,), (3 + 2*#landmarks,)*2
            predicted mean and covariance of eta.
        """
        # check inout matrix
        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P input"
        assert np.all(
            np.linalg.eigvals(P) >= 0
        ), "EKFSLAM.predict: non-positive eigen values in P input"
        assert (
            eta.shape * 2 == P.shape
        ), "EKFSLAM.predict: input eta and P shape do not match"

        x = eta[:3]

        etapred     = np.empty_like(eta)
        etapred[:3] = self.f(x, z_odo)      # robot state prediction
        etapred[3:] = eta[3:]               # landmarks: no effect

        Fx = self.Fx(etapred[:3], z_odo)    # jacobians
        Fu = self.Fu(etapred[:3], z_odo)

        # evaluate covariance prediction in place to save computation
        P[:3, :3] = Fx @ P[:3,:3] @ Fx.T + Fu @ self.Q @ Fu.T #+ self.Q_process
        P[:3, 3:] = Fx @ P[:3,3:]   # robot-map covariance prediction
        P[3:, :3] = P[:3, 3:].T #P[3:,:3] @ Fx.T # map-robot covariance: transpose of the above

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
         eta : (3 + 2 * #landmarks,) The robot state and landmarks stacked.
        Returns
         np.ndarray, shape=(2 * #landmarks,) The landmarks in the sensor frame.
        """
        x   = eta[:3]
        rho = x[:2]

        m   = eta[3:].reshape((-1, 2)).T
        R   = rotmat2d(-x[2])  # world->body
                               # R.T: body->world

        # relative position of landmark to sensor on robot (world)
        delta_m     = m - rho.reshape(2,1)  # No sensor offset here

        # predicted measurements in cartesian coordinates
        zpredcart   = R @ delta_m - self.sensor_offset.reshape(2,1)

        # Converting cartesian body -> polar body
        zpred_r     = np.apply_along_axis(np.linalg.norm, axis=0, arr=zpredcart)
        zpred_theta = utils.wrapToPi(np.arctan(zpredcart[1] / zpredcart[0]))
        zpred       = np.concatenate((zpred_r, zpred_theta),axis=0)

        # stack measurements along one dimension, [range1 bearing1 range2 bearing2 ...]
        zpred = zpred.T.ravel()

        assert (
            zpred.ndim == 1 and zpred.shape[0] == eta.shape[0] - 3
        ), "SLAM.h: Wrong shape on zpred"
        return zpred

    def H(self, eta: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of h.
        Parameters:
         eta : np.ndarray, shape=(3 + 2 * #landmarks,) The robot state and landmarks stacked.
        Returns:
         np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks) the jacobian of h wrt. eta.
        """

        x       = eta[:3]
        m       = eta[3:].reshape((-1, 2)).T ## reshape map (2, #landmarks), m[j] is the jth landmark
        numM    = m.shape[1]
        R       = rotmat2d(x[2])  # body->world

        # relative position of landmark to robot in world frame. m - rho that appears in (11.15) and (11.16)
        delta_m = m - x[:2].reshape(2,1)  # (world)
        zc      = delta_m - R @ self.sensor_offset.reshape(2,1)# measurements, cartesian coord (world frame)

        zpred   = self.h(eta)
        zr      = np.array([zpred[i] for i in range(0, len(zpred), 2)]) # ranges

        Rpihalf = rotmat2d(np.pi / 2)

        # In what follows you can be clever and avoid making this for all the landmarks you _know_
        # you will not detect (the maximum range should be available from the data).
        # But keep it simple to begin with.

        # Allocate H and set submatrices as memory views into H
        H  = np.zeros((2 * numM, 3 + 2 * numM)) # see eq (11.15), (11.16), (11.17)
        Hx = H[:, :3]  # slice view, setting elements of THIS will set H as well
        Hm = H[:, 3:]

        # proposed way is to go through landmarks one by one
        jac_z_cb = -np.eye(2, 3)  # preallocate and update this for some speed gain if looping
        for j in range(numM):  # But this whole loop can be vectorized
            ind  = 2 * j # starting postion of the ith landmark into H
            inds = slice(ind, ind + 2)  # the inds slice for the ith landmark into H

            zc_j    = zc[:,j].reshape(2,1)
            temp    = np.concatenate(( -np.eye(2), -Rpihalf @ delta_m[:,j].reshape(2,1) ), axis=1)
            Hx_top  = (zc_j.T / zr[j]) @ temp
            Hx_btm  = (zc_j.T @ Rpihalf.T / zr[j]**2) @ temp

            Hx[inds] = np.concatenate(( Hx_top, Hx_btm ),axis=0)

            #TODO : avoid making this for all the landmarks you _know_
            # you will not detect (the maximum range should be available from the data).
            Hm[inds, inds] = np.vstack(( zc_j.T / zr[j] ,
                                         zc_j.T @ Rpihalf / (zr[j]**2) ))

            #Hm[inds, inds]  = np.vstack(delta_m[:,j].T * 1 / la.norm(delta_m[:,j]),
            #                            delta_m[:,j].T @ Rpihalf * 1 / (la.norm(delta_m[:,j])**2) )

        #set some assertions here to make sure that some of the structure in H is correct
        assert (H.shape[0] == 2*numM and H.shape[1] ==  3+2*numM), "SLAM.H: Wrong shape of H"
        assert (Hx.shape[0] == 2*numM and Hx.shape[1] == 3), "SLAM.H: Wrong shape of Hx"
        return H

    def add_landmarks(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate new landmarks, their covariances and add them to the state.
        Parameters
         eta : np.ndarray, shape=(3 + 2*#landmarks,)   : the robot state and map concatenated
         P : np.ndarray, shape=(3 + 2*#landmarks,)*2   : the covariance of eta
         z : np.ndarray, shape(2 * #newlandmarks,)     : A set of measurements to create landmarks for
        Returns
         Tuple[np.ndarray, np.ndarray], shapes=(3 + 2*(#landmarks + #newlandmarks,), (3 + 2*(#landmarks + #newlandmarks,)*2
            eta with new landmarks appended, and its covariance
        """
        n = P.shape[0]
        assert z.ndim == 1, "SLAM.add_landmarks: z must be a 1d array"

        numLmk  = z.shape[0] // 2
        lmnew   = np.empty_like(z)

        Gx      = np.empty((numLmk * 2, 3))
        Rall    = np.zeros((numLmk * 2, numLmk * 2))

        I2 = np.eye(2) # Preallocate, used for Gx
        sensor_offset_world = rotmat2d(eta[2]) @ self.sensor_offset # For transforming landmark position into world frame
        sensor_offset_world_der = rotmat2d(eta[2] + np.pi / 2) @ self.sensor_offset # Used in Gx

        for j in range(numLmk):
            ind     = 2 * j
            inds    = slice(ind, ind + 2)
            zj      = z[inds]

            # calculate pos of new landmark in world frame
            zj_x, zj_y   = utils.polar2cartesian(zj[0], zj[1])
            lmnew[inds]  = rotmat2d(eta[2]) @ np.array([zj_x, zj_y]).reshape(2,) - sensor_offset_world

            Gx[inds,:2]  = I2
            Gx[inds, 2]  = zj[0] * np.array([ -np.sin(zj[1] + eta[2]), np.cos(zj[1] + eta[2]) ]).reshape(2,) + sensor_offset_world_der
            Gz           = rotmat2d(zj[1] + eta[2]) @ np.diag((1, zj[0]))
            # Gz * R * Gz^T, transform measurement covariance from polar to cartesian coordinates
            Rall[inds, inds] = Gz @ self.R @ Gz.T

        assert len(lmnew) % 2 == 0, "SLAM.add_landmark: lmnew not even length"

        etaadded        = np.append(eta, lmnew) # [eta; G() ] : append new landmarks to state vector

        low_r           = Gx @ P[:3,:3] @ Gx.T + Rall # + Gz @ self.R @ Gz.T
        P_added         = la.block_diag(P, low_r)     # block diagonal of P_new
        P_added[:n, n:] = P[:, :3] @ Gx.T      # top right corner of P_new
        P_added[n:, :n] = P_added[:n, n:].T   # transpose of above

        assert (
            etaadded.shape * 2 == P_added.shape
        ), "EKFSLAM.add_landmarks: calculated eta and P has wrong shape"
        assert np.allclose(
            P_added, P_added.T
        ), "EKFSLAM.add_landmarks: P_added not symmetric"
        assert np.all(
            np.linalg.eigvals(P_added) >= 0
        ), "EKFSLAM.add_landmarks: P_added not PSD"

        return etaadded, P_added

    def associate(
        self, z: np.ndarray, zpred: np.ndarray, H: np.ndarray, S: np.ndarray,
    ):  # -> Tuple[*((np.ndarray,) * 5)]:
        """Associate landmarks and measurements, and extract correct matrices for these.
        Parameters
         z : np.ndarray,            The measurements all in one vector
         zpred : np.ndarray         Predicted measurements in one vector
         H : np.ndarray             The measurement Jacobian matrix related to zpred
         S : np.ndarray             The innovation covariance related to zpred
        Returns
         Tuple[*((np.ndarray,) * 5)]  The extracted measurements, the corresponding zpred, H, S and the associations.
        Note
         See the associations are calculated  using JCBB. See this function for documentation
         of the returned association and the association procedure.
        """
        if self.do_asso:
            
            print(f"Associating...\n z.shape={z.shape}, zpred.shape={zpred.shape}, H.shape={H.shape}, S.shape={S.shape}")
            
            # Associate
            a = JCBB(z, zpred, S, self.alphas[0], self.alphas[1])
            
            print(f"Done associating: \n a= {a}")

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
            # should one do something here
            pass

    def update(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Update eta and P with z, associating landmarks and adding new ones.
        Parameters
         eta : np.ndarray
         P : np.ndarray
         z : np.ndarray, shape=(#detections, 2)
        Returns
         Tuple[np.ndarray, np.ndarray, float, np.ndarray]
            [description]
        """
        numLmk = (eta.size - 3) // 2
        assert (len(eta) - 3) % 2 == 0, "EKFSLAM.update: landmark lenght not even"
        
        print(f"numLmk={numLmk}")

        if numLmk > 0:
            # Prediction and innovation covariance
            zpred   = self.h(eta)
            H       = self.H(eta)

            #R = np.kron(np.eye(numLmk), self.R)
            R = np.diag( np.repeat([self.R[0,0], self.R[1,1]], numLmk) )
            S = H @ P @ H.T + R

            assert ( S.shape == zpred.shape * 2), f"EKFSLAM.update: wrong shape on either S or zpred"
            
            z = z.ravel()  # 2D -> flat

            # Perform data association
            za, zpred, Ha, Sa, a = self.associate(z, zpred, H, S)
            print(f"numLmk={numLmk}  S: {S.shape}, Sa: {Sa.shape}, R: {R.shape}")

            # No association could be made, so skip update
            if za.shape[0] == 0:
                etaupd  = eta
                Pupd    = P
                NIS     = 1 # TODO: beware this one when analysing consistency.

            else:
                # Create the associated innovation
                v = za.ravel() - zpred  # za: 2D -> flat
                v[1::2] = utils.wrapToPi(v[1::2])

                # Kalman mean update
                S_cho_factors = la.cho_factor(Sa)
                Sa_inv        = la.cho_solve(S_cho_factors, np.eye(Sa.shape[0]))

                W       = P @ Ha.T @ Sa_inv
                etaupd  = eta + W @ v # Kalman update
                # Kalman cov update: use Joseph form for stability
                jo = -W @ Ha
                jo[np.diag_indices(jo.shape[0])] += 1  # same as adding Identity mat

                Pupd = jo @ P @ jo.T
                Pupd += W @ R[:Sa.shape[0], :Sa.shape[0]] @ W.T  # TODO, Kalman update. This is the main workload on VP after speedups
                #Pupd = P - W @ Ha @ P

                #EKF:  P_upd = (I - W @ H) @ P @ (I - W @ H).T + W @ self.sensor_model.R(x) @ W.T

                # calculate NIS, can use S_cho_factors
                NIS  = self.NIS(Sa_inv, v)

                # When tested, remove for speed
                assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd not symmetric"
                assert np.all(     np.linalg.eigvals(Pupd) > 0 ), "EKFSLAM.update: Pupd not positive definite"

        else:  # All measurements are new landmarks,
            a = np.full(z.shape[0], -1)
            z = z.flatten()
            NIS     = 1 # TODO: beware this one, you can change the value to for instance 1
            etaupd  = eta
            Pupd    = P

        # Create new landmarks if any is available
        if self.do_asso:
            is_new_lmk = a == -1
            if np.any(is_new_lmk):
                z_new_inds = np.empty_like(z, dtype=bool)
                z_new_inds[::2]  = is_new_lmk
                z_new_inds[1::2] = is_new_lmk
                z_new = z[z_new_inds]
                # TODO, add new landmarks:
                etaupd, Pupd = self.add_landmarks(etaupd, Pupd, z_new) #(eta, P, z_new)

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

    @classmethod
    def NIS(cls, Sa_inv, v):
        NIS = v @ Sa_inv @ v.T
        assert NIS >= 0, f"EFKSLAM.NIS: Negative NIS: {round(NIS,3)}"
        return NIS
