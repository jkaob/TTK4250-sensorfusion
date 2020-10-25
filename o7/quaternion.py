import numpy as np
import utils


def quaternion_product(ql: np.ndarray, qr: np.ndarray) -> np.ndarray:
    """Perform quaternion product according to either (10.21) or (10.34).

    Args:
        ql (np.ndarray): Left quaternion of the product of either shape (3,) (pure quaternion) or (4,)
        qr (np.ndarray): Right quaternion of the product of either shape (3,) (pure quaternion) or (4,)

    Raises:
        RuntimeError: Left or right quaternion are of the wrong shape
        AssertionError: Resulting quaternion is of wrong shape

    Returns:
        np.ndarray: Quaternion product of ql and qr of shape (4,)s
    """
    if ql.shape == (4,):
        eta_left = ql[0]
        epsilon_left = ql[1:].reshape((3, 1))
    elif ql.shape == (3,):
        eta_left = 0
        epsilon_left = ql.reshape((3, 1))
    else:
        raise RuntimeError(
            f"utils.quaternion_product: Quaternion multiplication error, left quaternion shape incorrect: {ql.shape}"
        )

    if qr.shape == (4,):
        eta_right = qr[0]
        epsilon_right = qr[1:].reshape((3, 1))
        q_right = qr.copy()
    elif qr.shape == (3,):
        eta_right = 0
        epsilon_right = qr.reshape((3, 1))
        q_right = np.concatenate(([0], qr))
    else:
        raise RuntimeError(
            f"utils.quaternion_product: Quaternion multiplication error, right quaternion wrong shape: {qr.shape}"
        )

    quaternion      = np.zeros((4,))
    eta_new         = eta_left*eta_right - epsilon_left.T @ epsilon_right
    epsilon_new     = eta_right*epsilon_left + eta_left * epsilon_right + utils.cross_product_matrix(epsilon_left) @ epsilon_right
    quaternion      = np.array([eta_new, *epsilon_new])

    # Ensure result is of correct shape
    quaternion = quaternion.ravel()
    assert quaternion.shape == (4,), f"utils.quaternion_product: Quaternion multiplication error, result quaternion wrong shape: {quaternion.shape}"

    return quaternion


def quaternion_to_rotation_matrix(
    quaternion: np.ndarray, debug: bool = True
) -> np.ndarray:
    """Convert a quaternion to a rotation matrix

    Args:
        quaternion (np.ndarray): Quaternion of either shape (3,) (pure quaternion) or (4,)
        debug (bool, optional): Debug flag, could speed up by setting to False. Defaults to True.

    Raises:
        RuntimeError: Quaternion is of the wrong shape
        AssertionError: Debug assert fails, rotation matrix is not element of SO(3)

    Returns:
        np.ndarray: Rotation matrix of shape (3, 3)
    """
    if quaternion.shape == (4,):
        eta = quaternion[0]
        epsilon = quaternion[1:]
    elif quaternion.shape == (3,):
        eta = 0
        epsilon = quaternion.copy()
    else:
        raise RuntimeError(
            f"quaternion.quaternion_to_rotation_matrix: Quaternion to multiplication error, quaternion shape incorrect: {quaternion.shape}"
        )

    I   =  np.eye(3,3)
    S_e = utils.cross_product_matrix(epsilon)
    R   =  I + 2 * eta * S_e + 2 * S_e @ S_e


    if debug:
        assert np.allclose(
            np.linalg.det(R), 1
        ), f"quaternion.quaternion_to_rotation_matrix: Determinant of rotation matrix not close to 1"
        assert np.allclose(
            R.T, np.linalg.inv(R)
        ), f"quaternion.quaternion_to_rotation_matrix: Transpose of rotation matrix not close to inverse"

    return R


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion into euler angles

    Args:
        quaternion (np.ndarray): Quaternion of shape (4,)

    Returns:
        np.ndarray: Euler angles of shape (3,)
    """

    assert quaternion.shape == ( 4,
    ), f"quaternion.quaternion_to_euler: Quaternion shape incorrect {quaternion.shape}"

    quaternion_squared = quaternion ** 2

    R = quaternion_to_rotation_matrix(quaternion)
    R_11 = R[0,0]
    R_21 = R[1,0]
    R_31 = R[2,0]
    R_32 = R[2,1]
    R_33 = R[2,2]


    phi   =  np.arctan2(R_32,R_33)  #: Convert from quaternion to euler angles
    theta = -np.arcsin(R_31)  #: Convert from quaternion to euler angles
    psi   =  np.arctan2(R_21,R_11)  #: Convert from quaternion to euler angles

    #: Convert directly from quaternions to euler angles, eq 10.38
    eta, eps1, eps2, eps3  = quaternion.ravel()
    phi   = np.arctan2(2*(eps3*eps2 + eta*eps1), eta**2 - eps1**2 - eps2**2 + eps3**2)
    theta = np.arcsin(2*(eta*eps2 - eps1*eps3))
    psi   = np.arctan2(2*(eps1*eps2 + eta*eps3), eta**2 + eps1**2 - eps2**2 - eps3**2)

    euler_angles = np.array([phi, theta, psi])
    assert euler_angles.shape == (
        3,
    ), f"quaternion.quaternion_to_euler: Euler angles shape incorrect: {euler_angles.shape}"

    return euler_angles


def euler_to_quaternion(euler_angles: np.ndarray) -> np.ndarray:
    """Convert euler angles into quaternion

    Args:
        euler_angles (np.ndarray): Euler angles of shape (3,)

    Returns:
        np.ndarray: Quaternion of shape (4,)
    """

    assert euler_angles.shape == (
        3,
    ), f"quaternion.euler_to_quaternion: euler_angles shape wrong {euler_angles.shape}"

    half_angles = 0.5 * euler_angles
    c_phi2, c_theta2, c_psi2 = np.cos(half_angles)
    s_phi2, s_theta2, s_psi2 = np.sin(half_angles)

    quaternion = np.array(
        [
            c_phi2 * c_theta2 * c_psi2 + s_phi2 * s_theta2 * s_psi2,
            s_phi2 * c_theta2 * c_psi2 - c_phi2 * s_theta2 * s_psi2,
            c_phi2 * s_theta2 * c_psi2 + s_phi2 * c_theta2 * s_psi2,
            c_phi2 * c_theta2 * s_psi2 - s_phi2 * s_theta2 * c_psi2,
        ]
    )

    assert quaternion.shape == (
        4,
    ), f"quaternion.euler_to_quaternion: Quaternion shape incorrect {quaternion.shape}"

    return quaternion

def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion into conjugate"""

    assert quaternion.shape == (
        4,
    ), f"quaternion.euler_to_quaternion: Quaternion shape incorrect {quaternion.shape}"

    conjugate = np.array([quaternion[0], -quaternion[1:3]]).reshape(4,)

    assert conjugate.shape == (
        4,
    ), f"quaternion.euler_to_quaternion: Conjugate shape incorrect {q_conjugate.shape}"

    return conjugate

def quaternion_normalize(quaternion: np.ndarray) -> np.ndarray:
    """Normalize Quaternion"""
    assert quaternion.shape == (
        4,
    ), f"quaternion.euler_to_quaternion: Quaternion shape incorrect {quaternion.shape}"

    return quaternion/np.linalg.norm(quaternion)
