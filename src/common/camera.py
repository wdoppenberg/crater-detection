from collections.abc import Iterable
from typing import Union, Tuple

import numpy as np
import numpy.linalg as LA
from astropy.coordinates import spherical_to_cartesian
from scipy.spatial.transform import Rotation

import src.common.constants as const
from src.common.coordinates import nadir_attitude


def camera_matrix(fov=const.CAMERA_FOV, resolution=const.CAMERA_RESOLUTION, alpha=0):
    """Returns camera matrix [1] from Field-of-View, skew, and offset.

    Parameters
    ----------
    fov : float, Iterable
        Field-of-View angle (degrees), if type is Iterable it will be interpreted as (fov_x, fov_y)
    resolution : float, Iterable
        X- and Y-resolution of the image in pixels
    alpha : float
        Camera skew angle.

    Returns
    -------
    np.ndarray
        3x3 camera matrix

    References
    ----------
    .. [1] https://www.cs.ucf.edu/~mtappen/cap5415/lecs/lec19.pdf
    """

    if isinstance(resolution, Iterable):
        x_0, y_0 = map(lambda x: x / 2, resolution)
    else:
        x_0 = resolution / 2
        y_0 = resolution / 2

    if isinstance(fov, Iterable):
        f_x, f_y = map(lambda x, fov_: x / np.tan(np.radians(fov_ / 2)), (x_0, y_0), fov)
    else:
        f_x, f_y = map(lambda x: x / np.tan(np.radians(fov / 2)), (x_0, y_0))

    return np.array([[f_x, alpha, x_0],
                     [0, f_y, y_0],
                     [0, 0, 1]])


def projection_matrix(K, T_CM, r_M):
    """Return Projection matrix [1] according to:

    .. math:: ^x\mathbf{P}_C = \mathbf{K} [ ^x\mathbf{T^C_M}_C & -r_C]

    Parameters
    ----------
    K : np.ndarray
        3x3 camera matrix
    T_CM : np.ndarray
        3x3 attitude matrix of camera in selenographic frame.
    r_M : np.ndarray
        3x1 camera position in world reference frame

    Returns
    -------
    np.ndarray
        3x4 projection matrix

    References
    ----------
    .. [1] https://www.cs.ucf.edu/~mtappen/cap5415/lecs/lec19.pdf

    See Also
    --------
    camera_matrix

    """
    return K @ LA.inv(T_CM) @ np.concatenate((np.identity(3), -r_M), axis=1)


class Camera:
    """
    Camera data class with associated state attributes & functions.
    """

    def __init__(self,
                 position=None,
                 attitude=None,
                 fov=const.CAMERA_FOV,
                 resolution=const.CAMERA_RESOLUTION,
                 orbiting_body_radius=const.RMOON
                 ):

        self._orbiting_body_radius = orbiting_body_radius
        self.__position = None
        self.__attitude = None
        self.__fov = None
        self.__resolution = None

        self.position = position
        self.fov = fov
        self.resolution = resolution
        self.attitude = attitude

    @classmethod
    def from_coordinates(cls,
                         lat,
                         long,
                         altitude,
                         fov=const.CAMERA_FOV,
                         resolution=const.CAMERA_RESOLUTION,
                         attitude=None,
                         Rbody=const.RMOON,
                         convert_to_radians=False
                         ):
        if convert_to_radians:
            lat, long = map(np.radians, (lat, long))

        position = np.array(spherical_to_cartesian(Rbody + altitude, lat, long))
        return cls(position=position, attitude=attitude, fov=fov, resolution=resolution)

    @property
    def position(self) -> np.ndarray:
        return self.__position

    @position.setter
    def position(self, position: np.ndarray):
        """
        Sets instance's position in Cartesian space.

        If set to None, a random position above the moon will be generated between 150 and 400 km altitude.

        Parameters
        ----------
        position : np.ndarray
            3x1 position vector of camera.
        """
        if position is None:
            self.set_random_position()

        else:
            # Ensure 3x1 vector
            if len(position.shape) == 1:
                position = position[:, None]
            elif len(position.shape) > 2:
                raise ValueError("Position vector must be 1 or 2-dimensional (3x1)!")

            if LA.norm(position) < self._orbiting_body_radius:
                raise ValueError(
                    f"New position vector is inside the Moon! (Distance to center = {LA.norm(position):.2f} km, "
                    f"R_moon = {self._orbiting_body_radius})"
                )

            if not position.dtype == np.float64:
                position = position.astype(np.float64)

            self.__position = position

    def set_coordinates(self,
                        lat,
                        long,
                        height=None,
                        point_nadir=False,
                        convert_to_radians=False
                        ):
        if height is None:
            height = self.height

        if convert_to_radians:
            lat, long = map(np.radians, (lat, long))

        self.position = np.array(spherical_to_cartesian(self._orbiting_body_radius + height, lat, long))

        if point_nadir:
            self.point_nadir()

    def set_random_position(self, min_height=150, max_height=400):
        lat = np.random.randint(-90, 90)
        long = np.random.randint(-180, 180)
        height = np.random.randint(min_height, max_height)
        self.set_coordinates(lat, long, height, point_nadir=True, convert_to_radians=True)

    # Alias
    r: np.ndarray = position

    @property
    def attitude(self) -> np.ndarray:
        return self.__attitude

    @attitude.setter
    def attitude(self, attitude: Union[np.ndarray, Rotation]):
        """
        Sets instance's attitude

        Parameters
        ----------
        attitude : np.ndarray, Rotation
            Orientation / attitude matrix (3x3) or scipy.spatial.transform.Rotation

        See Also
        --------
        scipy.spatial.transform.Rotation
        """
        if attitude is None:
            self.point_nadir()
        else:
            if isinstance(attitude, Rotation):
                attitude = attitude.as_matrix()

            if not np.isclose(abs(LA.det(attitude)), 1):
                raise ValueError(f"Invalid rotation matrix! Determinant should be +-1, is {LA.det(attitude)}.")

            if LA.matrix_rank(attitude) != 3:
                raise ValueError("Invalid camera attitude matrix!:\n", attitude)

            self.__attitude = attitude

    # Alias
    T: np.ndarray = attitude

    @property
    def fov(self) -> Tuple:
        return self.__fov

    @fov.setter
    def fov(self, fov):
        """
        Set instance's Field-of-View in radians.

        Parameters
        ----------
        fov: int, float, Iterable
            Field-of-View angle (radians), if type is Iterable it will be interpreted as (fov_x, fov_y)
        """
        if not isinstance(fov, Iterable):
            self.__fov = (fov, fov)
        else:
            self.__fov = tuple(fov)

    @property
    def resolution(self) -> Tuple:
        return self.__resolution

    @resolution.setter
    def resolution(self, resolution):
        """
        Set instance's resolution in pixels.

        Parameters
        ----------
        resolution : int, Iterable
            Image resolution, if type is Iterable it will be interpreted as (res_x, res_y)
        """
        if not isinstance(resolution, Iterable):
            self.__resolution = (resolution, resolution)
        else:
            self.__resolution = tuple(resolution)

    @property
    def camera_matrix(self) -> np.ndarray:
        return camera_matrix(fov=self.fov, resolution=self.resolution)

    # Alias
    K: np.ndarray = camera_matrix

    @property
    def projection_matrix(self) -> np.ndarray:
        return projection_matrix(K=self.K, T_CM=self.attitude, r_M=self.position)

    # Alias
    P: np.ndarray = projection_matrix

    @property
    def height(self):
        return LA.norm(self.position) - self._orbiting_body_radius

    @height.setter
    def height(self, height):
        """
        Adjusts radial height without changing angular position.

        Parameters
        ----------
        height: int, float
            Height to set to in km.
        """
        if height <= 0:
            raise ValueError(f"Height cannot be below 0! (height = {height})")

        self.position = (self.position / LA.norm(self.position)) * (self._orbiting_body_radius + height)

    def rotate(self, axis: str, angle: float, degrees: bool = True, reset_first: bool = False):
        if axis not in ('x', 'y', 'z', 'pitch', 'yaw', 'roll'):
            raise ValueError("axis must be 'x', 'y', 'z', or 'pitch', 'yaw', 'roll'")

        if axis == 'roll':
            axis = 'z'
        elif axis == 'pitch':
            axis = 'x'
        elif axis == 'yaw':
            axis = 'y'

        if reset_first:
            self.point_nadir()

        self.attitude = (
                Rotation.from_matrix(self.attitude) * Rotation.from_euler(axis, angle, degrees=degrees)
            ).as_matrix()

    def point_nadir(self):
        self.attitude = np.concatenate(nadir_attitude(self.position), axis=-1)
