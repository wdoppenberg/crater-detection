import datetime as dt
import os
import uuid

import h5py
import numpy as np
from astropy.coordinates import spherical_to_cartesian
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm as tq

import src.common.constants as const
from .conics import ConicProjector, crater_representation, ellipse_axes
from .surrender import SurRenderer
from ..matching.database import load_craters, extract_robbins_dataset


class DataGenerator(SurRenderer, ConicProjector):
    def __init__(self,
                 r_craters_catalogue,
                 C_craters_catalogue,
                 axis_threshold=const.AXIS_THRESHOLD,
                 filled=False,
                 instancing=True,
                 mask_thickness=1,
                 position=None,
                 resolution=const.CAMERA_RESOLUTION,
                 fov=const.CAMERA_FOV,
                 orbiting_body_radius=const.RMOON
                 ):
        super(DataGenerator, self).__init__(position=position, resolution=resolution, fov=fov,
                                            orbiting_body_radius=orbiting_body_radius)

        self.axis_threshold = axis_threshold
        self.mask_thickness = mask_thickness
        self.instancing = instancing
        self.filled = filled
        self.C_craters_catalogue = C_craters_catalogue
        self.r_craters_catalogue = r_craters_catalogue

    @classmethod
    def from_robbins_dataset(cls,
                             file_path="data/lunar_crater_database_robbins_2018.csv",
                             diamlims=const.DIAMLIMS,
                             ellipse_limit=const.MAX_ELLIPTICITY,
                             arc_lims=const.ARC_LIMS,
                             axis_threshold=const.AXIS_THRESHOLD,
                             filled=False,
                             instancing=True,
                             mask_thickness=1,
                             position=None,
                             resolution=const.CAMERA_RESOLUTION,
                             fov=const.CAMERA_FOV,
                             orbiting_body_radius=const.RMOON
                             ):
        lat_cat, long_cat, major_cat, minor_cat, psi_cat, crater_id = extract_robbins_dataset(
            load_craters(file_path, diamlims=diamlims, ellipse_limit=ellipse_limit, arc_lims=arc_lims)
        )
        r_craters_catalogue = np.array(np.array(spherical_to_cartesian(const.RMOON, lat_cat, long_cat))).T[..., None]
        C_craters_catalogue = crater_representation(major_cat, minor_cat, psi_cat)

        return cls(r_craters_catalogue=r_craters_catalogue,
                   C_craters_catalogue=C_craters_catalogue,
                   axis_threshold=axis_threshold,
                   filled=filled,
                   instancing=instancing,
                   mask_thickness=mask_thickness,
                   resolution=resolution,
                   fov=fov,
                   orbiting_body_radius=orbiting_body_radius,
                   position=position
                   )

    def __visible(self):
        return (cdist(self.r_craters_catalogue.squeeze(), self.position.T) <=
                np.sqrt(2 * self.height * self._orbiting_body_radius + self.height ** 2)).ravel()

    def generate_mask(self, *args, **kwargs):
        r_craters = self.r_craters_catalogue[self.__visible()]
        C_craters = self.C_craters_catalogue[self.__visible()]

        r_craters_img = self.project_crater_centers(r_craters)
        in_image = np.logical_and.reduce(np.logical_and(r_craters_img > -50, r_craters_img < self.resolution[0] + 50),
                                         axis=1)

        r_craters = r_craters[in_image]
        C_craters = C_craters[in_image]

        A_craters = self.project_crater_conics(C_craters, r_craters)

        a_proj, b_proj = ellipse_axes(A_craters)
        axis_filter = np.logical_and(a_proj >= self.axis_threshold[0], b_proj >= self.axis_threshold[0])
        axis_filter = np.logical_and(axis_filter,
                                     np.logical_and(a_proj <= self.axis_threshold[1], b_proj <= self.axis_threshold[1]))
        A_craters = A_craters[axis_filter]

        return super(DataGenerator, self).generate_mask(A_craters=A_craters,
                                                        filled=self.filled,
                                                        instancing=self.instancing,
                                                        thickness=self.mask_thickness)

    def image_mask_pair(self):
        return self.generate_image(), self.generate_mask()


def generate(size,
             axis_threshold=const.AXIS_THRESHOLD,
             resolution=const.CAMERA_RESOLUTION,
             fov=const.CAMERA_FOV,
             min_sol_incidence=0,
             max_sol_incidence=85,
             filled=False,
             ellipse_limit=const.MAX_ELLIPTICITY,
             arc_lims=0.85,
             diamlims=const.DIAMLIMS,
             instancing=False,
             randomized_orientation=True,
             mask_thickness=1
             ):
    images_dataset = np.empty((size, 1, *resolution), np.float32)
    if instancing:
        masks_dataset = np.empty((size, 1, *resolution), np.int16)
    else:
        masks_dataset = np.empty((size, 1, *resolution), np.bool_)
    position_dataset = np.empty((size, 3, 1), np.float64)
    attitude_dataset = np.empty((size, 3, 3), np.float64)
    sol_incidence_dataset = np.empty((size, 1), np.float16)
    date_dataset = np.empty((size, 3), int)

    generator = DataGenerator.from_robbins_dataset(
        diamlims=diamlims,
        ellipse_limit=ellipse_limit,
        arc_lims=arc_lims,
        axis_threshold=axis_threshold,
        fov=fov,
        resolution=resolution,
        filled=filled,
        mask_thickness=mask_thickness,
        instancing=instancing
    )

    for i in tq(range(size), desc="Creating dataset"):
        date = dt.date(2021, np.random.randint(1, 12), 1)
        generator.scene_time = date
        date_dataset[i] = np.array((date.year, date.month, date.day))

        while not (min_sol_incidence <= generator.solar_incidence_angle <= max_sol_incidence):
            generator.set_random_position()  # Generate random position

        position_dataset[i] = generator.position
        sol_incidence_dataset[i] = generator.solar_incidence_angle

        if randomized_orientation:
            # Rotations are incremental (order matters)
            generator.rotate('roll', np.random.randint(-180, 180))
            generator.rotate('pitch', np.random.randint(-10, 10))
            generator.rotate('yaw', np.random.randint(-10, 10))

        attitude_dataset[i] = generator.attitude

        image, mask = generator.image_mask_pair()

        masks_dataset[i] = mask[None, None, ...]
        images_dataset[i] = image[None, None, ...]

    return images_dataset, masks_dataset, position_dataset, attitude_dataset, date_dataset, sol_incidence_dataset


def demo_settings(n_demo=20,
                  axis_threshold=const.AXIS_THRESHOLD,
                  resolution=const.CAMERA_RESOLUTION,
                  fov=const.CAMERA_FOV,
                  min_sol_incidence=0,
                  max_sol_incidence=85,
                  filled=False,
                  ellipse_limit=const.MAX_ELLIPTICITY,
                  arc_lims=const.ARC_LIMS,
                  diamlims=const.DIAMLIMS,
                  instancing=False,
                  randomized_orientation=True,
                  mask_thickness=1):
    images, mask, _, _, _, _ = generate(n_demo,
                                        axis_threshold,
                                        resolution,
                                        fov,
                                        min_sol_incidence,
                                        max_sol_incidence,
                                        filled,
                                        ellipse_limit,
                                        arc_lims,
                                        diamlims,
                                        instancing,
                                        randomized_orientation,
                                        mask_thickness)

    fig, axes = plt.subplots(n_demo, 2, figsize=(10, 5 * n_demo))
    for i in range(n_demo):
        axes[i, 0].imshow(images[i, 0], cmap='Greys_r')
        axes[i, 1].imshow((mask[i, 0] > 0).astype(float), cmap='gray')

    plt.show()


def make_dataset(n_training,
                 n_validation,
                 n_testing,
                 output_path=None,
                 identifier=None,
                 **generation_kwargs):
    if output_path is None:
        if identifier is None:
            identifier = str(uuid.uuid4())

        output_path = f"data/dataset_{identifier}.h5"

    if os.path.exists(output_path):
        raise ValueError(f"Dataset named `{os.path.basename(output_path)}` already exists!")

    with h5py.File(output_path, 'w') as hf:
        g_header = hf.create_group("header")

        for k, v in generation_kwargs.items():
            g_header.create_dataset(k, data=v)

        for group_name, dset_size in zip(
                ("training", "validation", "test"),
                (n_training, n_validation, n_testing)
        ):
            print(f"Creating dataset '{group_name}' @ {dset_size} images")
            group = hf.create_group(group_name)

            (images, masks, position, attitude, date, sol_incidence) = generate(dset_size, **generation_kwargs)
            for ds, name in zip(
                        (images, masks, position, attitude, date, sol_incidence),
                        ("images", "masks", "position", "attitude", "date", "sol_incidence")
                    ):

                group.create_dataset(name, data=ds)
