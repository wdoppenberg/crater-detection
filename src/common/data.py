import datetime as dt
import os
import uuid

import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm as tq

import src.common.constants as const
from src.common.conics import MaskGenerator
from src.common.surrender import SurRenderer


class DataGenerator(MaskGenerator, SurRenderer):
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
