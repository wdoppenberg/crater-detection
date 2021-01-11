import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from craterdetection.detection import DeepMoon
from craterdetection.detection.post_processing import crater_detection, draw_detections
from craterdetection.matching import (CoplanarInvariants,
                                      crater_representation)


def main():

    h5_craters = h5py.File("data/dev_craters.hdf5", "r")
    h5_images = h5py.File("data/dev_images.hdf5", "r")

    image_i = 30
    crater_test = h5_craters[f'img_{image_i:05}/block0_values'][...]
    df_target = pd.DataFrame(crater_test, columns=['diam', 'lat', 'long', 'x_pix', 'y_pix', 'diam_pix'])
    img_test = h5_images['input_images'][image_i] / 255
    df_target.eval('kmperpix = diam / diam_pix', inplace=True)

    target_mask = np.zeros_like(img_test)
    for i, r in df_target.iterrows():
        center_coordinates = (round(r['x_pix']), round(r['y_pix']))
        axes_length = (round(r['diam_pix'] / 2), round(r['diam_pix'] / 2))
        angle = 0
        target_mask = cv2.ellipse(target_mask, center_coordinates, axes_length,
                                  angle, 0, 360, (255, 255, 255), 1)

    batch = img_test.reshape(1, 1, 256, 256)

    try:
        from craterdetection.detection.VPU import OpenVINOHandler
        exp = OpenVINOHandler('DeepMoon', device='MYRIAD', root='craterdetection/detection/VPU/IR/')
        out = exp.infer(batch)
    except (ImportError, NameError):
        print("OpenVINOHandler could not be instantiated. Using PyTorch backend")
        net = DeepMoon()
        checkpoint_path = 'blobs/DeepMoon.pth'
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)
        net.eval()

        with torch.no_grad():
            out = net(torch.Tensor(batch))
            out = out.numpy()

    df_detections = crater_detection(out[0, 0], threshold_lower=80, ellipticity_threshold=0.01)

    img_out = [
        batch[0, 0],
        target_mask,
        out[0, 0],
        255 * batch[0, 0] + draw_detections(df_detections)
    ]

    fig, axes = plt.subplots(1, len(img_out), figsize=(20, 15))

    for ax, img in zip(axes, img_out):
        ax.imshow(img, cmap='Greys_r')
        ax.axis('off')

    for index, r in df_detections.iterrows():
        axes[-1].text(r['x_pix'], r['y_pix'], str(index), color='lightgreen')

    print(f"{len(df_detections)} craters detected!")

    # MATCHING

    x, y, a, b, psi, e, r, d = df_detections.to_numpy().T
    psi = np.deg2rad(psi)

    detected_craters = crater_representation(x, y, a, b, psi)
    ci = CoplanarInvariants(detected_craters)

    xy_target = df_target[['x_pix', 'y_pix']].to_numpy()
    df_matched = df_target.iloc[np.argmin(cdist(df_detections[['x_pix', 'y_pix']].to_numpy(), xy_target), axis=1)]
    df_matched.columns = [s + "_t" for s in df_matched.columns]
    df_matched = pd.concat([df_detections, df_matched.reset_index(drop=True)], axis=1)

    pole_crater = df_matched.iloc[0]
    lat_t, long_t = map(np.radians, (df_matched['lat_t'].to_numpy(), df_matched['long_t'].to_numpy()))
    lat_pole, long_pole = map(np.radians, (pole_crater['lat_t'], pole_crater['long_t']))

    dlat = lat_t - lat_pole
    dlong = long_t - long_pole
    R = 1737.1

    x_t = 2 * R * np.arcsin(np.cos(lat_pole) * np.sin(dlong / 2))
    y_t = R * dlat

    a_t, b_t = [df_matched['diam_t'].to_numpy() / 2] * 2
    phi_t = 0.

    matched_craters = crater_representation(x_t, y_t, a_t, b_t, phi_t)
    ci_t = CoplanarInvariants(matched_craters)

    # it = np.random.choice(range(len(ci_t)), 100)
    # for i in it:
    #     print(i)
    #     print(ci[i])
    #     print(ci_t[i])

    j = 1092
    print(ci[j])
    print(ci_t[j])


if __name__ == "__main__":
    main()
