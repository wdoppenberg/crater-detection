import numpy as np
import torch
import torch.linalg as LA
from tqdm.auto import tqdm as tq

from common.data import DataGenerator
from src import CraterDetector, CraterDatabase

if __name__ == "__main__":
    db = CraterDatabase.from_file("data/lunar_crater_database_robbins_2018.csv",
                                  latlims=(0, 30),
                                  longlims=(0, 30),
                                  diamlims=(3, 40)
                                  )

    cam = DataGenerator.from_robbins_dataset(file_path="data/lunar_crater_database_robbins_2018.csv", diamlims=(4, 40),
                                             axis_threshold=(1, 250))

    cda = CraterDetector()
    cda.load_state_dict(torch.load("blobs/CraterRCNN.pth"))
    cda.to('cuda')
    cda.eval()

    n_trials = 1
    test_size = 25000

    attitude_test = np.empty((n_trials, test_size, 3, 3), np.float64)
    true_position_test = np.full((n_trials, test_size, 3, 1), np.nan, np.float64)
    pred_ransac_test = np.full((n_trials, test_size, 3, 1), np.nan, np.float64)
    pred_reprojection_test = np.full((n_trials, test_size, 3, 1), np.nan, np.float64)
    n_detections_test = np.zeros((n_trials, test_size))
    n_inliers_test = np.zeros((n_trials, test_size))
    n_verified_test = np.zeros((n_trials, test_size))
    sun_angle_test = np.zeros((n_trials, test_size))
    pitch_angle_test = np.zeros((n_trials, test_size))
    yaw_angle_test = np.zeros((n_trials, test_size))

    match_stats = np.zeros((n_trials))
    error_stats = np.zeros((n_trials))
    trial_args = []

    for trial in range(n_trials):
        matches = 0
        errors = 0
        latest_position_error = 0.
        match_trial_args = dict(
            sigma_pix=4,
            k=30,
            max_distance=0.043,
            batch_size=500,
            residual_threshold=0.011,
            max_trials=1250
        )
        trial_args.append(match_trial_args)
        bar = tq(range(test_size), postfix={"matches": 0, "errors": 0, "latest_position_error": 0.})
        for i in bar:
            n_det = 0
            while n_det < 10:
                cam.set_coordinates(np.random.uniform(5, 25), np.random.uniform(5, 25))
                cam.height = np.random.uniform(50, 200)
                cam.point_nadir()
                cam.rotate('roll', np.random.uniform(0, 360))
                pitch = np.random.uniform(-10, 10)
                cam.rotate('pitch', pitch)
                yaw = np.random.uniform(-10, 10)
                cam.rotate('yaw', yaw)
                A_craters = cam.craters_in_image()
                n_det = len(A_craters)

            image = cam.generate_image()
            image = torch.as_tensor(image[None, None, ...]).to('cuda')
            with torch.no_grad():
                pred = cda(image)[0]

            scores = pred['scores']
            A_detections = pred['ellipse_matrices'][scores > 0.75].cpu().numpy()

            attitude_test[trial, i] = cam.attitude
            true_position_test[trial, i] = cam.position
            sun_angle_test[trial, i] = cam.solar_incidence_angle
            pitch_angle_test[trial, i] = pitch
            yaw_angle_test[trial, i] = yaw
            n_detections_test[trial, i] = n_det

            if len(A_detections) < 6:
                continue
            try:
                position_regressor = db.query_position(A_detections, T=cam.T, K=cam.K,
                                                       **match_trial_args)
                if position_regressor.ransac_match():
                    pred_ransac_test[trial, i] = position_regressor.est_pos_ransac
                    n_inliers_test[trial, i] = position_regressor.num_inliers
                    # print(f"[{i:05}]\tRANSAC position error: {LA.norm(position_regressor.est_pos_ransac - cam.position)*1000:12.2f} m\t"
                    #       f"| Inliers: {position_regressor.num_inliers}", end= "\t")

                    if position_regressor.reprojection_match():
                        pred_reprojection_test[trial, i] = position_regressor.est_pos_verified
                        n_verified_test[trial, i] = position_regressor.num_verified
                        # print(f"Verified position error:  {LA.norm(position_regressor.est_pos_verified - cam.position)*1000:12.2f} m", end="")
                        latest_position_error = LA.norm(position_regressor.est_pos_verified - cam.position)
                        if latest_position_error < 20:
                            matches += 1
                        else:
                            errors += 1

            except ValueError:
                errors += 1

            bar.set_postfix(
                ordered_dict={"matches": matches, "errors": errors, "latest_position_error": latest_position_error})

        match_stats[trial] = matches
        error_stats[trial] = errors
