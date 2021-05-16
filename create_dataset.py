import argparse

from src.common.data import make_dataset
import src.common.constants as const


def get_parser():
    parser = argparse.ArgumentParser(description='Create dataset containing input images and relevant ellipses using '
                                                 'SurRender.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_train', type=int, default=20000,
                        help='Number of training images')
    parser.add_argument('--n_val', type=int, default=2000,
                        help='Number of validation images')
    parser.add_argument('--n_test', type=int, default=1000,
                        help='Number of testing images')
    parser.add_argument('--identifier', type=str, default=None,
                        help='Number of testing images')
    parser.add_argument('--resolution', type=tuple, default=const.CAMERA_RESOLUTION,
                        help='Camera resolution')
    parser.add_argument('--fov', type=tuple, default=const.CAMERA_FOV,
                        help='Camera FoV')
    parser.add_argument('--min_sol_incidence', type=float, default=const.MIN_SOL_INCIDENCE,
                        help='Minimum solar incidence angle')
    parser.add_argument('--max_sol_incidence', type=float, default=const.MAX_SOL_INCIDENCE,
                        help='Maximum solar incidence angle')
    parser.add_argument('--ellipse_limit', type=float, default=const.MAX_ELLIPTICITY,
                        help='Maximum ellipticity for gt ellipse shapes (selenographic)')
    parser.add_argument('--filled', type=bool, default=const.FILLED,
                        help='Whether to fill the crater masks or not')
    parser.add_argument('--mask_thickness', type=int, default=const.MASK_THICKNESS,
                        help='How thick to make the mask rim if not filled')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    generation_kwargs = dict(
        resolution=args.resolution,
        fov=args.fov,
        min_sol_incidence=args.min_sol_incidence,
        max_sol_incidence=args.max_sol_incidence,
        ellipse_limit=args.ellipse_limit,
        filled=args.filled,
        mask_thickness=args.mask_thickness
    )

    make_dataset(
        n_training=args.n_train,
        n_validation=args.n_val,
        n_testing=args.n_test,
        identifier=args.identifier,
        generation_kwargs=generation_kwargs
    )




