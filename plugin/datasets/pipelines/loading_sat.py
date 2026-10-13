import os

import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadAID4ADSatelliteImage(object):
    """Load the ego-aligned AID4AD satellite tile of a sample.

    Tiles are looked up by sample token in {data_root}/{city}/{counter}_{token}.png
    (400x200 RGB at 0.15 m/px for the 60 m x 30 m ROI). A missing tile is
    replaced by a zero image.

    Args:
        data_root (str): root directory of the AID4AD frames.
        canvas_size (tuple, optional): resize to (W, H); None keeps the tile size.
        normalize (bool): ImageNet mean/std normalization.
    """

    CITIES = [
        'boston-seaport', 'singapore-onenorth',
        'singapore-hollandvillage', 'singapore-queenstown',
    ]

    def __init__(self, data_root, canvas_size=None, normalize=True):
        self.data_root = data_root
        self.canvas_size = canvas_size
        self.normalize = normalize

        self.token_lookup = {}
        for city in self.CITIES:
            city_dir = os.path.join(data_root, city)
            if not os.path.isdir(city_dir):
                continue
            for fname in os.listdir(city_dir):
                if fname.endswith('.png'):
                    token = fname.split('_', 1)[1].replace('.png', '')
                    self.token_lookup[token] = os.path.join(city, fname)
        print(f'[LoadAID4ADSatelliteImage] indexed {len(self.token_lookup)} tiles '
              f'from {data_root}')

    def __call__(self, results):
        rel_path = self.token_lookup.get(results['token'])
        if rel_path is not None:
            img = cv2.imread(os.path.join(self.data_root, rel_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.flipud(img).copy()  # rows bottom-up, as in the BEV grid
            if self.canvas_size is not None:
                W, H = self.canvas_size
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
        else:
            W, H = self.canvas_size if self.canvas_size is not None else (400, 200)
            img = np.zeros((H, W, 3), dtype=np.float32)

        if self.normalize:
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            img = (img - mean) / std

        results['sat_img'] = img.transpose(2, 0, 1)  # (3, H, W)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(data_root={self.data_root}, '
                f'canvas_size={self.canvas_size})')
