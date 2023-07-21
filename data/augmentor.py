import numpy as np


class FlowAugmentor:
    def __init__(self, size):
        # spatial augmentation params
        self.crop_size = (size, size) if isinstance(size, int) else size

    def spatial_transform(self, sample_dict):
        # randomly crop
        ht, wd = list(sample_dict.values())[0].shape[:2]
        y0 = np.random.randint(0, ht - self.crop_size[0])
        x0 = np.random.randint(0, wd - self.crop_size[1])

        def crop_fn(x):
            return x[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1], :]

        for k, v in sample_dict.items():
            sample_dict[k] = crop_fn(v)

        return sample_dict

    def __call__(self, sample_dict):
        # random crop
        sample_dict = self.spatial_transform(sample_dict)
        return sample_dict
