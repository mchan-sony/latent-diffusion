from glob import glob

import numpy as np
import PIL
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class DarkZurichBase(Dataset):
    def __init__(self, data_path, size=None, interpolation="bicubic", flip_p=0.5):
        self.files = glob(data_path)
        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image = Image.open(self.files[i])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        (
            h,
            w,
        ) = (
            img.shape[0],
            img.shape[1],
        )
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example = {"image": (image / 127.5 - 1.0).astype(np.float32)}
        return example


class DarkZurichNightTrain(DarkZurichBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_path="/data/matthew/dark_zurich/train/rgb_anon/train/night/**/*.png",
            **kwargs
        )


class DarkZurichNightValidation(DarkZurichBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(
            data_path="/data/matthew/dark_zurich/val/rgb_anon/val/night/**/*.png",
            flip_p=flip_p,
            **kwargs
        )

class DarkZurichDayTrain(DarkZurichBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_path="/data/matthew/dark_zurich/train/rgb_anon/train/day/**/*.png",
            **kwargs
        )


class DarkZurichDayValidation(DarkZurichBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(
            data_path="/data/matthew/dark_zurich/val/rgb_anon/val_ref/day/**/*.png",
            flip_p=flip_p,
            **kwargs
        )
