import math

import torch.nn
import torchvision.transforms
import torchvision.transforms.functional as T_f
from omegaconf import ListConfig
from torchvision.transforms import InterpolationMode

from ..utils import get_class


# From https://stackoverflow.com/a/16778797/10444046
def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return int(wr), int(hr)


class RandomRotationWithCrop(torch.nn.Module):

    def __init__(self, angle_deg: float, interpolation: str | InterpolationMode = InterpolationMode.BILINEAR):
        super().__init__()
        assert angle_deg > 0
        self.angle_deg = angle_deg

        if isinstance(interpolation, str):
            interpolation = InterpolationMode(interpolation)
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor):
        rand_angle_deg = torchvision.transforms.RandomRotation.get_params([-self.angle_deg, self.angle_deg])
        rotate_pad = T_f.rotate(img, rand_angle_deg, self.interpolation, expand=True)

        w, h = img.shape[-1], img.shape[-2]
        w_c, h_c = rotatedRectWithMaxArea(w, h, math.radians(rand_angle_deg))
        return T_f.center_crop(rotate_pad, [h_c, w_c])

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(angle_deg={self.angle_deg},
interpolation={self.interpolation.value})"""


class AugmentTransforms:
    def __init__(self, transform_config: ListConfig):
        self.transforms = torchvision.transforms.Compose(
            [get_class(item.name)(**item.get("params", {})) for item in transform_config]
        )

    def __call__(self, img: torch.Tensor):
        return self.transforms(img)

    def __repr__(self) -> str:
        return self.transforms.__repr__()
