"""
Implements all transformation functions working specifically with this dataset dictionary.
"""

import torch

import torchvision as tv


class Scale(object):
    """
    Rescales the image to the given scale and the ground truth to size/2 to fit the output of the network.
    """

    def __init__(self, size):
        self.transform_image = tv.transforms.Resize(size)
        geo_size = int(size/2)
        self.transform_geometry = tv.transforms.Resize(geo_size, interpolation=tv.transforms.InterpolationMode.NEAREST)

    def __call__(self, sample):
        image, depth, normal = sample['image'], sample['depth'], sample['normal']
        mask, planes = sample['mask'], sample['planes']

        image = self.transform_image(image)
        depth = self.transform_geometry(depth)
        normal = self.transform_geometry(normal)
        mask = self.transform_geometry(mask)
        planes = self.transform_geometry(planes)

        # reassigns plane labels as some planes might have been lost during rescale
        planes[1] = ((planes[1].unsqueeze(-1) == planes[1].unique()) *
                     torch.arange(len(planes[1].unique())).unsqueeze(0).unsqueeze(0)).sum(dim=2)

        return {'image': image, 'depth': depth, 'normal': normal, 'mask': mask, 'planes': planes}

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self):
        self.transform = tv.transforms.ToTensor()

    def __call__(self, sample):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        image, depth, normal = sample['image'], sample['depth'], sample['normal']
        mask, planes = sample['mask'], sample['planes']
        return {'image': self.transform(image), 'depth': self.transform(depth),
                'normal': self.transform(normal), 'mask': self.transform(mask),
                'planes': self.transform(planes)}


class SetNANTo(object):
    """
    Sets all the nan values from the mask to 'value' value.
    """
    def __init__(self, value):
        self.value = value

    def __call__(self, sample):
        image, depth, normal = sample['image'], sample['depth'], sample['normal']
        mask, planes = sample['mask'], sample['planes']
        depth[torch.isnan(depth)] = self.value
        normal[torch.isnan(normal)] = self.value
        return {'image': image, 'depth': depth, 'normal': normal, 'mask': mask, 'planes': planes}


class Grayscale(object):
    def __init__(self, p=0.1):
        self.transform = tv.transforms.RandomGrayscale(p)

    def __call__(self, sample):
        image, depth, normal = sample['image'], sample['depth'], sample['normal']
        mask, planes = sample['mask'], sample['planes']
        return {'image': self.transform(image), 'depth': depth, 'normal': normal, 'mask': mask, 'planes': planes}


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.transform = tv.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        image, depth, normal = sample['image'], sample['depth'], sample['normal']
        mask, planes = sample['mask'], sample['planes']
        return {'image': self.transform(image), 'depth': depth, 'normal': normal, 'mask': mask, 'planes': planes}
