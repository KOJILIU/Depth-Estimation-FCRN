import os
import numpy as np
import h5py
from PIL import Image
import torch
import torch.utils.data as data
import flow_transforms
import torchvision.transforms as transforms


class NyuDepthLoader(data.Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        # 相当于把list里面的树全拿出来
        img_idx = self.lists[index]

        # depth y x
        img = self.imgs[img_idx].transpose(2, 1, 0)
        #img = self.imgs[img_idx]
        #  y x
        dpt = self.dpts[img_idx].transpose(1, 0)
        #dpt = self.dpts[img_idx]

        """ Rescales the inputs and target arrays to the given 'size'.
                    'size' will be the size of the smaller edge.
                    For example, if height > width, then image will be
                    rescaled to (size * height / width, size)
                    size: size of the smaller edge
                    interpolation order: Default: 2 (bilinear)
        """
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

        input_transform = transforms.Compose([flow_transforms.Scale(228),
                                              flow_transforms.ArrayToTensor()])

        target_depth_transform = transforms.Compose([flow_transforms.Scale_Single(228),
                                                     flow_transforms.ArrayToTensor()])

        img = input_transform(img)
        dpt = target_depth_transform(dpt)

        #image = Image.fromarray(np.uint8(img))
        #image.save('img2.png')
        return img, dpt

    def __len__(self):
        return len(self.lists)
