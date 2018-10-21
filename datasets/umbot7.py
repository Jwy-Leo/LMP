from __future__ import print_function
import random
import torch
import torch.utils.data as data
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import maskUtils

import os
import os.path
import pdb
__all__ = ['Umbot7', 'get_dts_idx_list', 'get_iter_dts_loder']


def get_dts_idx_list(loader_list):
    dts_size_list = []
    for loader in loader_list:
        dts_size_list.append(len(loader))
    n_batch = sum(dts_size_list)
    dts_idx_list = torch.LongTensor(n_batch)
    cnt = 0
    for i in range(0, len(dts_size_list)):
        dts_idx_list[cnt:cnt + dts_size_list[i]] = i
        cnt += dts_size_list[i]

    return dts_idx_list


def get_iter_dts_loder(loader_list):
    iter_dts_loader_list = []
    for loader in loader_list:
        iter_dts_loader_list.append(iter(loader))
    return iter_dts_loader_list


class Umbot7(data.Dataset):
    def __init__(self, img_dir, infos_file_path, n_class):

        assert os.path.isdir(img_dir), \
            '%s is not a directory' % img_dir

        assert os.path.isfile(infos_file_path), \
            '%s is not a file' % infos_file_path

        self._img_dir = img_dir
        self._infos_file_path = infos_file_path
        self._infos = load_lua(infos_file_path)
        self._n_class = n_class
    def __getitem__(self, index):

        img_file = os.path.join(self._img_dir,
                                str(self._infos.data[index].id) + '.jpg')

        img = transforms.ToTensor()(Image.open(img_file).convert('RGB'))
        img_w, img_h = img.size(2), img.size(1)

        target = torch.LongTensor(img_h, img_w).zero_()
        for inst in self._infos.data[index].insts:

            polys = []
            # { bg, person, bicycle, car, motorcycle, truck, bus, train }
            if self._n_class == 7:
                if inst.category_idx == 8:
                    inst.category_idx = 5
                if inst.category_idx <= 6:
                    for poly in inst.seg:
                        polys.append(poly.tolist())
            # { bg, person }
            elif self._n_class == 2:
                if inst.category_idx == 1:
                    for poly in inst.seg:
                        polys.append(poly.tolist())

            if polys:
                rles = maskUtils.frPyObjects(polys, img_h, img_w)
                rle = maskUtils.merge(rles)
                mask = maskUtils.decode(rle)
                target.masked_fill_(torch.from_numpy(mask),
                                    inst.category_idx)

        p_w = self._infos.patchSize.w
        p_h = self._infos.patchSize.h

        x0 = random.randint(0, (img_w - p_w))
        y0 = random.randint(0, (img_h - p_h))

        img = img[:, y0:y0+p_h, x0:x0+p_w]
        target = target[y0:y0+p_h, x0:x0+p_w]

        return img, target

    def __len__(self):
        return len(self._infos.data)
