'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import os
import random
import hashlib
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


BUF_SIZE = 65536


def get_md5_from_source_path(source_path):
    pattern = os.path.join(source_path, '**', '**', '*.png')
    files = sorted(list(glob(pattern)))
    assert len(files)

    md5 = hashlib.md5()

    for file_path in files:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)

    return md5.hexdigest()


def _save_images_to_folder(dataset, transform, path, split_name, idx, format_='.png'):
    scales = {}
    for el in dataset:
        img = transform(el[0])
        out = os.path.join(path, split_name, str(el[1]))
        if not os.path.exists(out):
            os.makedirs(out)
        img_path = os.path.join(out, str(idx) + format_)
        img.save(img_path)
        idx += 1
    return idx


def make_mnist_scale_50k(source, dest, min_scale, max_scale, download=False, seed=0, **kwargs):
    '''
    We follow the procedure described in 
    https://arxiv.org/pdf/1807.11783.pdf
    https://arxiv.org/pdf/1906.03861.pdf
    '''
    MNIST_TRAIN_SIZE = 10000
    MNIST_VAL_SIZE = 2000
    MNIST_TEST_SIZE = 50000

    np.random.seed(seed)
    random.seed(seed)
    # 3 stands for PIL.Image.BICUBIC
    transform = transforms.RandomAffine(0, scale=(min_scale, max_scale), resample=3)

    dataset_train = datasets.MNIST(root=source, train=True, download=download)
    dataset_test = datasets.MNIST(root=source, train=False, download=download)
    concat_dataset = ConcatDataset([dataset_train, dataset_test])

    labels = [el[1] for el in concat_dataset]
    train_val_size = MNIST_TRAIN_SIZE + MNIST_VAL_SIZE
    train_val, test = train_test_split(concat_dataset, train_size=train_val_size,
                                       test_size=MNIST_TEST_SIZE, stratify=labels)

    labels = [el[1] for el in train_val]
    train, val = train_test_split(train_val, train_size=MNIST_TRAIN_SIZE,
                                  test_size=MNIST_VAL_SIZE, stratify=labels)

    dest = os.path.expanduser(dest)
    dataset_path = os.path.join(dest, 'MNIST_scale', "seed_{}".format(seed))
    dataset_path = os.path.join(dataset_path, "scale_{}_{}".format(min_scale, max_scale))
    print('OUTPUT: {}'.format(dataset_path))

    idx = _save_images_to_folder(train, transform, dataset_path, 'train', 0, '.png')
    idx = _save_images_to_folder(test, transform, dataset_path, 'test', idx, '.png')
    idx = _save_images_to_folder(val, transform, dataset_path, 'val', idx, '.png')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='source folder of the dataset')
    parser.add_argument('--dest', type=str, required=True, help='destination folder for the output')
    parser.add_argument('--min_scale', type=float, required=True,
                        help='min scale for the generated dataset')
    parser.add_argument('--max_scale', type=float, default=1.0,
                        help='max scale for the generated dataset')
    parser.add_argument('--download', action='store_true',
                        help='donwload stource dataset if needed.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--validate', action='store_true', default=False)
    args = parser.parse_args()

    if args.validate:
        dest = os.path.expanduser(args.dest)
        dataset_path = os.path.join(dest, 'MNIST_scale', "seed_{}".format(args.seed))
        dataset_path = os.path.join(dataset_path,
                                    "scale_{}_{}".format(args.min_scale, args.max_scale))
        print(get_md5_from_source_path(dataset_path))
    else:
        for k, v in vars(args).items():
            print('{}={}'.format(k, v))

        make_mnist_scale_50k(**vars(args))
