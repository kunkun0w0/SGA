import os
import torch
import glob
import os.path as osp
from model import Generator
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from torchvision.utils import save_image
import cv2
import numpy as np


device1 = torch.device('cuda:0')
device2 = torch.device('cuda:1')
G = Generator()


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class DataSet(data.Dataset):
    def __init__(self, img_transform_gt, img_transform_sketch, dataset_name):
        self.img_transform_gt = img_transform_gt
        self.img_transform_sketch = img_transform_sketch

        if dataset_name == 'anime':
            self.img_dir = '/hdd/user1/LZK/animeGAN/anime/test2'
            self.skt_dir = '/hdd/user1/LZK/animeGAN/anime/test0'
            self.data_list = glob.glob(os.path.join(self.img_dir, '*.png'))
        elif dataset_name == 'afhq_cat':
            self.img_dir = '/hdd/user1/LZK/animeGAN/afhq/val/cat'
            self.skt_dir = '/hdd/user1/LZK/animeGAN/afhq/val/cat_sketch'
            self.data_list = glob.glob(os.path.join(self.img_dir, '*.jpg'))
        elif dataset_name == 'afhq_dog':
            self.img_dir = '/hdd/user1/LZK/animeGAN/afhq/val/dog'
            self.skt_dir = '/hdd/user1/LZK/animeGAN/afhq/val/dog_sketch'
            self.data_list = glob.glob(os.path.join(self.img_dir, '*.jpg'))
        elif dataset_name == 'afhq_wild':
            self.img_dir = '/hdd/user1/LZK/animeGAN/afhq/val/wild'
            self.skt_dir = '/hdd/user1/LZK/animeGAN/afhq/val/wild_sketch'
            self.data_list = glob.glob(os.path.join(self.img_dir, '*.jpg'))

        self.img_size = (256, 256, 3)

        self.data_list = [osp.basename(x) for x in self.data_list]
        self.data_list = list(set(self.data_list))

    def __getitem__(self, index):
        fid = self.data_list[index]
        reference = Image.open(
            osp.join(self.img_dir, '{}'.format(fid))).convert('RGB')
        sketch = Image.open(
            osp.join(self.skt_dir, '{}'.format(fid))).convert('L')

        return fid, self.img_transform_gt(reference), self.img_transform_sketch(sketch)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(dataset_name):
    img_transform_gt = list()
    img_transform_sketch = list()
    img_size = 256

    img_transform_gt.append(T.Resize((img_size, img_size)))
    img_transform_gt.append(T.ToTensor())
    img_transform_gt.append(T.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform_gt = T.Compose(img_transform_gt)

    img_transform_sketch.append(T.Resize((img_size, img_size)))
    img_transform_sketch.append(T.ToTensor())
    img_transform_sketch.append(T.Normalize(mean=0.5, std=0.5))
    img_transform_sketch = T.Compose(img_transform_sketch)

    dataset = DataSet(img_transform_gt, img_transform_sketch, dataset_name)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True)
    return data_loader


def image_save(gen, fid, sample_dir):
    img_data = list(torch.chunk(gen, 16, dim=0))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for i in range(16):
        sample_path = os.path.join(sample_dir, fid[i])
        save_image(denorm(img_data[i].data.cpu()), sample_path, nrow=1, padding=0)


def load_model(dataset_name, epoch):
    G_path = './' + dataset_name + '/models/{}-G.pth'.format(epoch)
    G_checkpoint = torch.load(G_path)
    G.load_state_dict(G_checkpoint['model'])
    G.to(device1)
    G.eval()


if __name__ == '__main__':
    dataset_name = 'anime'
    test_loader = get_loader(dataset_name)

    iterations = len(test_loader)
    test_iter = iter(test_loader)

    sample_dir = None
    test_dir = None
    if dataset_name == 'anime':
        sample_dir = './anime/predict'
        test_dir = '/hdd/user1/LZK/animeGAN/anime/test2'
    elif dataset_name == 'afhq_cat':
        sample_dir = './afhq_cat/predict'
        test_dir = '/hdd/user1/LZK/animeGAN/afhq/val/cat'
    elif dataset_name == 'afhq_dog':
        sample_dir = './afhq_dog/predict'
        test_dir = '/hdd/user1/LZK/animeGAN/afhq/val/dog'
    elif dataset_name == 'afhq_wild':
        sample_dir = './afhq_wild/predict'
        test_dir = '/hdd/user1/LZK/animeGAN/afhq/val/wild'
    assert sample_dir is not None
    assert test_dir is not None

    save_dir = './' + dataset_name + '/npy'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    avg1 = list()
    avg2 = list()

    print("start predict!")
    print(iterations)
    load_model(dataset_name, 20)
    for i in range(iterations):
        try:
            fid, ref, skt = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            fid, ref, skt = next(test_iter)

        split_ref = list(torch.chunk(ref, 16, dim=0))
        first_split_ref = split_ref[0]
        del split_ref[0]
        split_ref.append(first_split_ref)
        shift_ref = torch.cat(split_ref, dim=0)

        shift_ref = shift_ref.to(device1)
        skt = skt.to(device1)

        result, before, after = G.predict(shift_ref, skt)

        u1, s1, v1 = torch.linalg.svd(before.data.cpu().squeeze())
        u2, s2, v2 = torch.linalg.svd(after.data.cpu().squeeze())
        s1, indices1 = s1.sort(descending=True)
        s2, indices2 = s2.sort(descending=True)

        s1 = (s1 ** 2.0).numpy()
        L1 = 256
        for j in range(L1 - 1):
            s1[:, j + 1] += s1[:, j]
        sm1 = s1[:, -1]
        sm1 = sm1[:, np.newaxis]
        s1 = s1 / sm1
        s1 = np.mean(s1, axis=0)
        npy_path_1 = osp.join(save_dir, f'before{i + 1}.npy')
        np.save(npy_path_1, s1)

        s2 = (s2 ** 2.0).numpy()
        L2 = 256
        for j in range(L2 - 1):
            s2[:, j + 1] += s2[:, j]
        sm2 = s2[:, -1]
        sm2 = sm2[:, np.newaxis]
        s2 = s2 / sm2
        s2 = np.mean(s2, axis=0)
        npy_path_2 = osp.join(save_dir, f'after{i + 1}.npy')
        np.save(npy_path_2, s2)

        avg1.append(s1)
        avg2.append(s2)

    print("finish predict!")

    sm1 = avg1[0]
    for i in range(1, len(avg1)):
        sm1 += avg1[i]

    avg1 = sm1 / len(avg1)
    npy_path_1 = osp.join(save_dir, 'SGA_before_avg.npy')
    np.save(npy_path_1, avg1)

    sm2 = avg2[0]
    for i in range(1, len(avg2)):
        sm2 += avg2[i]

    avg2 = sm2 / len(avg2)
    npy_path_2 = osp.join(save_dir, 'SGA_after_avg.npy')
    np.save(npy_path_2, avg2)



