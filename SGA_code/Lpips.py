import os
import torch
import glob
import os.path as osp
from model import Generator
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths
import cv2
import numpy as np
import lpips


device = torch.device('cuda:0')
G = Generator()
loss_fn = lpips.LPIPS(net='alex').to(device)


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class DataSet(data.Dataset):
    def __init__(self, img_transform_gt, img_transform_sketch, dataset_name):
        self.img_transform_gt = img_transform_gt
        self.img_transform_sketch = img_transform_sketch

        if dataset_name == 'anime':
            self.img_dir = './anime/test2'
            self.skt_dir = './anime/test0'
            self.data_list = glob.glob(os.path.join(self.img_dir, '*.png'))
        elif dataset_name == 'afhq_cat':
            self.img_dir = './afhq/val/cat'
            self.skt_dir = './afhq/val/cat_sketch'
            self.data_list = glob.glob(os.path.join(self.img_dir, '*.jpg'))
        elif dataset_name == 'afhq_dog':
            self.img_dir = './afhq/val/dog'
            self.skt_dir = './afhq/val/dog_sketch'
            self.data_list = glob.glob(os.path.join(self.img_dir, '*.jpg'))
        elif dataset_name == 'afhq_wild':
            self.img_dir = './afhq/val/wild'
            self.skt_dir = './afhq/val/wild_sketch'
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
                                  batch_size=8,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True)
    return data_loader


def image_save(gen, fid, sample_dir):
    img_data = list(torch.chunk(gen, 8, dim=0))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for i in range(8):
        sample_path = os.path.join(sample_dir, fid[i])
        save_image(denorm(img_data[i].data.cpu()), sample_path, nrow=1, padding=0)


def load_model(dataset_name, epoch):
    G_path = './' + dataset_name + '/models/{}-G.pth'.format(epoch)
    G_checkpoint = torch.load(G_path)
    G.load_state_dict(G_checkpoint['model'])
    G.to(device)
    G.eval()


if __name__ == '__main__':
    dataset_name = 'afhq_cat'
    test_loader = get_loader(dataset_name)

    iterations = len(test_loader)
    test_iter = iter(test_loader)
    epochs = 40

    sample_dir = None
    test_dir = None
    txt_path = None
    if dataset_name == 'anime':
        sample_dir = './anime/predict'
        test_dir = './anime/test2'
        txt_path = './anime/lpips.txt'
    elif dataset_name == 'afhq_cat':
        sample_dir = './afhq_cat/predict'
        test_dir = './afhq/val/cat'
        txt_path = './afhq_cat/lpips.txt'
    elif dataset_name == 'afhq_dog':
        sample_dir = './afhq_dog/predict'
        test_dir = './afhq/val/dog'
        txt_path = './afhq_dog/lpips.txt'
    elif dataset_name == 'afhq_wild':
        sample_dir = './afhq_wild/predict'
        test_dir = './afhq/val/wild'
        txt_path = './afhq_wild/lpips.txt'
    assert sample_dir is not None
    assert test_dir is not None
    assert txt_path is not None

    print("start predict!")
    print(iterations)
    for e in range(epochs):
        load_model(dataset_name, e + 1)
        lpips_score = 0
        for i in range(iterations):
            try:
                fid, ref, skt = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                fid, ref, skt = next(test_iter)

            split_ref = list(torch.chunk(ref, 8, dim=0))
            first_split_ref = split_ref[0]
            del split_ref[0]
            split_ref.append(first_split_ref)
            shift_ref = torch.cat(split_ref, dim=0)

            shift_ref = shift_ref.to(device)
            skt = skt.to(device)

            shift_gen = G(shift_ref, skt)
            image_save(shift_gen, fid, sample_dir)
            lpips_score += loss_fn.forward(shift_gen.data.to(device), shift_ref.data.to(device)).mean()

        lpips_score = lpips_score.item() / float(iterations)

        with open(txt_path, 'a') as fp:
            print(f'Epoch {e + 1}', file=fp)
            print(f'Lpips : {lpips_score}', file=fp)

    print("finish predict!")
