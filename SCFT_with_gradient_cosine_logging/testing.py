import os
import torch
import glob
import os.path as osp
from model import Generator
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')
G = Generator()


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class DataSet(data.Dataset):
    def __init__(self, img_transform_gt, img_transform_sketch, dataset_name):
        self.img_transform_gt = img_transform_gt
        self.img_transform_sketch = img_transform_sketch

        self.img_dir = './' + dataset_name + '/exp'

        self.ref_name = glob.glob(os.path.join(self.img_dir, 'ref.*'))
        self.skt_name = glob.glob(os.path.join(self.img_dir, 'skt.*'))
        print(self.ref_name)
        print(self.skt_name)

        self.img_size = (256, 256, 3)

    def __getitem__(self, index):
        reference = Image.open(self.ref_name[0]).convert('RGB')
        sketch = Image.open(self.skt_name[0]).convert('L')

        return self.img_transform_gt(reference), self.img_transform_sketch(sketch)

    def __len__(self):
        """Return the number of images."""
        return 2


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
                                  batch_size=1,
                                  num_workers=1)
    return data_loader


def image_save(gen, fid, sample_dir):
    sample_path = sample_dir + '/' + fid + '.png'
    save_image(denorm(gen.data.cpu()), sample_path, nrow=1, padding=0)


def load_model(dataset_name, epoch):
    G_path = './' + dataset_name + '/models/{}-G.pth'.format(epoch)
    G_checkpoint = torch.load(G_path)
    G.load_state_dict(G_checkpoint['model'])
    # G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    G.to(device)
    G.eval()

if __name__ == '__main__':
    dataset_name = 'anime'
    test_loader = get_loader(dataset_name)
    load_model(dataset_name, 20)

    test_iter = iter(test_loader)
    ref, skt = next(test_iter)
    ref = ref.to(device)
    skt = skt.to(device)

    result, attention_map = G.att(ref, skt)
    image_save(result, 'result', './' + dataset_name + '/exp')

    # u1, s1, v1 = torch.linalg.svd(before.data.cpu().squeeze())
    # u2, s2, v2 = torch.linalg.svd(after.data.cpu().squeeze())
    # s1, indices1 = s1.sort(descending=True)
    # s2, indices2 = s2.sort(descending=True)
    #
    # s1 = (s1 ** 2.0).numpy()
    # L1 = len(s1)
    # for i in range(L1 - 1):
    #     s1[i + 1] += s1[i]
    # s1 = s1 / s1[-1]
    # np.save("./"+dataset_name+"/exp/sga_before.npy", s1)
    #
    # s2 = (s2 ** 2.0).numpy()
    # L2 = len(s2)
    # for i in range(L2 - 1):
    #     s2[i + 1] += s2[i]
    # s2 = s2 / s2[-1]
    # np.save("./" + dataset_name + "/exp/sga_after.npy", s2)

    # plt.plot(s1, label='before')
    # plt.plot(s2, label='after')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(fontsize=15)
    # plt.xlabel('k', fontsize=15, weight='bold')
    # plt.ylabel('ratio', fontsize=15, weight='bold')
    # plt.savefig("./"+dataset_name+"/exp/line.png")

