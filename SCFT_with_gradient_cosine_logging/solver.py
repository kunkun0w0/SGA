import os
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import glob
import os.path as osp
from logger import Logger

from model import Generator
from model import Discriminator
from torchvision.models import vgg19
from torchvision.utils import save_image
import torch.nn.functional as F


vgg_activation = dict()


def get_activation(name):
    def hook(model, input, output):
        vgg_activation[name] = output

    return hook


def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class Solver(object):

    def __init__(self, config, data_loader):
        """Initialize configurations."""
        self.data_loader = data_loader
        self.img_size = config['MODEL_CONFIG']['IMG_SIZE']
        assert self.img_size in [256]

        self.epoch = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.g_lr = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr = float(config['TRAINING_CONFIG']['D_LR'])
        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_g_recon = config['TRAINING_CONFIG']['LAMBDA_G_RECON']
        self.lambda_g_style = config['TRAINING_CONFIG']['LAMBDA_G_SYTLE']
        self.lambda_g_percep = config['TRAINING_CONFIG']['LAMBDA_G_PERCEP']
        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.d_critic = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic = config['TRAINING_CONFIG']['G_CRITIC']
        self.mse_loss = nn.MSELoss()

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']
        self.gradient_cosine = (config['TRAINING_CONFIG']['GRADIENT_COSINE']=='true')

        self.adversarial_loss = torch.nn.MSELoss()

        self.l1_loss = torch.nn.L1Loss()

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']

        self.g_spec = config['TRAINING_CONFIG']['G_SPEC'] == 'True'
        self.d_spec = config['TRAINING_CONFIG']['D_SPEC'] == 'True'

        self.gpu = config['TRAINING_CONFIG']['GPU']

        self.target_layer = ['relu_3', 'relu_8']

        # Directory
        self.train_dir = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir = os.path.join(
            self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(
            self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(
            self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir = os.path.join(
            self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step = config['TRAINING_CONFIG']['LR_DECAY_STEP']

        # build net
        self.vgg = vgg19(pretrained=True)
        self.D = Discriminator(spec_norm=self.d_spec, LR=0.2).to(self.gpu)
        self.G = Generator(spec_norm=self.g_spec).to(self.gpu)
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr,
                                            (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr,
                                            (self.beta1, self.beta2))
        self.build_model()

        self.writer = Logger(self.log_dir)

    def build_model(self):
        for layer in self.target_layer:
            self.vgg.features[int(layer.split('_')[-1])].register_forward_hook(get_activation(layer))
        self.vgg.to(self.gpu)
        """
        self.vgg.features[3].register_forward_hook(get_activation('relu_3'))
        self.vgg.features[8].register_forward_hook(get_activation('relu_8'))
        self.vgg.features[17].register_forward_hook(get_activation('relu_17'))
        self.vgg.features[26].register_forward_hook(get_activation('relu_26'))
        self.vgg.features[35].register_forward_hook(get_activation('relu_35'))
        """
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir, 'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params), file=fp)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def restore_model(self):

        pth_list = glob.glob(osp.join(self.model_dir, '*-G.pth'))

        if len(pth_list) == 0:
            return 0

        pth_list = [int(osp.basename(x).split("-")[0]) for x in pth_list]
        pth_list.sort()
        epoch = pth_list[-1]
        G_path = os.path.join(self.model_dir, '{}-G.pth'.format(epoch))
        D_path = os.path.join(self.model_dir, '{}-D.pth'.format(epoch))

        G_checkpoint = torch.load(G_path)
        self.G.load_state_dict(G_checkpoint['model'])
        self.g_optimizer.load_state_dict(G_checkpoint['optimizer'])

        D_checkpoint = torch.load(D_path)
        self.D.load_state_dict(D_checkpoint['model'])
        self.d_optimizer.load_state_dict(D_checkpoint['optimizer'])

        self.G.to(self.gpu)
        self.D.to(self.gpu)
        return epoch

    def image_reporting(self, fixed_sketch, fixed_reference, fixed_elastic_reference, epoch, postfix=''):
        image_report = list()
        image_report.append(fixed_sketch.expand_as(fixed_reference))
        image_report.append(fixed_elastic_reference)
        image_report.append(fixed_reference)

        fake_result = self.G(fixed_elastic_reference, fixed_sketch)
        image_report.append(fake_result)

        x_concat = torch.cat(image_report, dim=3)
        sample_path = os.path.join(self.sample_dir, '{}-images{}.png'.format(epoch, postfix))
        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

    @staticmethod
    def cal_cosine(grad1, grad2):
        grad1 = F.normalize(grad1, dim=-1)
        grad2 = F.normalize(grad2, dim=-1)
        dot = grad1 * grad2
        cosine = torch.sum(dot, dim=-1).view(-1)
        return cosine

    def gradient_cosine_statics(self, Z_grad, Q_grad, K_grad, V_grad):
        grad_k = K_grad[0]
        grad_v = V_grad[0]
        grad_z = Z_grad[0]
        grad_q = Q_grad[0]
        grad_X = grad_z + grad_q
        grad_Y = grad_k + grad_v
        b, n, c = grad_k.size()

        # pixel level ref and skt value and attention
        cosine1 = self.cal_cosine(grad_v, grad_Y).data.cpu().numpy()
        cosine2 = self.cal_cosine(grad_z, grad_X).data.cpu().numpy()
        cosine3 = self.cal_cosine(grad_k, grad_Y).data.cpu().numpy()
        cosine4 = self.cal_cosine(grad_q, grad_X).data.cpu().numpy()

        grad_k = grad_k.reshape(b, -1)
        grad_v = grad_v.reshape(b, -1)
        grad_z = grad_z.reshape(b, -1)
        grad_q = grad_q.reshape(b, -1)
        grad_X = grad_X.reshape(b, -1)
        grad_Y = grad_Y.reshape(b, -1)

        # instance level ref and skt value and attention
        cosine5 = self.cal_cosine(grad_v, grad_Y).data.cpu().numpy()
        cosine6 = self.cal_cosine(grad_z, grad_X).data.cpu().numpy()
        cosine7 = self.cal_cosine(grad_k, grad_Y).data.cpu().numpy()
        cosine8 = self.cal_cosine(grad_q, grad_X).data.cpu().numpy()

        return cosine1, cosine2, cosine3, cosine4, cosine5, cosine6, cosine7, cosine8

    def percep_style(self, reference, fake_images):
        fake_activation = dict()
        real_activation = dict()

        # percep_style
        g_loss_style = 0
        g_loss_percep = 0

        vgg_activation.clear()
        self.vgg(reference)
        for layer in self.target_layer:
            real_activation[layer] = vgg_activation[layer]
        vgg_activation.clear()

        self.vgg(fake_images)
        for layer in self.target_layer:
            fake_activation[layer] = vgg_activation[layer]
        vgg_activation.clear()

        for layer in self.target_layer:
            g_loss_percep += self.l1_loss(fake_activation[layer], real_activation[layer])
            g_loss_style += self.l1_loss(gram_matrix(fake_activation[layer]), gram_matrix(real_activation[layer]))

        return g_loss_percep, g_loss_style

    def G_train(self, sketch, reference, elastic_reference, loss_dict, gradient_cosine=False):
        if gradient_cosine:
            fake_images, Z_grad, Q_grad, K_grad, V_grad = self.G.gradient_cosine(elastic_reference, sketch)
        else:
            fake_images = self.G(elastic_reference, sketch)

        fake_score = self.D(torch.cat([fake_images, sketch], dim=1))

        g_loss_fake = self.adversarial_loss(fake_score, torch.ones_like(fake_score))

        g_loss_recon = self.l1_loss(fake_images, reference)

        g_loss_percep, g_loss_style = self.percep_style(reference, fake_images)

        g_loss = self.lambda_g_fake * g_loss_fake + \
                 self.lambda_g_percep * g_loss_percep + \
                 self.lambda_g_style * g_loss_style + \
                 self.lambda_g_recon * g_loss_recon

        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        # Logging.
        loss_dict['G/loss_fake'] = self.lambda_g_fake * g_loss_fake.item()
        loss_dict['G/loss_style'] = self.lambda_g_style * g_loss_style.item()
        loss_dict['G/loss_percep'] = self.lambda_g_percep * g_loss_percep.item()
        loss_dict['G/loss_recon'] = self.lambda_g_recon * g_loss_recon.item()
        if gradient_cosine:
            return Z_grad, Q_grad, K_grad, V_grad

    def D_train(self, sketch, reference, elastic_reference, loss_dict):
        fake_images = self.G(elastic_reference, sketch)

        real_score = self.D(torch.cat([reference, sketch], dim=1))
        fake_score = self.D(torch.cat([fake_images.detach(), sketch], dim=1))

        d_loss_real = self.adversarial_loss(real_score, torch.ones_like(real_score))
        d_loss_fake = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))

        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake

        # Backward and optimize.
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Logging.
        loss_dict['D/loss_real'] = self.lambda_d_real * d_loss_real.item()
        loss_dict['D/loss_fake'] = self.lambda_d_fake * d_loss_fake.item()

    def train(self):
        # Set data loader.
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print('iterations : ', iterations)

        data_iter = iter(data_loader)
        _, fixed_elastic_reference, fixed_reference, fixed_sketch = next(data_iter)

        split_fixed_sketch = list(torch.chunk(fixed_sketch, self.batch_size, dim=0))  
        first_fixed_sketch = split_fixed_sketch[0] 
        del split_fixed_sketch[0] 
        split_fixed_sketch.append(first_fixed_sketch)  
        shifted_fixed_sketch = torch.cat(split_fixed_sketch, dim=0) 

        fixed_sketch = fixed_sketch.to(self.gpu)
        fixed_reference = fixed_reference.to(self.gpu)
        fixed_elastic_reference = fixed_elastic_reference.to(self.gpu)
        shifted_fixed_sketch = shifted_fixed_sketch.to(self.gpu)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_epoch = self.restore_model()
        start_time = time.time()
        print(self.train_dir)
        print('Start training...')
        for e in range(start_epoch, self.epoch):
            hist = list()

            for i in range(iterations):
                try:
                    _, elastic_reference, reference, sketch = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, elastic_reference, reference, sketch = next(data_iter)

                elastic_reference = elastic_reference.to(self.gpu)
                reference = reference.to(self.gpu)
                sketch = sketch.to(self.gpu)

                loss_dict = dict()
                if (i + 1) % self.d_critic == 0:
                    self.D_train(sketch, reference, elastic_reference, loss_dict)

                if (i + 1) % self.g_critic == 0:
                    if self.gradient_cosine:
                        Z_grad, Q_grad, K_grad, V_grad = self.G_train(sketch, reference, elastic_reference, loss_dict, gradient_cosine=True)
                        cosine = self.gradient_cosine_statics(Z_grad, Q_grad, K_grad, V_grad)
                        hist.append(cosine)
                    else:
                        self.G_train(sketch, reference, elastic_reference, loss_dict)

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(
                        e + 1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    txt_name = f'{e + 1}.txt'
                    with open(os.path.join(self.log_dir, txt_name), 'a') as fp:
                        print(log, file=fp)

            if (e + 1) % self.sample_step == 0:
                self.G.eval()
                with torch.no_grad():
                    self.image_reporting(fixed_sketch, fixed_reference, fixed_elastic_reference, e + 1, postfix='')
                    self.image_reporting(shifted_fixed_sketch, fixed_reference, fixed_elastic_reference,
                                         e + 1, postfix='_shifted')
                    print('Saved real and fake images into {}...'.format(self.sample_dir))
                self.G.train()

            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                G_path = os.path.join(self.model_dir, '{}-G.pth'.format(e + 1))
                D_path = os.path.join(self.model_dir, '{}-D.pth'.format(e + 1))

                G_state = {'model': self.G.state_dict(), 'optimizer': self.g_optimizer.state_dict()}
                D_state = {'model': self.D.state_dict(), 'optimizer': self.d_optimizer.state_dict()}

                torch.save(G_state, G_path)
                torch.save(D_state, D_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

            if self.gradient_cosine:
                hist1 = np.concatenate([x[0] for x in hist], axis=0)
                hist2 = np.concatenate([x[1] for x in hist], axis=0)
                hist3 = np.concatenate([x[2] for x in hist], axis=0)
                hist4 = np.concatenate([x[3] for x in hist], axis=0)
                hist5 = np.concatenate([x[4] for x in hist], axis=0)
                hist6 = np.concatenate([x[5] for x in hist], axis=0)
                hist7 = np.concatenate([x[6] for x in hist], axis=0)
                hist8 = np.concatenate([x[7] for x in hist], axis=0)

                self.writer.histogram_summary('v-pixel ref', hist1, e + 1)
                self.writer.histogram_summary('v-pixel skt', hist2, e + 1)
                self.writer.histogram_summary('a-pixel ref', hist3, e + 1)
                self.writer.histogram_summary('a-pixel skt', hist4, e + 1)
                self.writer.histogram_summary('v-instance ref', hist5, e + 1)
                self.writer.histogram_summary('v-instance skt', hist6, e + 1)
                self.writer.histogram_summary('a-instance ref', hist7, e + 1)
                self.writer.histogram_summary('a-instance skt', hist8, e + 1)

                print(hist1)

        print('Training is finished')

    def load_model(self, dataset_name, epoch):
        G_path = './' + dataset_name + '/models/{}-G.pth'.format(epoch)
        G_checkpoint = torch.load(G_path)
        self.G.load_state_dict(G_checkpoint['model'])
        self.G.to(self.gpu)

    def gradient_cosine_test(self):
        # viz = Visdom()
        # Set data loader
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print('iterations : ', iterations)
        data_iter = iter(data_loader)

        print(self.train_dir)
        self.load_model(self.train_dir, 40)
        print('Start gathering gradient cosine ...')
        histogram1 = list()
        histogram2 = list()
        histogram3 = list()
        histogram4 = list()

        for i in range(iterations):
            try:
                _, elastic_reference, reference, sketch = next(data_iter)
            except:
                data_iter = iter(data_loader)
                _, elastic_reference, reference, sketch = next(data_iter)

            elastic_reference = elastic_reference.to(self.gpu)
            reference = reference.to(self.gpu)
            sketch = sketch.to(self.gpu)

            # loss
            fake_images, Z_grad, Q_grad, K_grad, V_grad = self.G.gradient_cosine(elastic_reference, sketch)
            fake_score = self.D(torch.cat([fake_images, sketch], dim=1))
            g_loss_fake = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
            g_loss_recon = self.l1_loss(fake_images, reference)
            g_loss_percep, g_loss_style = self.percep_style(reference, fake_images)
            g_loss = self.lambda_g_fake * g_loss_fake + \
                     self.lambda_g_percep * g_loss_percep + \
                     self.lambda_g_style * g_loss_style + \
                     self.lambda_g_recon * g_loss_recon
            self.reset_grad()
            g_loss.backward()

            grad_k = K_grad[0]
            grad_v = V_grad[0]
            grad_z = Z_grad[0]
            grad_q = Q_grad[0]

            grad_z = grad_z + grad_q
            grad_v = grad_v + grad_k

            b, n, c = grad_k.size()

            # pixel level ref and skt
            cosine1 = self.cal_cosine(grad_k, grad_v)
            cosine2 = self.cal_cosine(grad_z, grad_q)

            grad_k = grad_k.view(b, -1)
            grad_v = grad_v.view(b, -1)
            grad_z = grad_z.view(b, -1)
            grad_q = grad_q.view(b, -1)

            # instance level ref and skt
            cosine3 = self.cal_cosine(grad_k, grad_v)
            cosine4 = self.cal_cosine(grad_z, grad_q)

            histogram1.append(cosine1)
            histogram2.append(cosine2)
            histogram3.append(cosine3)
            histogram4.append(cosine4)

        print('Finish gathering gradient cosine ...')

        hist1 = torch.cat(histogram1, dim=0)
        hist2 = torch.cat(histogram2, dim=0)
        hist3 = torch.cat(histogram3, dim=0)
        hist4 = torch.cat(histogram4, dim=0)

        print(f"pixel ref => mean={torch.mean(hist1)} \t std={torch.std(hist1)}")
        print(f"pixel skt => mean={torch.mean(hist2)} \t std={torch.std(hist2)}")
        print(f"instance ref => mean={torch.mean(hist3)} \t std={torch.std(hist3)}")
        print(f"instance skt => mean={torch.mean(hist4)} \t std={torch.std(hist4)}")

        hist1 = hist1.data.cpu().numpy()
        hist2 = hist2.data.cpu().numpy()
        hist3 = hist3.data.cpu().numpy()
        hist4 = hist4.data.cpu().numpy()

        # viz.histogram(hist1, opts=dict(numbins=200, title='a-pixel ref'))
        # viz.histogram(hist2, opts=dict(numbins=200, title='a-pixel skt'))
        # viz.histogram(hist3, opts=dict(numbins=200, title='a-instance ref'))
        # viz.histogram(hist4, opts=dict(numbins=200, title='a-instance skt'))
