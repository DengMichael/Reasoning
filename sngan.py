# exponential learning rate

from scipy.stats import truncnorm
from gsa_pytorch import GSA
from adabelief_pytorch import AdaBelief
from einops import rearrange
from tqdm import tqdm
from diff_augment import DiffAugment
from kornia import filter2D
from torchvision import transforms
import torchvision
import os
import json
import multiprocessing
import random
import math
from math import log2, floor
from functools import partial
from contextlib import contextmanager, ExitStack
from pathlib import Path
from shutil import rmtree

from dreamnist_classifier_res import WideResNet, get_batch_classification_accuracy

from dreamnist import DREAMNISTDataset

import torch
from torch.nn.utils import spectral_norm
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as nnf

import lpips

NUMBERS = [str(i) for i in range(10)]
OPERATORS = ['+', '-', 'x']
EQUALS = ['=']
TRAIN_AMOUNT = 0.9


# asserts

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']
CALC_FID_NUM_IMAGES = 12800

# helpers


def exists(val):
    return val is not None


@contextmanager
def null_context():
    yield


def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts


def is_power_of_two(val):
    return log2(val).is_integer()


def default(val, d):
    return val if exists(val) else d


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(
            map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts = head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(
        zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * \
        low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def truncated_normal(size, threshold=1.5):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return torch.from_numpy(values)

# helper classes


class NanException(Exception):
    pass


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random.random() < self.prob else self.fn_else
        return fn(x)

# augmentations


def random_hflip(tensor, prob):
    if prob > random.random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob=0., types=[], detach=False, **kwargs):
        context = torch.no_grad if detach else null_context

        with context():
            if random.random() < prob:
                images = random_hflip(images, prob=0.5)
                images = DiffAugment(images, types=types)

        return self.D(images, **kwargs)

# modifiable global variables


norm_class = nn.BatchNorm2d


def upsample(scale_factor=2):
    return nn.Upsample(scale_factor=scale_factor)

# classes


class Generator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        latent_dim=256,
        w_g_G=4
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        self.latent_dim = latent_dim

        self.w_g_G = w_g_G

        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, 4, stride=2,
                               padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2,
                               padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2,
                               padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2,
                               padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 3, stride=1, padding=(1, 1), bias=False),
            nn.Sigmoi()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        leak=0.1,
        w_g_D=32
    ):
        super().__init__()

        self.leak = leak
        self.w_g_D = w_g_D

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, 3, stride=1,
                                    padding=(1, 1), bias=False)),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(64, 64, 4, stride=2,
                                    padding=(1, 1), bias=False)),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 3, stride=1,
                                    padding=(1, 1), bias=False)),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(128, 128, 4, stride=2,
                                    padding=(1, 1), bias=False)),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 3, stride=1,
                                    padding=(1, 1), bias=False)),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(256, 256, 4, stride=2,
                                    padding=(1, 1), bias=False)),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 3, stride=1,
                                    padding=(1, 1), bias=False)),
            nn.LeakyReLU(self.leak, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 3, stride=1,
                                    padding=(1, 1), bias=False)),
            nn.LeakyReLU(self.leak, inplace=True),
            nn.Sigmoid()
        )

        self.linear = spectral_norm(
            nn.Linear((self.w_g_D // 8) * (self.w_g_D // 8) * 512, 1, bias=False))

    def forward(self, x):
        print("hohohoho")
        print(x.size())
        x = self.model(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        return x


class SNGAN(nn.Module):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        optimizer="adam",
        leak=0.1,
        w_g_G=4,
        w_g_D=32,
        ttur_mult=1.,
        lr=2e-4,
        rank=0,
        ddp=False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size=image_size,
            latent_dim=latent_dim,
            w_g_G=w_g_G
        )

        self.G = Generator(**G_kwargs)

        self.D = Discriminator(
            leak=leak,
            w_g_D=w_g_D
        )
        
        self.GE = Generator(**G_kwargs)

        self.ema_updater = EMA(0.995)
        set_requires_grad(self.GE, False)

        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr=lr *
                              ttur_mult, betas=(0.5, 0.9))
        elif optimizer == "adabelief":
            self.G_opt = AdaBelief(self.G.parameters(),
                                   lr=lr, betas=(0.5, 0.9))
            self.D_opt = AdaBelief(self.D.parameters(),
                                   lr=lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(
                    old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(
                    ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented

# trainer


class Trainer():
    def __init__(
        self,
        name='default',
        results_dir='results',
        models_dir='models',
        fid_dir='fid',
        classify_dir='classify_images',
        lpips_dir='lpips_images',
        base_dir='./',
        optimizer="adam",
        latent_dim=256,
        image_size=32,
        num_image_tiles=8,
        batch_size=4,
        w_g_G=4,
        w_g_D=32,
        gp_weight=10,
        gradient_accumulate_every=1,
        lr=2e-4,
        lr_mlp=1.,
        ttur_mult=1.,
        save_every=1000,
        evaluate_every=1000,
        trunc_psi=0.6,
        aug_prob=None,
        aug_types=['translation', 'cutout'],
        dataset_aug_prob=0.,
        calculate_fid_every=None,
        classify_every=None,
        calculate_lpips_every=None,
        is_ddp=False,
        rank=0,
        world_size=1,
        log=False,
        amp=False,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = fid_dir
        self.classify_dir = classify_dir
        self.lpips_dir = lpips_dir
        self.config_path = self.models_dir / name / '.config.json'

        assert is_power_of_two(
            image_size), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'

        self.image_size = image_size
        self.num_image_tiles = num_image_tiles

        self.latent_dim = latent_dim

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.optimizer = optimizer
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.gp_weight = gp_weight

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.generator_top_k_gamma = 0.99
        self.generator_top_k_frac = 0.5

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_fid = None
        self.last_num_correct_1 = None
        self.last_num_correct_2 = None
        self.last_lpips_1 = None
        self.last_lpips_2 = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.classify_every = classify_every
        self.calculate_lpips_every = calculate_lpips_every

        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.amp = amp
        self.G_scaler = GradScaler(enabled=self.amp)
        self.D_scaler = GradScaler(enabled=self.amp)

    @property
    def image_extension(self):
        return 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    def init_GAN(self):
        args, kwargs = self.GAN_params

        # instantiate GAN

        self.GAN = SNGAN(
            optimizer=self.optimizer,
            lr=self.lr,
            latent_dim=self.latent_dim,
            image_size=self.image_size,
            ttur_mult=self.ttur_mult,
            rank=self.rank,
            *args,
            **kwargs
        )

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [
                self.rank], 'output_device': self.rank, 'find_unused_parameters': True}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists(
        ) else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.optimizer = config.pop('optimizer', 'adam')
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            'image_size': self.image_size,
            'optimizer': self.optimizer
        }

    def set_classifier(self):
        device = torch.device(f'cuda:{self.rank}')
        self.classifier = WideResNet(3, 1, 5).to(device)
        self.classifier.load_state_dict(torch.load('wideres-9919-(315).pth'))

    def set_data_src(self):
        self.dataset = DREAMNISTDataset(
            "./dataset", image_size=self.image_size, aug_prob=self.dataset_aug_prob, download=True)
        sampler = DistributedSampler(
            self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = DataLoader(self.dataset, num_workers=math.ceil(NUM_CORES / self.world_size), batch_size=math.ceil(
            self.batch_size / self.world_size), sampler=sampler, drop_last=True, pin_memory=True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(
                f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(
            self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'
        device = torch.device(f'cuda:{self.rank}')

        if not exists(self.GAN):
            self.init_GAN()

        BCE = nn.BCEWithLogitsLoss()

        self.GAN.train()
        total_disc_loss = torch.zeros([], device=device)
        total_gen_loss = torch.zeros([], device=device)

        batch_size = math.ceil(self.batch_size / self.world_size)

        latent_dim = self.GAN.latent_dim

        aug_prob = default(self.aug_prob, 0)
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        apply_gradient_penalty = self.steps % 4 == 0

        # amp related contexts and functions

        amp_context = autocast if self.amp else null_context

        # train discriminator
        self.GAN.D_opt.zero_grad()
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, G]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)
            image_batch = next(self.loader)
            image_batch = image_batch[0].cuda(self.rank)
            image_batch.requires_grad_()

            with amp_context():
                with torch.no_grad():
                    generated_images = G(latents)

                fake_output = D_aug(
                    generated_images, **aug_kwargs)

                real_output = D_aug(
                    image_batch, **aug_kwargs)

                real_output_loss = real_output
                fake_output_loss = fake_output

                disc_loss = BCE(fake_output_loss, torch.zeros_like(
                    fake_output_loss)) + BCE(real_output_loss, torch.ones_like(real_output_loss))

            if apply_gradient_penalty:
                outputs = [real_output]
                outputs = list(map(self.D_scaler.scale, outputs)
                               ) if self.amp else outputs

                scaled_gradients = torch_grad(outputs=outputs, inputs=image_batch,
                                              grad_outputs=list(map(lambda t: torch.ones(
                                                  t.size(), device=image_batch.device), outputs)),
                                              create_graph=True, retain_graph=True, only_inputs=True)[0]

                inv_scale = (1. / self.D_scaler.get_scale()
                             ) if self.amp else 1.
                gradients = scaled_gradients * inv_scale

                with amp_context():
                    gradients = scaled_gradients.reshape(batch_size, -1)
                    gp = self.gp_weight * \
                        ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    if not torch.isnan(gp):
                        disc_loss = disc_loss + gp
                        self.last_gp_loss = gp.clone().detach().item()

            with amp_context():
                disc_loss = disc_loss / self.gradient_accumulate_every

            disc_loss.register_hook(raise_if_nan)
            self.D_scaler.scale(disc_loss).backward()
            total_disc_loss += disc_loss

        self.d_loss = float(total_disc_loss.item() /
                            self.gradient_accumulate_every)
        self.D_scaler.step(self.GAN.D_opt)
        self.D_scaler.update()

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[G, D_aug]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)

            with amp_context():
                generated_images = G(latents)
                fake_output = D_aug(
                    generated_images, **aug_kwargs)
                fake_output_loss = fake_output.mean(
                    dim=1)

                epochs = (self.steps * batch_size *
                          self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma **
                             epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(
                        k=k, largest=False)

                loss = fake_output_loss.mean()
                gen_loss = loss

                gen_loss = gen_loss / self.gradient_accumulate_every

            gen_loss.register_hook(raise_if_nan)
            self.G_scaler.scale(gen_loss).backward()
            total_gen_loss += loss

        self.g_loss = float(total_gen_loss.item() /
                            self.gradient_accumulate_every)
        self.G_scaler.step(self.GAN.G_opt)
        self.G_scaler.update()

        # calculate moving averages

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(
                f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        del total_disc_loss
        del total_gen_loss

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 20000):
                self.evaluate(num=floor(self.steps / self.evaluate_every),
                              num_image_tiles=self.num_image_tiles)

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(CALC_FID_NUM_IMAGES / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

            if exists(self.classify_every) and self.steps % self.classify_every == 0 and self.steps != 0:
                num_batches = math.ceil(CALC_FID_NUM_IMAGES / self.batch_size)
                num_correct = self.classify(num_batches, num=floor(
                    self.steps / self.evaluate_every), num_image_tiles=self.num_image_tiles)
                self.last_num_correct_1 = num_correct[0]
                self.last_num_correct_2 = num_correct[1]

                with open(str(self.results_dir / self.name / f'classifier_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},default,{num_correct[0]}\n')
                    f.write(f'{self.steps},ema,{num_correct[1]}\n')

            if exists(self.calculate_lpips_every) and self.steps % self.calculate_lpips_every == 0 and self.steps != 0:
                num_batches = math.ceil(CALC_FID_NUM_IMAGES / self.batch_size)
                lpips = self.calculate_lpips(num_batches, num=floor(
                    self.steps / self.evaluate_every), num_image_tiles=self.num_image_tiles)
                self.last_lpips_1 = lpips[0]
                self.last_lpips_2 = lpips[1]

                with open(str(self.results_dir / self.name / f'lpips_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},default,{lpips[0]}\n')
                    f.write(f'{self.steps},ema,{lpips[1]}\n')

        self.steps += 1

    @torch.no_grad()
    def evaluate(self, num=0, num_image_tiles=4, trunc=1.0):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise

        latents = torch.randn((num_rows ** 2, latent_dim)).cuda(self.rank)

        # regular

        generated_images = self.generate_truncated(self.GAN.G, latents)
        torchvision.utils.save_image(generated_images, str(
            self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)

        # moving averages

        generated_images = self.generate_truncated(self.GAN.GE, latents)
        torchvision.utils.save_image(generated_images, str(
            self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def generate(self, num=0, num_image_tiles=4, checkpoint=None, types=['default', 'ema']):
        self.GAN.eval()

        latent_dim = self.GAN.latent_dim
        dir_name = self.name + str('-generated-') + str(checkpoint)
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension

        if not dir_full.exists():
            os.mkdir(dir_full)

        # regular
        if 'default' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated default images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_truncated(self.GAN.G, latents)
                path = str(self.results_dir / dir_name /
                           f'{str(num)}-{str(i)}.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        # moving averages
        if 'ema' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated EMA images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_truncated(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name /
                           f'{str(num)}-{str(i)}-ema.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        return dir_full

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = str(self.results_dir / self.name / 'fid_real') + '/'
        fake_path = str(self.results_dir / self.name / 'fid_fake') + '/'

        # remove any existing files used for fid calculation and recreate directories
        rmtree(real_path, ignore_errors=True)
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(real_path)
        os.makedirs(fake_path)

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
            real_batch = next(self.loader)
            real_batch_images = real_batch[0]
            for k in range(real_batch_images.size(0)):
                torchvision.utils.save_image(
                    real_batch_images[k, :, :, :], real_path + '{}.png'.format(k + batch_num * self.batch_size))

        # generate a bunch of fake images in results / name / fid_fake
        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = torch.randn(self.batch_size, latent_dim).cuda(self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.GE, latents)

            for j in range(generated_images.size(0)):
                torchvision.utils.save_image(generated_images[j, :, :, :], str(
                    Path(fake_path) / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([real_path, fake_path], 256, latents.device, 2048)

    @torch.no_grad()
    def classify(self, num_batches, num=0, num_image_tiles=4, trunc=1.0):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        num_correct_1 = 0
        num_correct_2 = 0

        # regular

        for batch_num in tqdm(range(num_batches), desc='classifying default images'):
            # latents and noise
            latents = torch.randn((num_rows ** 2, latent_dim)).cuda(self.rank)

            generated_images = self.generate_truncated(self.GAN.G, latents)
            generated_images = nnf.interpolate(generated_images, size=(
                32, 32), mode='bilinear', align_corners=False)
            num_correct_1 += get_batch_classification_accuracy(
                self.classifier, generated_images)

            if batch_num == 0:
                torchvision.utils.save_image(generated_images, str(
                    self.results_dir / self.classify_dir / f'classify-{str(num)}.{ext}'), nrow=num_rows)

            # moving averages

            generated_images = self.generate_truncated(self.GAN.GE, latents)
            generated_images = nnf.interpolate(generated_images, size=(
                32, 32), mode='bilinear', align_corners=False)
            num_correct_2 += get_batch_classification_accuracy(
                self.classifier, generated_images)

            if batch_num == 0:
                torchvision.utils.save_image(generated_images, str(
                    self.results_dir / self.classify_dir / f'classify-{str(num)}-ema.{ext}'), nrow=num_rows)

        total_number = num_batches * (num_rows ** 2)

        return [num_correct_1/total_number, num_correct_2/total_number]

    @ torch.no_grad()
    def calculate_lpips(self, num_batches, num=0, num_image_tiles=4, trunc=1.0):
        percept = lpips.LPIPS(net='vgg').cuda(self.rank)
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim

        lpips_1 = 0
        lpips_2 = 0
        
        lpips_dataset = DREAMNISTDataset(
            "./dataset", image_size=self.image_size, aug_prob=self.dataset_aug_prob, download=True)
        lpips_sampler = DistributedSampler(
            self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        lpips_dataloader = DataLoader(self.dataset, num_workers=math.ceil(NUM_CORES / self.world_size), batch_size=
            num_rows ** 2, sampler=lpips_sampler, drop_last=True, pin_memory=True)
        lpips_loader = cycle(lpips_dataloader)

        # regular

        for batch_num in tqdm(range(num_batches), desc='calculating lpips for default images'):
            # latents and noise
            latents = torch.randn((num_rows ** 2, latent_dim)).cuda(self.rank)

            real_batch = next(lpips_loader)
            real_batch_images = real_batch[0].cuda(self.rank)
            real_batch_images = real_batch_images * 2 - 1
            real_batch_images = real_batch_images.expand(-1, 3, -1, -1)

            generated_images = self.generate_truncated(self.GAN.G, latents)
            generated_images = generated_images * 2 - 1
            generated_images = generated_images.expand(-1, 3, -1, -1)

            print("yo")
            print(latents.size())
            print(real_batch_images.size())
            print(generated_images.size())

            rec_loss = percept.forward(generated_images, real_batch_images)
            lpips_1 += rec_loss.sum()

            if batch_num == 0:
                torchvision.utils.save_image(generated_images, str(
                    self.results_dir / self.lpips_dir / f'lpips-{str(num)}.{ext}'), nrow=num_rows)

            print(self.GAN.G.state_dict)
            # moving averages

            generated_images = self.generate_truncated(self.GAN.GE, latents)
            generated_images = generated_images * 2 - 1
            generated_images = generated_images.expand(-1, 3, -1, -1)

            print(self.GAN.GE.state_dict)
            print("hi")
            print(real_batch_images.size())
            print(generated_images.size())

            rec_loss = percept.forward(generated_images, real_batch_images)
            lpips_2 += rec_loss.sum()

            if batch_num == 0:
                torchvision.utils.save_image(generated_images, str(
                    self.results_dir / self.lpips_dir / f'lpips-{str(num)}-ema.{ext}'), nrow=num_rows)

        total_number = num_rows ** 2 * num_batches
        return [lpips_1/total_number, lpips_2/total_number]

    @ torch.no_grad()
    def generate_truncated(self, G, style, trunc_psi=0.75, num_image_tiles=8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    @ torch.no_grad()
    def generate_interpolation(self, num=0, num_image_tiles=8, trunc=1.0, num_steps=100, save_frames=False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise

        latents_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
        latents_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            generated_images = self.generate_truncated(
                self.GAN.GE, interp_latents)
            images_grid = torchvision.utils.make_grid(
                generated_images, nrow=num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())

            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name /
                           f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('FID', self.last_fid),
            ('Classifier percentage default', self.last_num_correct_1),
            ('Classifier percentage default', self.last_num_correct_2),
            ('lpips default', self.last_lpips_1),
            ('lpips ema', self.last_lpips_2),
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.results_dir / self.lpips_dir).mkdir(parents=True, exist_ok=True)
        (self.results_dir / self.classify_dir).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.results_dir / self.lpips_dir), True)
        rmtree(str(self.results_dir / self.classify_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'G_scaler': self.G_scaler.state_dict(),
            'D_scaler': self.D_scaler.state_dict()
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(
                self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(
                map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e

        if 'G_scaler' in load_data:
            self.G_scaler.load_state_dict(load_data['G_scaler'])
        if 'D_scaler' in load_data:
            self.D_scaler.load_state_dict(load_data['D_scaler'])
