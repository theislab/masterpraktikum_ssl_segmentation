# python3 train.py --lr_g 1e-4 --lr_d 5e-5 --gan_type wasserstein --n_epochs 250 --weight_decay 1e-4 --lamda3 0.5 --lamda1 1
# python3 train.py --lr_g 1e-4 --lr_d 5e-5 --gan_type wasserstein --n_epochs 250 --weight_decay 0 --lamda3 0.5 --lamda1 1 --lr_c 5e-4
from __future__ import print_function, absolute_import, division

import itertools
import logging

import pandas as pd
import torch.optim as optim
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from utils import *

use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if use_cuda else 'cpu')

logging.basicConfig(level=logging.INFO,
                    filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s: %(name)s [%(levelname)s] %(message)s')
info_string1 = ('Epoch: %3d/%3d|Batch: %2d/%2d||D_loss: %.4f|D1_loss: %.4f|'
                'D2_loss: %.4f||G_loss: %.4f|R1_loss: %.4f|R2_loss: %.4f|R121_loss: %.4f|'
                'R212_loss: %.4f')


class ViT_AE():
    """Vision Transformer Autoencoder"""

    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', output_hidden_states=True)

        # self.encoder = model.vit.encoder

    def forward(self, x):
        print("ViT_AE forward")
        print(x)

        pca_latent = PCA(n_components=512)
        latents = []

        for img in x:
            image = Image.open(img)
            input = self.processor(images=image, return_tensors="pt")
            output = self.model(**input)
            hidden_states = output.hidden_states
            print("Hidden states: ", hidden_states[-1].shape)

            latent = hidden_states[-1][0][0]
            print("Hidden states2: ", hidden_states[-1][0].shape)
            print("Hidden states3: ", hidden_states[-1][0][0].shape)
            # print(latent)
            # use pca to get same embedding shape as the cell embedding
            latents.append(latent)
        latents = pd.DataFrame(latents)
        print(latents.shape)
        # latents = latents.transpose()
        # print(latents.shape)
        principal_components_latents = pca_latent.fit_transform(latents)

        return principal_components_latents


class CellPLM_AE():
    def __init__(self, model):
        self.pipeline = CellEmbeddingPipeline(pretrain_prefix=model,  # Specify the pretrain checkpoint to load
                                              pretrain_directory='ckpt')
        self.device = DEVICE
        self.encoder = self.pipeline

    def forward(self, x):
        latent = self.encoder.predict(x,  # An AnnData object
                                      device=DEVICE)  # Specify a gpu or cpu for model inference
        return latent


class DeepAE(nn.Module):
    """DeepAE: FC AutoEncoder"""

    def __init__(self, input_dim=1, hiddens=[1], batchnorm=False):
        print(">>>init DeepAE")
        super(DeepAE, self).__init__()
        self.depth = len(hiddens)
        self.channels = [input_dim] + hiddens  # [5, 3, 3]

        encoder_layers = []
        for i in range(self.depth):
            encoder_layers.append(
                nn.Linear(self.channels[i], self.channels[i + 1]))
            if i < self.depth - 1:
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                if batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(self.channels[i + 1]))
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for i in range(self.depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self.channels[i], self.channels[i - 1]))
            decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if i > 1 and batchnorm:
                decoder_layers.append(nn.BatchNorm1d(self.channels[i - 1]))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent


class MultimodalGAN:
    def __init__(self, args, config):
        print(">>>init MultimodalGAN")
        self.args = args
        self.config = config

        self._init_logger()
        self.logger.debug('All settings used:')
        for k, v in sorted(vars(self.args).items()):
            self.logger.debug("{0}: {1}".format(k, v))
        for k, v in sorted(self.config.items()):
            self.logger.debug("{0}: {1}".format(k, v))

        assert config['img_hiddens'][-1] == config['txt_hiddens'][-1], \
            'Inconsistent latent dim!'

        # Visual transformer embedding
        self.imgAE = ViT_AE()

        # CellPLM embedding
        self.txtAE = CellPLM_AE(model='20231027_85M')

        # self._build_dataloader()
        self._build_dataloader_masterpraktikum()

        # Generator
        self.latent_dim = config['img_hiddens'][-1]
        print("latent_dim: ", self.latent_dim)
        # Autoencoders imgAE and txtAE

        # Generators img2txt and txt2img
        self.img2txt = DeepAE(input_dim=self.latent_dim,
                              hiddens=config['img2txt_hiddens'],
                              batchnorm=config['batchnorm'])
        self.txt2img = DeepAE(input_dim=self.latent_dim,
                              hiddens=config['txt2img_hiddens'],
                              batchnorm=config['batchnorm'])
        # Discriminator (modality classifier)
        self.D_img = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim / 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.latent_dim / 4), 1)
        )
        self.D_txt = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim / 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.latent_dim / 4), 1)
        )
        print(">>> Loading models done")
        # Optimizer for Generators and Discriminators

        params = [
            # {'params': itertools.chain( self.imgAE.parameters(), self.txtAE.parameters())},       not needed
            # because we don't optimize the embedders
            {'params': itertools.chain(
                self.img2txt.parameters(), self.txt2img.parameters()),
                'lr': self.args.lr_ae}]

        if self.args.gan_type == 'wasserstein':
            self.optimizer_D = optim.RMSprop(
                itertools.chain(
                    self.D_img.parameters(), self.D_txt.parameters()),
                lr=self.args.lr_d,
                weight_decay=self.args.weight_decay)
            self.optimizer_G = optim.RMSprop(
                params,
                lr=self.args.lr_g,
                weight_decay=self.args.weight_decay)
        else:
            self.optimizer_D = optim.Adam(
                itertools.chain(
                    self.D_img.parameters(), self.D_txt.parameters()),
                lr=self.args.lr_d,
                betas=(self.args.b1, self.args.b2),
                weight_decay=self.args.weight_decay)
            self.optimizer_G = optim.Adam(
                params,
                lr=self.args.lr_g, betas=(self.args.b1, self.args.b2),
                weight_decay=self.args.weight_decay)
        self.set_writer()
        self.adv_loss_fn = F.binary_cross_entropy_with_logits

    def train(self, epoch):
        # self.set_model_status(training=True)
        print("train_loader: ", self.train_loader)
        print("train_loader.dataset: ", self.train_loader.dataset)

        # for step, (ids, feats, modalitys, labels) in enumerate(self.train_loader):

        for step, (txt_embed, img_embed) in enumerate(self.train_loader):

            ids, feats, modalitys, labels = \
                ids.to(DEVICE), feats.to(DEVICE), modalitys.to(DEVICE), labels.to(DEVICE)

            modalitys = modalitys.view(-1)

            img_idx = modalitys == 0
            txt_idx = modalitys == 1

            # -----------------
            #  Train Generator
            # -----------------

            self.optimizer_G.zero_grad()

            img_feats = feats[img_idx]
            print(">img_feats ", img_feats.shape)
            txt_feats = feats[txt_idx]
            print(">txt_feats ", txt_feats.shape)
            img_batch_size = img_feats.size(0)
            txt_batch_size = txt_feats.size(0)

            imgs_latent = self.imgAE.forward(img_feats[1])
            txts_latent = self.txtAE.forward(txt_feats)
            img2txt_recon, _ = self.img2txt(imgs_latent)
            img2txt_recon, x = self.img2txt(imgs_latent)
            print(x.shape)
            img_latent_recon, _ = self.txt2img(img2txt_recon)
            # print(">img2txt_recon: ", img2txt_recon.shape)
            txt2img_recon, _ = self.txt2img(txts_latent)
            txt_latent_recon, _ = self.img2txt(txt2img_recon)

            # we only want to keep the L1 cycle losses
            img_cycle_loss = F.l1_loss(imgs_latent, img_latent_recon)
            txt_cycle_loss = F.l1_loss(txts_latent, txt_latent_recon)

            img_real = torch.ones(img_batch_size, 1).to(DEVICE)
            img_fake = torch.zeros(img_batch_size, 1).to(DEVICE)
            txt_real = torch.ones(txt_batch_size, 1).to(DEVICE)
            txt_fake = torch.zeros(txt_batch_size, 1).to(DEVICE)

            print(">>img_real: ", img_real.shape)
            print(">>img_fake: ", img_fake.shape)

            if self.args.gan_type == 'naive':
                d_loss = self.adv_loss_fn(self.D_img(txt2img_recon), txt_real) + \
                         self.adv_loss_fn(self.D_txt(img2txt_recon), img_real)
            elif 'wasserstein' in self.args.gan_type:
                d_loss = -self.D_img(txt2img_recon).mean() - \
                         self.D_txt(img2txt_recon).mean()

            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if (step + 1) % self.args.update_d_freq == 0:
                self.optimizer_D.zero_grad()

                if self.args.gan_type == 'naive':
                    img_D_loss = (self.adv_loss_fn(self.D_img(imgs_latent.detach()), img_real) +
                                  self.adv_loss_fn(self.D_img(txt2img_recon.detach()), txt_fake)) / 2
                    txt_D_loss = (self.adv_loss_fn(self.D_txt(txts_latent.detach()), txt_real) +
                                  self.adv_loss_fn(self.D_txt(img2txt_recon.detach()), img_fake)) / 2
                    D_loss = (img_D_loss + txt_D_loss) * self.args.lamda3
                elif self.args.gan_type == 'wasserstein':
                    img_D_loss = self.D_img(txt2img_recon.detach()).mean() - \
                                 self.D_img(imgs_latent.detach()).mean()
                    txt_D_loss = self.D_txt(img2txt_recon.detach()).mean() - \
                                 self.D_txt(txts_latent.detach()).mean()
                    D_loss = (img_D_loss + txt_D_loss) * self.args.lamda3
                D_loss.backward()
                self.optimizer_D.step()

                # weight clipping
                if self.args.gan_type == 'wasserstein':
                    for p in self.D_img.parameters():
                        p.data.clamp_(-self.args.clip_value,
                                      self.args.clip_value)
                    for p in self.D_txt.parameters():
                        p.data.clamp_(-self.args.clip_value,
                                      self.args.clip_value)

            if (step + 1) % self.args.log_freq == 0:
                self.logger.info(info_string1 % (
                    epoch, self.args.n_epochs, step, len(self.train_loader),
                    D_loss.item(), img_D_loss.item(), txt_D_loss.item(),
                    img_cycle_loss.item(),
                    txt_cycle_loss.item()))
                self.writer.add_scalar(
                    'Train/D_loss', D_loss.item(),
                    step + len(self.train_loader) * epoch)

        if epoch > 10 and (epoch + 1) % self.args.save_freq == 0:
            self.save_cpt(epoch)

    def embedding(self, dataloader, unify_modal='img'):
        """
        :param dataloader:
        :param unify_modal: img or txt
        :return: latent representation
        """
        print(">>> embedding")
        self.set_model_status(training=False)
        with torch.no_grad():
            latent = None
            target = None
            modality = None
            for step, (ids, feats, modalitys, labels) in enumerate(dataloader):
                batch_size = feats.shape[0]
                feats, modalitys = feats.to(DEVICE), modalitys.to(DEVICE)
                img_idx = modalitys.view(-1) == 0
                txt_idx = modalitys.view(-1) == 1
                imgs_recon, imgs_latent = self.imgAE(feats[img_idx])
                txts_recon, txts_latent = self.txtAE(feats[txt_idx])
                latent_code = torch.zeros(batch_size, self.latent_dim).to(DEVICE)
                print(">>>latent_code: ", latent_code.shape)
                if unify_modal == 'img':
                    txt2img_recon, _ = self.txt2img(txts_latent)
                    latent_code[img_idx] = imgs_latent
                    latent_code[txt_idx] = txt2img_recon
                elif unify_modal == 'txt':
                    img2txt_recon, _ = self.img2txt(imgs_latent)
                    latent_code[img_idx] = img2txt_recon
                    latent_code[txt_idx] = txts_latent
                else:
                    latent_code[img_idx] = imgs_latent
                    latent_code[txt_idx] = txts_latent
                latent = latent_code if step == 0 else torch.cat(
                    [latent, latent_code], 0)
                target = labels if step == 0 else torch.cat(
                    [target, labels], 0)
                modality = modalitys if step == 0 else torch.cat(
                    [modality, modalitys], 0)
            return latent.cpu().numpy(), target.cpu().numpy(), modality.cpu().numpy()

    def _build_dataloader(self):
        kwargs = {'num_workers': self.args.n_cpu, 'pin_memory': True}
        train_data = MFeatDataSet(
            file_mat=os.path.join(self.args.data_dir, 'train_file.mat'),
            has_filename=self.config['has_filename'])
        self.train_loader = DataLoader(dataset=train_data,
                                       batch_size=self.args.batch_size,
                                       shuffle=True, **kwargs)
        self.train_loader_ordered = DataLoader(dataset=train_data,
                                               batch_size=self.args.batch_size,
                                               shuffle=False, **kwargs)
        test_data = MFeatDataSet(
            file_mat=os.path.join(self.args.data_dir, 'test_file.mat'),
            has_filename=self.config['has_filename'])
        self.test_loader = DataLoader(dataset=test_data,
                                      batch_size=self.args.batch_size,
                                      shuffle=False, **kwargs)
        print(test_data)
        print(">>>Test data entry: ", test_data[0])
        print("Input length: ", len(test_data[0][1]))

    def _build_dataloader_masterpraktikum(self):
        kwargs = {'num_workers': self.args.n_cpu, 'pin_memory': True}

        img_path = 'data/neu'
        h5ad_path = 'data/adata.h5ad'

        # img_data as file paths
        imgs = os.listdir(img_path)
        imgs = ['data/neu/' + i for i in imgs]
        print(imgs)
        # cell_data as h5ad
        anndata = ad.read_h5ad(h5ad_path)

        # generate embeddings
        img_emb = self.imgAE.forward(imgs)
        cell_emb = self.txtAE.forward(anndata)

        train_data = CustomDataset(img_emb=img_emb, cell_emb=cell_emb)

        self.train_loader = DataLoader(dataset=train_data,
                                       batch_size=self.args.batch_size,
                                       shuffle=True, **kwargs)
        self.train_loader_ordered = DataLoader(dataset=train_data,
                                               batch_size=self.args.batch_size,
                                               shuffle=False, **kwargs)

        test_data = CustomDataset(img_emb=img_emb, cell_emb=cell_emb)
        self.test_loader = DataLoader(dataset=test_data,
                                      batch_size=self.args.batch_size,
                                      shuffle=False, **kwargs)

    def set_model_status(self, training=True):
        if training:
            self.imgAE.train()
            self.txtAE.train()
            self.img2txt.train()
            self.txt2img.train()
            self.D_img.train()
            self.D_txt.train()
        else:
            self.imgAE.eval()
            self.txtAE.eval()
            self.img2txt.eval()
            self.txt2img.eval()
            self.D_img.eval()
            self.D_txt.eval()

    def to_cuda(self):
        self.imgAE.to(DEVICE)
        self.txtAE.to(DEVICE)
        self.img2txt.to(DEVICE)
        self.txt2img.to(DEVICE)
        self.D_img.to(DEVICE)
        self.D_txt.to(DEVICE)

    def save_cpt(self, epoch):
        state_dict = {'epoch': epoch,
                      'G1_state_dict': self.imgAE.state_dict(),
                      'G2_state_dict': self.txtAE.state_dict(),
                      'G12_state_dict': self.img2txt.state_dict(),
                      'G21_state_dict': self.txt2img.state_dict(),
                      'D1_state_dict': self.D_img.state_dict(),
                      'D2_state_dict': self.D_txt.state_dict(),
                      'optimizer_G': self.optimizer_G.state_dict(),
                      'optimizer_D': self.optimizer_D.state_dict()
                      }
        cptname = '{}_checkpt_{}.pkl'.format(self.args.dataset, epoch)
        cptpath = os.path.join(self.args.cpt_dir, cptname)
        self.logger.info("> Save checkpoint '{}'".format(cptpath))
        torch.save(state_dict, cptpath)

    def load_cpt(self, cptpath):
        if os.path.isfile(cptpath):
            self.logger.info("> Load checkpoint '{}'".format(cptpath))
            dicts = torch.load(cptpath)
            self.epoch = dicts['epoch']
            self.imgAE.load_state_dict(dicts['G1_state_dict'])
            self.txtAE.load_state_dict(dicts['G2_state_dict'])
            self.img2txt.load_state_dict(dicts['G12_state_dict'])
            self.txt2img.load_state_dict(dicts['G21_state_dict'])
            self.D_img.load_state_dict(dicts['D1_state_dict'])
            self.D_txt.load_state_dict(dicts['D2_state_dict'])
            self.optimizer_G.load_state_dict(dicts['optimizer_G'])
            self.optimizer_D.load_state_dict(dicts['optimizer_D'])
            # self.scheduler.load_state_dict(dicts['scheduler'])
        else:
            self.logger.error("> No checkpoint found at '{}'".format(cptpath))

    def set_writer(self):
        self.logger.info('> Create writer at \'{}\''.format(self.args.log_dir))
        self.writer = SummaryWriter(self.args.log_dir)

    def _init_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s: %(name)s [%(levelname)s] %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S')

        file_handler = logging.FileHandler(os.path.join(
            self.args.log_dir, self.config['log_file']))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
