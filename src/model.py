# python3 train.py --lr_g 1e-4 --lr_d 5e-5 --gan_type wasserstein --n_epochs 250 --weight_decay 1e-4 --lamda3 0.5 --lamda1 1
# python3 train.py --lr_g 1e-4 --lr_d 5e-5 --gan_type wasserstein --n_epochs 250 --weight_decay 0 --lamda3 0.5 --lamda1 1 --lr_c 5e-4
from __future__ import print_function, absolute_import, division

import os
import logging
import itertools

import anndata
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from utils import h5ad_Dataset, img_Dataset, Custom_Dataloader, run_PCA

logging.basicConfig(
    level=logging.INFO,
    filename="output.log",
    datefmt="%Y/%m/%d %H:%M:%S",
    format="%(asctime)s: %(name)s [%(levelname)s] %(message)s",
)
info_string1 = (
    "Epoch: %3d/%3d|Batch: %2d/%2d||D_loss: %.4f|D1_loss: %.4f|"
    "D2_loss: %.4f||G_loss: %.4f|R121_loss: %.4f|R212_loss: %.4f"
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DeepAE(nn.Module):
    """DeepAE: FC AutoEncoder"""

    def __init__(self, input_dim=1, hiddens=[1], batchnorm=False):
        super(DeepAE, self).__init__()
        self.depth = len(hiddens)
        self.channels = [input_dim] + hiddens

        encoder_layers = []
        for i in range(self.depth):
            encoder_layers.append(nn.Linear(self.channels[i], self.channels[i + 1]))
            if i < self.depth - 1:
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                if batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(self.channels[i + 1]))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self.depth, 0, -1):
            decoder_layers.append(nn.Linear(self.channels[i], self.channels[i - 1]))
            decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if i > 1 and batchnorm:
                decoder_layers.append(nn.BatchNorm1d(self.channels[i - 1]))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent


class CellPLM_AE:
    def __init__(self, model: str):
        ckpt_directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../ckpt")
        )
        self.pipeline = CellEmbeddingPipeline(
            pretrain_prefix=model,  # specify the pretrain checkpoint to load
            pretrain_directory=ckpt_directory,
        )
        self.device = DEVICE

    def forward(self, x: anndata.AnnData):
        embedding = self.pipeline.predict(
            x, device=self.device  # x: AnnData object  # device: gpu or cpu
        )
        return embedding


class ViT_AE:
    def __init__(self, hugging_face: str):
        self.processor = ViTImageProcessor.from_pretrained(hugging_face)
        self.model = ViTForImageClassification.from_pretrained(
            hugging_face, output_hidden_states=True
        )

    def forward(self, x):
        def infer(img):
            img = Image.open(img).convert("RGB")
            inputs = self.processor(
                img, return_tensors="pt"
            )  # preprocesses for correct input format
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states
            return hidden_states[-1][0][0]

        embeds = [infer(image) for image in x]
        return torch.stack(embeds)


class MultimodalGAN:
    def __init__(self, args, config):
        print("Initializing MultimodalGAN...")
        self.args = args
        self.config = config

        self._init_logger()
        self.logger.debug("All settings used:")
        for k, v in sorted(vars(self.args).items()):
            self.logger.debug("{0}: {1}".format(k, v))
        for k, v in sorted(self.config.items()):
            self.logger.debug("{0}: {1}".format(k, v))

        # Encoders
        self.cellplm = CellPLM_AE(self.args.cellplm_model)
        self.vit = ViT_AE(self.args.hugging_face)

        self.latent_dim_img = self.config["img_latent_dim"]
        self.latent_dim_txt = self.config["txt_latent_dim"]
        self.latent_dim = min(
            self.config["img_latent_dim"], self.config["txt_latent_dim"]
        )

        print("Initializing data loaders...")
        self._build_masterpraktikum_dataloader()

        # Generators
        self.img2txt = DeepAE(
            input_dim=min(self.n_components, self.latent_dim),
            hiddens=self.config["img2txt_hiddens"],
            batchnorm=self.config["batchnorm"],
        )
        self.txt2img = DeepAE(
            input_dim=min(self.n_components, self.latent_dim),
            hiddens=self.config["txt2img_hiddens"],
            batchnorm=self.config["batchnorm"],
        )

        # Discriminators (modality classifiers)
        self.D_img = nn.Sequential(
            nn.Linear(
                min(self.n_components, self.latent_dim), int(self.latent_dim / 4)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.latent_dim / 4), 1),
        )
        self.D_txt = nn.Sequential(
            nn.Linear(
                min(self.n_components, self.latent_dim), int(self.latent_dim / 4)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.latent_dim / 4), 1),
        )

        print("Initializing optimizers...")
        # Optimizers
        params = [
            {
                "params": itertools.chain(
                    self.img2txt.parameters(), self.txt2img.parameters()
                )
            }
        ]
        if self.args.gan_type == "wasserstein":
            self.optimizer_D = optim.RMSprop(
                itertools.chain(self.D_img.parameters(), self.D_txt.parameters()),
                lr=self.args.lr_d,
                weight_decay=self.args.weight_decay,
            )
            self.optimizer_G = optim.RMSprop(
                params, lr=self.args.lr_g, weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer_D = optim.Adam(
                itertools.chain(self.D_img.parameters(), self.D_txt.parameters()),
                lr=self.args.lr_d,
                betas=(self.args.b1, self.args.b2),
                weight_decay=self.args.weight_decay,
            )
            self.optimizer_G = optim.Adam(
                params,
                lr=self.args.lr_g,
                betas=(self.args.b1, self.args.b2),
                weight_decay=self.args.weight_decay,
            )

        self.set_writer()
        self.adv_loss_fn = F.binary_cross_entropy_with_logits

    def train(self, epoch):
        self.set_model_status(training=True)
        for step, (txt_embed, img_embed) in enumerate(self.train_loader):
            # -----------------
            #  Train Generator
            # -----------------
            self.optimizer_G.zero_grad()

            txt_embed = txt_embed.to(DEVICE)
            img_embed = img_embed.to(DEVICE)

            img_batch_size = img_embed.size(0)
            txt_batch_size = txt_embed.size(0)

            img2txt_recon, _ = self.img2txt(img_embed)
            img_latent_recon, _ = self.txt2img(img2txt_recon)
            txt2img_recon, _ = self.txt2img(txt_embed)
            txt_latent_recon, _ = self.img2txt(txt2img_recon)

            img_cycle_loss = F.l1_loss(img_embed, img_latent_recon)
            txt_cycle_loss = F.l1_loss(txt_embed, txt_latent_recon)
            recon_loss = (img_cycle_loss + txt_cycle_loss) * self.args.lamda1

            img_real = torch.ones(img_batch_size, 1).to(DEVICE)
            img_fake = torch.zeros(img_batch_size, 1).to(DEVICE)
            txt_real = torch.ones(txt_batch_size, 1).to(DEVICE)
            txt_fake = torch.zeros(txt_batch_size, 1).to(DEVICE)

            if self.args.gan_type == "naive":
                d_loss = self.adv_loss_fn(
                    self.D_img(txt2img_recon), txt_real
                ) + self.adv_loss_fn(self.D_txt(img2txt_recon), img_real)
            elif self.args.gan_type == "wasserstein":
                d_loss = (
                    -self.D_img(txt2img_recon).mean() - self.D_txt(img2txt_recon).mean()
                )
            else:
                raise ValueError()
            G_loss = recon_loss + self.args.lamda3 * d_loss
            G_loss.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            if (step + 1) % self.args.update_d_freq == 0:
                self.optimizer_D.zero_grad()

                if self.args.gan_type == "naive":
                    img_D_loss = (
                        self.adv_loss_fn(self.D_img(img_embed.detach()), img_real)
                        + self.adv_loss_fn(self.D_img(txt2img_recon.detach()), txt_fake)
                    ) / 2
                    txt_D_loss = (
                        self.adv_loss_fn(self.D_txt(txt_embed.detach()), txt_real)
                        + self.adv_loss_fn(self.D_txt(img2txt_recon.detach()), img_fake)
                    ) / 2
                    D_loss = (img_D_loss + txt_D_loss) * self.args.lamda3
                elif self.args.gan_type == "wasserstein":
                    img_D_loss = (
                        self.D_img(txt2img_recon.detach()).mean()
                        - self.D_img(img_embed.detach()).mean()
                    )
                    txt_D_loss = (
                        self.D_txt(img2txt_recon.detach()).mean()
                        - self.D_txt(txt_embed.detach()).mean()
                    )
                    D_loss = (img_D_loss + txt_D_loss) * self.args.lamda3
                else:
                    raise ValueError()
                D_loss.backward()
                self.optimizer_D.step()

                # weight clipping
                if self.args.gan_type == "wasserstein":
                    for p in self.D_img.parameters():
                        p.data.clamp_(-self.args.clip_value, self.args.clip_value)
                    for p in self.D_txt.parameters():
                        p.data.clamp_(-self.args.clip_value, self.args.clip_value)

            if (step + 1) % self.args.log_freq == 0:
                self.logger.info(
                    info_string1
                    % (
                        epoch,
                        self.args.n_epochs,
                        step,
                        len(self.train_loader),
                        D_loss.item(),
                        img_D_loss.item(),
                        txt_D_loss.item(),
                        G_loss.item(),
                        img_cycle_loss.item(),
                        txt_cycle_loss.item(),
                    )
                )
                self.writer.add_scalar(
                    "Train/G_loss", G_loss.item(), step + len(self.train_loader) * epoch
                )
                self.writer.add_scalar(
                    "Train/D_loss", D_loss.item(), step + len(self.train_loader) * epoch
                )

        if epoch > 10 and (epoch + 1) % self.args.save_freq == 0:
            self.save_cpt(epoch)

    def _build_masterpraktikum_dataloader(self):
        kwargs = {
            "num_workers": self.args.n_cpu,
            "shuffle": self.args.shuffle,
            "pin_memory": True,
        }

        h5ad_dataset = h5ad_Dataset(self.args.h5ad_data)
        img_dataset = img_Dataset(self.args.img_data)

        print("Embedding txt data...")
        # embed h5ad data
        h5ad_embed = self.cellplm.forward(h5ad_dataset.data)
        print(h5ad_embed.shape)

        print("Embedding img data...")
        # embed img data batch by batch
        img_loader = DataLoader(
            dataset=img_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        img_embed = []
        for imgs in tqdm(img_loader):
            img_embed.extend(self.vit.forward(imgs))
        img_embed = torch.stack(img_embed)
        print(img_embed.shape)

        # Run PCA to ensure both modalities have the same dimensions
        if self.config["img_latent_dim"] != self.config["txt_latent_dim"]:
            print("Running PCA since text and image dimensions don't match...")
            n_samples = min(h5ad_embed.shape[0], img_embed.shape[0])
            self.n_components = self.latent_dim
            if n_samples < self.n_components:
                print("Running PCA on GEX and image embeddings...")
                self.n_components = min(n_samples, 50)
                h5ad_embed = run_PCA(h5ad_embed, self.n_components)
                img_embed = run_PCA(img_embed, self.n_components)
            elif self.n_components == self.config["img_latent_dim"]:
                print("Running PCA on GEX embeddings...")
                h5ad_embed = run_PCA(h5ad_embed, self.n_components)
            else:
                print("Running PCA on image embeddings...")
                img_embed = run_PCA(img_embed, self.n_components)

        # create a list of identifiers so we can distinguish the modalities
        modalities = [0 for embed in h5ad_embed] + [1 for embed in img_embed]

        train_data = torch.cat((h5ad_embed, img_embed), dim=0)

        self.train_loader = Custom_Dataloader(
            dataset=train_data,
            modal=modalities,
            batch_size=self.args.batch_size,
            shuffle=bool(kwargs["shuffle"]),
        )

        # TODO test_data; test_loader

    def embedding(
        self, dataloader, unify_modal="img"
    ):  # actually encodes / makes predictions
        self.set_model_status(training=False)
        with torch.no_grad():
            return_latent = None
            for step, (txt_embed, img_embed) in enumerate(dataloader):
                txt_embed = txt_embed.to(DEVICE)
                img_embed = img_embed.to(DEVICE)
                if unify_modal == "img":
                    latent, _ = self.txt2img(txt_embed)
                elif unify_modal == "txt":
                    latent, _ = self.img2txt(img_embed)
                else:
                    latent = (txt_embed, img_embed)
                return_latent = (
                    latent if step == 0 else torch.cat([return_latent, latent], 0)
                )
            return return_latent.cpu().numpy()

    def set_model_status(self, training=True):
        if training:
            self.img2txt.train()
            self.txt2img.train()
            self.D_img.train()
            self.D_txt.train()
        else:
            self.img2txt.eval()
            self.txt2img.eval()
            self.D_img.eval()
            self.D_txt.eval()

    def to_cuda(self):
        # self.cellplm.cuda()
        # self.vit.cuda()
        self.img2txt.cuda()
        self.txt2img.cuda()
        self.D_img.cuda()
        self.D_txt.cuda()

    def save_cpt(self, epoch):
        state_dict = {
            "epoch": epoch,
            "G12_state_dict": self.img2txt.state_dict(),
            "G21_state_dict": self.txt2img.state_dict(),
            "D1_state_dict": self.D_img.state_dict(),
            "D2_state_dict": self.D_txt.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
        }
        cptname = "{}_checkpt_{}.pkl".format(self.args.dataset, epoch)
        cptpath = os.path.join(self.args.cpt_dir, cptname)
        self.logger.info("> Save checkpoint '{}'".format(cptpath))
        torch.save(state_dict, cptpath)

    def load_cpt(self, cptpath):
        if os.path.isfile(cptpath):
            self.logger.info("> Load checkpoint '{}'".format(cptpath))
            dicts = torch.load(cptpath)
            self.epoch = dicts["epoch"]
            self.img2txt.load_state_dict(dicts["G12_state_dict"])
            self.txt2img.load_state_dict(dicts["G21_state_dict"])
            self.D_img.load_state_dict(dicts["D1_state_dict"])
            self.D_txt.load_state_dict(dicts["D2_state_dict"])
            self.optimizer_G.load_state_dict(dicts["optimizer_G"])
            self.optimizer_D.load_state_dict(dicts["optimizer_D"])
            # self.scheduler.load_state_dict(dicts['scheduler'])
        else:
            self.logger.error("> No checkpoint found at '{}'".format(cptpath))

    def set_writer(self):
        self.logger.info("> Create writer at '{}'".format(self.args.log_dir))
        self.writer = SummaryWriter(self.args.log_dir)

    def _init_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s: %(name)s [%(levelname)s] %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(
            os.path.join(self.args.log_dir, self.config["log_file"])
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
