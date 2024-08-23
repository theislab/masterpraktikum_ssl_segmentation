from __future__ import print_function, absolute_import, division

import os
import time
import argparse

import torch
from sklearn.cluster import KMeans

from model import MultimodalGAN
from utils import calculate_metrics, check_dir_exist
import numpy as np

METRIC_PRINT = 'metrics: ' + ', '.join(['{:.4f}'] * 7)
CPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ckpt"))
DAT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)  # 128
parser.add_argument("--lr_g", type=float, default=1e-5,  # 1e-4
                    help="adam: learning rate for G")
parser.add_argument("--lr_d", type=float, default=1e-4,  # 1e-4
                    help="adam: learning rate for D")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--lamda1", type=float, default=1.0,
                    help="reg for cycle consistency")
parser.add_argument("--lamda3", type=float, default=1.0,  # 1.0
                    help="reg for adversarial loss")
parser.add_argument("--gan_type", type=str, default='naive',
                    choices=['naive', 'wasserstein'])
parser.add_argument("--clip_value", type=float, default=0.05,
                    help="gradient clipping")

parser.add_argument("--n_cpu", type=int, default=8,
                    help="# of cpu threads during batch generation")
parser.add_argument("--shuffle", type=int, default=1)
parser.add_argument("--seed", type=int, default=2018)

parser.add_argument('--update_p_freq', type=int, default=10)
parser.add_argument('--update_d_freq', type=int, default=5)
parser.add_argument('--tol', type=int, default=1e-3)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--log_freq', type=int, default=2)
parser.add_argument('--test_freq', type=int, default=1)
#parser.add_argument('--pretrain', type=str, default='None',
#                    choices=['img', 'txt', 'load_ae', 'load_all', 'None'])
parser.add_argument('--dataset', type=str, default='masterpraktikum')
parser.add_argument('--log_dir', type=str, default=LOG_DIR)
parser.add_argument('--cpt_dir', type=str, default=CPT_DIR,
                    help='dir for saved checkpoint')
parser.add_argument('--cellplm_model', type=str, default='20230926_85M',
                    help='CellPLM ckpt')
parser.add_argument('--hugging_face', type=str, default='google/vit-base-patch16-224',
                    help='Hugging Face ViT identifier')
parser.add_argument('--h5ad_data', type=str, default=f'/p/project1/hai_pathology/embeddings/gex_embed/',  # change as needed
                    help='path to GEX data')
parser.add_argument('--img_data', type=str, default=f'/p/project1/hai_pathology/subgroup_merel/image_data/',  # change as needed
                    help='path to image data')
parser.add_argument('--test', type=str, default='None') # either 'None' or a checkpoint
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.benchmark = True

METRIC_PRINT = 'metrics: ' + ', '.join(['{:.4f}'] * 7)

if __name__ == '__main__':
    config = dict()
    if args.dataset == 'masterpraktikum':
        config['img_latent_dim'] = 768
        config['txt_latent_dim'] = 512
        # hidden dims <= data dims
        config['img2txt_hiddens'] = [512, 512]
        config['txt2img_hiddens'] = [512, 512]
        # config['has_filename'] = False
    else:
        raise ValueError()
    
    config['batchnorm'] = True
    check_dir_exist(args.log_dir)
    check_dir_exist(args.cpt_dir)

    use_cuda = torch.cuda.is_available()
    if args.test == 'None':
        current_time = time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        config['log_file'] = current_time + '.txt'
        args.cpt_dir = os.path.join(args.cpt_dir, current_time)
        args.log_dir = os.path.join(args.log_dir, current_time)
        os.mkdir(args.cpt_dir)
        os.mkdir(args.log_dir)
        model = MultimodalGAN(args, config)

        print(f"CUDA is available: {use_cuda}")
        if use_cuda:
            model.to_cuda()

        for epoch in range(args.n_epochs):
            print(epoch)
            model.train(epoch)

        orig_embedding, train_embedding = model.embedding(
            model.train_loader, unify_modal='txt')

        # TODO: save train_embedding
        np.save(os.path.join(DAT_DIR, 'train_embeds'), train_embedding)
    else: # testing initialized
        epoch = 19
        args.log_dir = os.path.join(args.log_dir, args.test) # adding a log dir
        config['log_file'] = args.test + '_testing' + '.txt'
        model = MultimodalGAN(args, config)
        print('testing...')
        model.load_cpt(os.path.join(args.cpt_dir, args.test), epoch=19)
        if use_cuda:
            model.to_cuda()
        print('model loaded, now embedding')
        orig_txt, latent_txt = model.embedding(model.test_loader_ordered, 'txt')
        orig_img, latent_img = model.embedding(model.test_loader_ordered, 'img')

        print('embeddings saved') # saving embeddings for clustering
        np.save(os.path.join(DAT_DIR, 'orig_txt'), orig_txt)
        np.save(os.path.join(DAT_DIR, 'latent_txt'), latent_txt)
        np.save(os.path.join(DAT_DIR, 'orig_img'), orig_img)
        np.save(os.path.join(DAT_DIR, 'latent_img'), latent_img)




    
    # test_embedding, test_target, test_modality = model.embedding( no test set as of now
    #    model.test_loader, unify_modal='img')
    # no need for kmeans
    # kmeans = KMeans(config['n_clusters'], max_iter=1000,
    #                tol=5e-5, n_init=20).fit(train_embedding)
    # train_metrics = calculate_metrics(train_target, kmeans.labels_)
    # y_pred = kmeans.predict(test_embedding)
    # test_metrics = calculate_metrics(test_target, y_pred)
    # print('>Train', METRIC_PRINT.format(*train_metrics))
    # print('>Test ', METRIC_PRINT.format(*test_metrics))