import os.path
import random
import numpy as np

import torch
from robustness import model_utils
import helpers.defaults_helpers as defaults
import helpers.trainer_helpers as train
from helpers.dataset_helpers import CIFAR, Places10, ImageNet, CIFAR100
import torch as ch

os.environ['CUDA_VISIBLE_DEVICES'] = '2' #'2, 3'

# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

from argparse import ArgumentParser
torch.backends.cudnn.benckmarks = True
parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default="cifar10", help='dataset name')
parser.add_argument('--seed', type=int, default=6, help='random seed')

args = parser.parse_args()
# Hard-coded dataset, architecture, batch size, workers
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def Train():
    if args.dataset == 'cifar10':
        ds = CIFAR(data_path='dataset/cifar10')
        ds.num_classes = 10
        batch_size = 128
        lr = 0.01
        arch = "resnet18wide"
        m, _ = model_utils.make_and_restore_model(arch=arch, dataset=ds, pytorch_pretrained=False)
    elif args.dataset == 'cifar100':
        ds = CIFAR100(data_path='dataset/cifar100')
        ds.num_classes = 100
        batch_size = 128
        lr = 0.1
        arch = "resnet18wide"
        m, _ = model_utils.make_and_restore_model(arch=arch, dataset=ds, pytorch_pretrained=False)
    elif args.dataset == 'places10':
        ds = Places10('dataset/places365standard_easyformat(2)/places_10')
        ds.num_classes = 10
        batch_size = 128
        arch = "resnet50"
        m, _ = model_utils.make_and_restore_model(arch=arch, dataset=ds, pytorch_pretrained=False)
    elif args.dataset == 'imagenet':
        ds = ImageNet('dataset/imagenet')
        ds.num_classes = 1000
        batch_size = 256
        arch = "resnet50"
        resume_path = None
        m, _ = model_utils.make_and_restore_model(arch=arch, dataset=ds, resume_path=resume_path, pytorch_pretrained=True) # resume_path="pretrain_model_imagenet/2c1b4424-9b0e-4265-bbf7-0c9567480938/checkpoint.pt.best"
    else:
        raise NotImplementedError


    train_loader, val_loader = ds.make_loaders(batch_size=batch_size, workers=12)

    # Create a cox store for logging
    out_store = cox.store.Store(f"pretrain_model_{args.dataset}")

    # Hard-coded base parameters
    train_kwargs = {
        'out_dir': "train_out",
        'adv_eval': 0,
        'adv_train': 0,
        'constraint': '2',
        'eps': 0.5,
        'attack_lr': 1.5,
        'attack_steps': 20,
        'epochs': 200,
        'lr': 0.0001,
        'log_iters': 5
    }
    train_args = Parameters(train_kwargs)

    # Fill whatever parameters are missing from the defaults
    if args.dataset in ['cifar', 'cifar10', 'cifar100']:
        train_args = defaults.check_and_fill_args(train_args,
                                defaults.TRAINING_ARGS, CIFAR)
        train_args = defaults.check_and_fill_args(train_args,
                                defaults.PGD_ARGS, CIFAR)
    elif args.dataset in ['imagenet',  'places10']:
        train_args = defaults.check_and_fill_args(train_args,
                                                  defaults.TRAINING_ARGS, ImageNet)
        train_args = defaults.check_and_fill_args(train_args,
                                                  defaults.PGD_ARGS, ImageNet)


    # Train a model
    if "original" in arch:
        train.train_model_original(train_args, m, (train_loader, val_loader), store=out_store)
    else:
        train.train_model(train_args, m, (train_loader, val_loader), store=out_store)

    torch.save(m.state_dict(), os.path.join(f"pretrain_model_{args.dataset}", f"{arch}.pkl"))

if __name__ == '__main__':
    seed_everything(args.seed)
    Train()