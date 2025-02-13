import os, math, time

import torch
import torch as ch
import torch.nn as nn
from torch.optim import Adam, SGD

# from robustness.datasets import DATASETS

# be sure to pip install glm_saga, or clone the repo from 
# https://github.com/madrylab/glm_saga
# from glm_saga.elasticnet import glm_saga
from helpers.model_helpers import glm_saga, Proj_Model, nnzs_sparse

import helpers.data_helpers as data_helpers
import helpers.feature_helpers as feature_helpers

from argparse import ArgumentParser


ch.manual_seed(0)
# ch.set_grad_enabled(False)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ImageNet", help='dataset name')
    parser.add_argument('--dataset-type', type=str, default="vision", help='One of ["language", "vision"]')
    parser.add_argument('--dataset-path', type=str, default="dataset/imagenet", help='path to dataset')
    parser.add_argument('--model-path', type=str, help='path to model checkpoint')
    parser.add_argument('--arch', type=str, help='model architecture type')
    parser.add_argument('--out-path', default="dataset/imagenet", help='location for saving results')
    parser.add_argument('--cache', action='store_true', help='cache deep features')
    parser.add_argument('--balance', action='store_true', help='balance classes for evaluation')

    parser.add_argument('--device', default='cuda')
    # parser.add_argument('--gpu-id', default=1)
    parser.add_argument('--random-seed', default=0)
    parser.add_argument('--num-workers', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--lr-decay-factor', type=float, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--verbose', type=int, default=200)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--lookbehind', type=int, default=50)
    parser.add_argument('--lam-factor', type=float, default=0.001)
    parser.add_argument('--group', action='store_true')
    args = parser.parse_args()

    # args.device = 'cuda:' + str(args.gpu_id) if ch.cuda.is_available() else 'cpu'
    start_time = time.time()
    
    out_dir = args.out_path
    out_dir_ckpt = f'{out_dir}/checkpoint'
    out_dir_feats = f'{out_dir}/features'
    for path in [out_dir, out_dir_ckpt, out_dir_feats]:
        if not os.path.exists(path): 
            os.makedirs(path)
        
    print("Initializing dataset and loader...")
        
    dataset, train_loader, test_loader = data_helpers.load_dataset(args.dataset,
                                                                   os.path.expandvars(args.dataset_path),
                                                                   args.dataset_type,
                                                                   args.batch_size,
                                                                   args.num_workers, 
                                                                   shuffle=False,
                                                                   model_path=args.model_path)
    if args.dataset.startswith('places'):
        dataset.num_classes = 10
    num_classes = dataset.num_classes
    Ntotal = len(train_loader.dataset)
    
    print("Loading model...")
    model, pooled_output = feature_helpers.load_model("pretrain_model_imagenet/resnet50_used_from_start/checkpoint.pt.best",#args.model_path
                                                      args.arch, #  "_original"
                                                      dataset, 
                                                      args.dataset,
                                                      args.dataset_type,
                                                      device=args.device)

    print("Computing/loading deep features...")
    feature_loaders = {}
    for mode, loader in zip(['train', 'test'], [train_loader, test_loader]): 
        print(f"For {mode} set...")
        
        sink_path = f"{out_dir_feats}/features_{mode}" if args.cache else None
        metadata_path = f"{out_dir_feats}/metadata_{mode}.pth" if args.cache else None
        
        feature_ds, feature_loader = feature_helpers.compute_features(loader, 
                                                                      model, 
                                                                      dataset_type=args.dataset_type, 
                                                                      pooled_output=pooled_output,
                                                                      batch_size=args.batch_size, 
                                                                      num_workers=args.num_workers,
                                                                      shuffle=(mode == 'test'),
                                                                      device=args.device,
                                                                      filename=sink_path, 
                                                                      balance=args.balance if mode == 'test' else False)
        
        if mode == 'train':
            metadata = feature_helpers.calculate_metadata(feature_loader, 
                                          num_classes=num_classes, 
                                          filename=metadata_path)
            split_datasets, split_loaders = feature_helpers.split_dataset(feature_ds, 
                                                 Ntotal,
                                                 val_frac=args.val_frac,
                                                 batch_size=args.batch_size, 
                                                 num_workers=args.num_workers,
                                                 random_seed=args.random_seed, 
                                                 shuffle=True, balance=args.balance)
            feature_loaders.update({mm : data_helpers.add_index_to_dataloader(split_loaders[mi])
                                 for mi, mm in enumerate(['train', 'val'])})
            
        else:
            feature_loaders[mode] = feature_loader
   
    
    num_features = metadata["X"]["num_features"][0]
    assert metadata["y"]["num_classes"].numpy() == num_classes
    
    print("Initializing linear model...")


    # resume_pth = "tmp/imagenet/checkpoint/params54.pth"
    # print(f'Loading elastic net from {resume_pth}')
    # resum_params = ch.load(resume_pth)
    # linear_state_dict = linear.state_dict()
    # for name, param in linear_state_dict.items():
    #     if name in resum_params:
    #         param.copy_(torch.tensor(resum_params[name]))




    # linear_add = nn.Sequential(nn.Linear(num_features, num_features*2),
    #                            nn.ReLU(),
    #                            nn.Linear(num_features*2, num_features)) # for places-10

    proj_layer = Proj_Model(num_features, num_classes).to(args.device)
    params = proj_layer.parameters()
    # optimizer = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # optimizer = Adam(params, lr=0.001)

    print("Preparing normalization preprocess and indexed dataloader")
    preprocess = data_helpers.NormalizedRepresentation(feature_loaders['train'],
                                                       metadata=metadata,
                                                       device=args.device)





    print("Calculating the regularization path")
    params = nnzs_sparse(proj_layer,
                     feature_loaders['train'],
                     args.lr,
                     args.max_epochs,
                     args.alpha,
                     use_relu=True,
                     dataset_name=args.dataset,
                     val_loader=feature_loaders['val'],
                     test_loader=feature_loaders['test'],
                     n_classes=num_classes,
                     checkpoint=out_dir_ckpt,
                     verbose=args.verbose,
                     tol=args.tol,
                     lookbehind=args.lookbehind,
                     lr_decay_factor=args.lr_decay_factor,
                     group=args.group,
                     epsilon=args.lam_factor,
                     metadata=metadata,
                     preprocess=preprocess)

    print(f"Total time: {time.time() - start_time}")

