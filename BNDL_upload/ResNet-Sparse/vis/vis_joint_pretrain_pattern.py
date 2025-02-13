#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import torch as ch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from robustness.model_utils import make_and_restore_model
from robustness.tools.vis_tools import show_image_row

from helpers import data_helpers, feature_helpers, decisionlayer_helpers, vis_helpers, model_helpers
from helpers.vis_helpers import show_image_row_heatmap

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')

# In[2]:


DATASET_NAME = 'imagenet'
DATASET_PATH = '/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/DebuggableDeepNetworks/dataset/imagenet'  # Path to dataset
DATASET_TYPE = 'vision'

MODEL_PATH = "pretrain_model_imagenet/resnet50_used_from_start/checkpoint.pt.best"#"pretrain_model_imagenet/sparse_resnet50_from_start/checkpoint.pt.best"  #None # Path to deep network
ARCH = 'resnet50'  # Architecture type


BATCH_SIZE = 256
NUM_WORKERS = 10

# ## Load dataset and models

# In[3]:


dataset, train_loader, test_loader = data_helpers.load_dataset(DATASET_NAME,
                                                               os.path.expandvars(DATASET_PATH),
                                                               DATASET_TYPE,
                                                               BATCH_SIZE,
                                                               NUM_WORKERS,
                                                               shuffle=False,
                                                               model_path=MODEL_PATH)
class_dict = data_helpers.get_label_mapping(DATASET_NAME)

# In[4]:


model, _ = make_and_restore_model(arch=ARCH, dataset=dataset, resume_path=MODEL_PATH, parallel=True)
model.eval()
model.cuda()
pass

# In[5]:


test_images, test_labels = [], []
for _, (im, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
    test_images.append(im)
    test_labels.append(label)
test_images, test_labels = ch.cat(test_images), ch.cat(test_labels).numpy()

# ## Load pre-computed deep features (see main.py)

# In[6]:

FEATURE_PATH = "tmp/joint_features/imagenet_from_start/features/"#f"tmp/joint_features/{DATASET_NAME}/"
features, feature_mean, feature_std = feature_helpers.load_features_mode(FEATURE_PATH, mode='test',
                                                                         batch_size=BATCH_SIZE,
                                                                         num_workers=NUM_WORKERS)
Original_Feature_Path = "tmp/imagenet/features"
feature_o, features_mean_o, feature_std_o = feature_helpers.load_features_mode(Original_Feature_Path, mode='test',
                                                                         batch_size=BATCH_SIZE,
                                                                         num_workers=NUM_WORKERS)
# In[7]:


maximal_idx = np.argmax(features, axis=0)


CRITERION = 'absolute'
FACTOR = 5

Nplot = 5
Nfeatures = 5
FEATURE_SEL = 'rand'
MODE = 'pos'  # Visualize top-k weights in actual value ('pos') or magnitude ('all')
WT_THRESH = 1e-3  # Ignore very small weights
WT_THRESH_DENSE = 1e-3

rng = np.random.RandomState(88)

# In[12]:
weight = F.relu(model.module.model.fc.weight.data).detach().cpu()
original_m, _ = make_and_restore_model(arch="resnet50_original", dataset=dataset, pytorch_pretrained=True)
original_m.eval()
dense_weight = original_m.model.fc.weight.data.detach().cpu()
print(f'sparse weight sparsity{(weight.abs() > 1e-5).sum() / 2048000}')
print(f'dense weight sparsity{(dense_weight.abs() > 1e-5).sum() / 2048000}')
Nclasses = weight.shape[0]

for pi in range(Nplot):
    img_indices = None
    pc = rng.choice(Nclasses, 1)[0] # choices[pi]#rng.choice(choices, 1)[0] # dog (Nclasses, 1)[0]
    for used_weight, Thresh, mode in zip([weight, dense_weight], [WT_THRESH, WT_THRESH_DENSE], ['sparse', 'dense']):
        feature_indices, signs, weights, Nfs = [], [], [], []

        for weight_type in ['sparse', 'dense']:
            # Find top-k features, where k=#features used by sparse model
            weight_class = vis_helpers.normalize_weight(used_weight[pc])
            # dense_weight_class = vis_helpers.normalize_weight(dense_weight[pc])

            if MODE == 'all':
                rel_idx = np.where(np.abs(weight_class) > Thresh)[0]
                # dense_rel_idx = np.where(np.abs(dense_weight_class) > WT_THRESH)[0]
            else:
                rel_idx = np.where(weight_class > Thresh)[0] # feature indices
                # dense_rel_idx = np.where(dense_weight_class > WT_THRESH)[0]

            if weight_type == 'sparse':
                Nf = len(rel_idx)


            if MODE == 'all':
                largest_features = rel_idx[np.argsort(-np.abs(weight_class[rel_idx]))]
            else:
                largest_features = rel_idx[np.argsort(-weight_class[rel_idx])]


            # Plot Nplot randomly-chosen ones
            feature_idx = rng.choice(largest_features[: Nfeatures], Nfeatures, replace=True) # todo
            feature_idx = feature_idx[np.argsort(-weight_class[feature_idx])]


            for x, y in zip([feature_indices, signs, weights, Nfs],
                            [feature_idx, np.sign(weight_class[feature_idx]),
                             weight_class[feature_idx], len(rel_idx)]):
                x.append(y)


        feature_indices, signs = np.concatenate(feature_indices), np.concatenate(signs)

        if img_indices is None:
            # img_indices = maximal_idx[feature_indices]
            class_idx = rng.choice(np.where(test_labels == pc)[0], Nfeatures, replace=False)
            img_indices = class_idx


        inp = ch.Tensor(test_images[img_indices]).cuda(non_blocking=True)
        target = ch.Tensor(test_labels[img_indices]).cuda(non_blocking=True)
        (output, weibull_lambda, k), final_inp = model(inp, force_sample=False, target=target)
        pred_target = ch.argmax(output, dim=-1)

        if mode == "sparse":
            used_mean, used_std = feature_mean, feature_std
        elif mode == 'dense':
            used_mean, used_std = features_mean_o, feature_std_o
        else:
            raise NotImplementedError

        lime_exp, bg_mask = vis_helpers.get_lime_explanation(model,
                                                    feature_indices,
                                                    signs,
                                                    test_images[img_indices].double(),
                                                    used_mean, used_std,
                                                    NLime=10, # 10
                                                    background_color=0.1)



        print(f"---Class: {class_dict[pc]}---")

        show_image_row([test_images[class_idx]],
                       ['Image samples'])
        vis_img = test_images[img_indices]

        plt.savefig(f"vis/vis_result/sparse_dense_blurr/{class_dict[pc]}_o.svg", bbox_inches='tight')

        for idx, model_type, Nf in zip([np.arange(Nfeatures)],
                                       ['Sparse'],
                                       Nfs[::-1]):
            print(f'--{mode} model--')
            print(f'Number of features used for this class: {Nf}')
            # show_image_row([vis_img[idx]], [f'Original Image for {mode}'])
            show_image_row_heatmap([vis_img[idx]], [bg_mask[idx]],
                                   tlist=[[f"W={w:.3f} " for w in np.concatenate(weights)[idx]],  # pred class {item}
                                          ["" for _ in range(Nfeatures)]])

        plt.savefig(f"vis/vis_result/sparse_dense_blurr/{class_dict[pc]}_{mode}.svg", bbox_inches='tight')
        plt.show()
