# import waitGPU
# waitGPU.wait(nproc=0, ngpu=2)

# debugging
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch as ch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad
from torch.utils.data import random_split

from .trainer_helpers import batch_uncertain_finetune, uncertain_cal, KL_GamWei

# Toy example
import numpy as np
import time
import math
import copy
from tqdm import tqdm
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Logging
import logging
import sys
import warnings

from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from robustness.tools import helpers, constants
import dill
from robustness import attack_steps

# Helper function for getting device of a module
def get_device(module):
    if hasattr(module, 'device'):
        return module.device
    return next(module.parameters()).device


STEPS = {
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep,
    'fourier': attack_steps.FourierStep,
    'random_smooth': attack_steps.RandomStep
}

class Attacker(ch.nn.Module):
    """
    Attacker class, used to make adversarial examples.

    This is primarily an internal class, you probably want to be looking at
    :class:`robustness.attacker.AttackerModel`, which is how models are actually
    served (AttackerModel uses this Attacker class).

    However, the :meth:`robustness.Attacker.forward` function below
    documents the arguments supported for adversarial attacks specifically.
    """
    def __init__(self, model, dataset):
        """
        Initialize the Attacker

        Args:
            nn.Module model : the PyTorch model to attack
            Dataset dataset : dataset the model is trained on, only used to get mean and std for normalization
        """
        super(Attacker, self).__init__()
        self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, x, target, *_, constraint, eps, step_size, iterations,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True,
                orig_input=None, use_best=True, return_image=True,
                est_grad=None, mixed_precision=False):
        """
        Implementation of forward (finds adversarial examples). Note that
        this does **not** perform inference and should not be called
        directly; refer to :meth:`robustness.attacker.AttackerModel.forward`
        for the function you should actually be calling.

        Args:
            x, target (ch.tensor) : see :meth:`robustness.attacker.AttackerModel.forward`
            constraint
                ("2"|"inf"|"unconstrained"|"fourier"|:class:`~robustness.attack_steps.AttackerStep`)
                : threat model for adversarial attacks (:math:`\ell_2` ball,
                :math:`\ell_\infty` ball, :math:`[0, 1]^n`, Fourier basis, or
                custom AttackerStep subclass).
            eps (float) : radius for threat model.
            step_size (float) : step size for adversarial attacks.
            iterations (int): number of steps for adversarial attacks.
            random_start (bool) : if True, start the attack with a random step.
            random_restarts (bool) : if True, do many random restarts and
                take the worst attack (in terms of loss) per input.
            do_tqdm (bool) : if True, show a tqdm progress bar for the attack.
            targeted (bool) : if True (False), minimize (maximize) the loss.
            custom_loss (function|None) : if provided, used instead of the
                criterion as the loss to maximize/minimize during
                adversarial attack. The function should take in
                :samp:`model, x, target` and return a tuple of the form
                :samp:`loss, None`, where loss is a tensor of size N
                (per-element loss).
            should_normalize (bool) : If False, don't normalize the input
                (not recommended unless normalization is done in the
                custom_loss instead).
            orig_input (ch.tensor|None) : If not None, use this as the
                center of the perturbation set, rather than :samp:`x`.
            use_best (bool) : If True, use the best (in terms of loss)
                iterate of the attack process instead of just the last one.
            return_image (bool) : If True (default), then return the adversarial
                example as an image, otherwise return it in its parameterization
                (for example, the Fourier coefficients if 'constraint' is
                'fourier')
            est_grad (tuple|None) : If not None (default), then these are
                :samp:`(query_radius [R], num_queries [N])` to use for estimating the
                gradient instead of autograd. We use the spherical gradient
                estimator, shown below, along with antithetic sampling [#f1]_
                to reduce variance:
                :math:`\\nabla_x f(x) \\approx \\sum_{i=0}^N f(x + R\\cdot
                \\vec{\\delta_i})\\cdot \\vec{\\delta_i}`, where
                :math:`\delta_i` are randomly sampled from the unit ball.
            mixed_precision (bool) : if True, use mixed-precision calculations
                to compute the adversarial examples / do the inference.
        Returns:
            An adversarial example for x (i.e. within a feasible set
            determined by `eps` and `constraint`, but classified as:

            * `target` (if `targeted == True`)
            *  not `target` (if `targeted == False`)

        .. [#f1] This means that we actually draw :math:`N/2` random vectors
            from the unit ball, and then use :math:`\delta_{N/2+i} =
            -\delta_{i}`.
        """
        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class and attacker criterion
        criterion = ch.nn.CrossEntropyLoss(reduction='none')
        step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
        step = step_class(eps=eps, orig_input=orig_input, step_size=step_size)

        def calc_loss(inp, target):
            '''
            Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            '''
            if should_normalize:
                inp = self.normalize(inp)
            output = self.model(inp)
            if custom_loss:
                return custom_loss(self.model, inp, target)

            return criterion(output, target), output

        # Main function for making adversarial examples
        def get_adv_examples(x):
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x = step.random_perturb(x)

            iterator = range(iterations)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = loss.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            # PGD iterates
            for _ in iterator:
                x = x.clone().detach().requires_grad_(True)
                losses, out = calc_loss(step.to_image(x), target)
                assert losses.shape[0] == x.shape[0], \
                        'Shape of losses must match input!'

                loss = ch.mean(losses)

                if step.use_grad:
                    if (est_grad is None) and mixed_precision:
                        with amp.scale_loss(loss, []) as sl:
                            sl.backward()
                        grad = x.grad.detach()
                        x.grad.zero_()
                    elif (est_grad is None):
                        grad, = ch.autograd.grad(m * loss, [x])
                    else:
                        f = lambda _x, _y: m * calc_loss(step.to_image(_x), _y)[0]
                        grad = helpers.calc_est_grad(f, x, target, *est_grad)
                else:
                    grad = None

                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, x)

                    x = step.step(x, grad)
                    x = step.project(x)
                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))

            # Save computation (don't compute last loss) if not use_best
            if not use_best:
                ret = x.clone().detach()
                return step.to_image(ret) if return_image else ret

            losses, _ = calc_loss(step.to_image(x), target)
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args)
            return step.to_image(best_x) if return_image else best_x

        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None

            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                adv = get_adv_examples(orig_cpy)

                if to_ret is None:
                    to_ret = adv.detach()

                _, output = calc_loss(adv, target)
                corr, = helpers.accuracy(output, target, topk=(1,), exact=True)
                corr = corr.byte()
                misclass = ~corr
                to_ret[misclass] = adv[misclass]

            adv_ret = to_ret
        else:
            adv_ret = get_adv_examples(x)

        return adv_ret

class AttackerModel(ch.nn.Module):
    """
    Wrapper class for adversarial attacks on models. Given any normal
    model (a ``ch.nn.Module`` instance), wrapping it in AttackerModel allows
    for convenient access to adversarial attacks and other applications.::

        model = ResNet50()
        model = AttackerModel(model)
        x = ch.rand(10, 3, 32, 32) # random images
        y = ch.zeros(10) # label 0
        out, new_im = model(x, y, make_adv=True) # adversarial attack
        out, new_im = model(x, y, make_adv=True, targeted=True) # targeted attack
        out = model(x) # normal inference (no label needed)

    More code examples available in the documentation for `forward`.
    For a more comprehensive overview of this class, see
    :doc:`our detailed walkthrough <../example_usage/input_space_manipulation>`.
    """
    def __init__(self, model, dataset):
        super(AttackerModel, self).__init__()
        self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
        self.attacker = Attacker(model, dataset)

    def forward(self, inp, target=None, force_sample=False, top_k=-1, make_adv=False, with_latent=False,
                fake_relu=False, no_relu=False, with_image=True, **attacker_kwargs):
        """
        Main function for running inference and generating adversarial
        examples for a model.

        Parameters:
            inp (ch.tensor) : input to do inference on [N x input_shape] (e.g. NCHW)
            target (ch.tensor) : ignored if `make_adv == False`. Otherwise,
                labels for adversarial attack.
            make_adv (bool) : whether to make an adversarial example for
                the model. If true, returns a tuple of the form
                :samp:`(model_prediction, adv_input)` where
                :samp:`model_prediction` is a tensor with the *logits* from
                the network.
            with_latent (bool) : also return the second-last layer along
                with the logits. Output becomes of the form
                :samp:`((model_logits, model_layer), adv_input)` if
                :samp:`make_adv==True`, otherwise :samp:`(model_logits, model_layer)`.
            fake_relu (bool) : useful for activation maximization. If
                :samp:`True`, replace the ReLUs in the last layer with
                "fake ReLUs," which are ReLUs in the forwards pass but
                identity in the backwards pass (otherwise, maximizing a
                ReLU which is dead is impossible as there is no gradient).
            no_relu (bool) : If :samp:`True`, return the latent output with
                the (pre-ReLU) output of the second-last layer, instead of the
                post-ReLU output. Requires :samp:`fake_relu=False`, and has no
                visible effect without :samp:`with_latent=True`.
            with_image (bool) : if :samp:`False`, only return the model output
                (even if :samp:`make_adv == True`).

        """
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            if prev_training:
                self.train()

            inp = adv

        normalized_inp = self.normalizer(inp)

        if no_relu and (not with_latent):
            print("WARNING: 'no_relu' has no visible effect if 'with_latent is False.")
        if no_relu and fake_relu:
            raise ValueError("Options 'no_relu' and 'fake_relu' are exclusive")

        output = self.model(normalized_inp, force_sample=force_sample, top_k=top_k, with_latent=with_latent,
                                fake_relu=fake_relu, no_relu=no_relu)
        if with_image:
            return (output, inp)
        return output

class DummyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        return self.model(x)


def make_and_restore_model(*_, arch, dataset, resume_path=None,
                           parallel=False, pytorch_pretrained=False, add_custom_forward=False):
    """
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint saved with the
            robustness library (ignored if ``arch`` is not a string)
        not a string
        parallel (bool): if True, wrap the model in a DataParallel
            (defaults to False)
        pytorch_pretrained (bool): if True, try to load a standard-trained
            checkpoint from the torchvision library (throw error if failed)
        add_custom_forward (bool): ignored unless arch is an instance of
            nn.Module (and not a string). Normally, architectures should have a
            forward() function which accepts arguments ``with_latent``,
            ``fake_relu``, and ``no_relu`` to allow for adversarial manipulation
            (see `here`<https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures>
            for more info). If this argument is True, then these options will
            not be passed to forward(). (Useful if you just want to train a
            model and don't care about these arguments, and are passing in an
            arch that you don't want to edit forward() for, e.g.  a pretrained model)
    Returns:
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    if (not isinstance(arch, str)) and add_custom_forward:
        arch = DummyModel(arch)

    classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
        isinstance(arch, str) else arch

    model = AttackerModel(classifier_model, dataset)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = ch.load(resume_path, pickle_module=dill)

        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    if parallel:
        model = ch.nn.DataParallel(model)
    model = model.cuda()

    return model, checkpoint



#######
# Solver assumes standardized input
class IndexedTensorDataset(TensorDataset):
    def __getitem__(self, index):
        val = super(IndexedTensorDataset, self).__getitem__(index)
        return val + (index,)


class IndexedDataset(Dataset):
    def __init__(self, ds, sample_weight=None):
        super(Dataset, self).__init__()
        self.dataset = ds
        self.sample_weight = sample_weight

    def __getitem__(self, index):
        val = self.dataset[index]
        if self.sample_weight is None:
            return val + (index,)
        else:
            weight = self.sample_weight[index]
            return val + (weight, index)

    def __len__(self):
        return len(self.dataset)


class Proj_Model(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Proj_Model, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.linear_k = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Softplus()
        )
        self.linear_kw = nn.Sequential(
            nn.Linear(num_classes, 1),
            nn.Softplus()
        )

        self.linear_add = nn.Sequential(nn.Linear(num_features, num_features))  # for imagenet, cifar-10
        # self.linear_add = nn.Sequential(nn.Linear(num_features, num_features*2),
        #                            nn.ReLU(),
        #                            nn.Linear(num_features*2, num_features)) # for places-10
        self.linear = nn.Linear(num_features, num_classes)

    def reparameterize(self, lbd, kappa, force_sample=False):
        '''
            weibull reparameterization: z = lbd * (- ln(1 - u)) ^ (1/kappa), u ~ uniform(0,1)
            z: node-community affiliation.
            lbd: scale parameter, kappa: shape parameter
        '''

        def log_max(input, SMALL=1e-10):
            device = input.device
            input_ = torch.max(input, torch.tensor([SMALL]).to(device))
            return torch.log(input_)

        # print(force_sample)
        if self.training or force_sample:
            u = torch.rand_like(lbd)
            z = lbd * (- log_max(1 - u)).pow(1 / kappa)
            # print('force sample')
        else:
            z = lbd * torch.exp(torch.lgamma(1 + kappa.pow(-1)))
            # print('fixed')
        return z

    def forward(self, X, factor_z=0, factor_w=0, force_sample=False):
        z = self.linear_add(X)

        # non-negative z
        gelu_z = F.relu(z - factor_z)
        z_out = gelu_z - gelu_z.data + F.relu(z - factor_z).data

        # reparameterize z
        k = self.linear_k(z_out) #torch.ones_like(z_out, requires_grad=False, device=z_out.device) * 10
        # print(f'k_min {k.min()}\t k_max {k.max()}')
        weibull_lambda = z_out / torch.exp(torch.lgamma(1 + 1 / k))
        pre_out = self.reparameterize(weibull_lambda, k, force_sample)  # N * H

        # reparameterize w
        gelu_z_w = F.relu(self.linear.weight.transpose(1, 0) - factor_w)
        z_out_w = gelu_z_w - gelu_z_w.data + F.relu(self.linear.weight.transpose(1, 0) - factor_w).data
        # z_out_w = F.softmax(self.linear.weight.transpose(1, 0), dim=-1)
        k_w = self.linear_kw(z_out_w) #torch.ones_like(z_out_w, requires_grad=False, device=z_out.device) * 100000
        # print(f'k_w min {k_w.min()}\n k_w max {k_w.max()}')
        weibull_lambda_w = z_out_w / torch.exp(torch.lgamma(1 + 1 / k_w))
        pre_out_w = self.reparameterize(weibull_lambda_w, k_w, force_sample)  # H * C

        # pre_out = F.normalize(pre_out, dim=-1)
        out = torch.mm(pre_out, pre_out_w) + F.relu(self.linear.bias - factor_w)
        return out, z_out, weibull_lambda, 1/k, weibull_lambda_w, 1/k_w


# create a new dataloader which returns example indices
def add_index_to_dataloader(loader, sample_weight=None):
    return DataLoader(
        IndexedDataset(loader.dataset, sample_weight=sample_weight),
        batch_size=loader.batch_size,
        sampler=loader.sampler,
        # batch_sampler=loader.batch_sampler, 
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context
        # generator=loader.generator
    )


# L1 regularization
# proximal operator for f(\beta) = lam * \|\beta\|_1
def soft_threshold(beta, lam):
    return (beta - lam) * (beta > lam) + (beta + lam) * (beta < -lam)


# Grouped L1 regularization
# proximal operator for f(weight) = lam * \|weight\|_2 
# where the 2-norm is taken columnwise
def group_threshold(weight, lam):
    norm = weight.norm(p=2, dim=0)
    return (weight - lam * weight / norm) * (norm > lam)


# Elastic net regularization
# proximal operator for f(x) = alpha * \|x\|_1 + beta * \|x\|_2^2
def soft_threshold_with_shrinkage(x, alpha, beta):
    y = soft_threshold(x, alpha)
    return y / (1 + beta)


# Elastic net regularization with group sparsity
# proximal operator for f(x) = alpha * \|x\|_1 + beta * \|x\|_2^2
# where the 2-norm is taken columnwise
def group_threshold_with_shrinkage(x, alpha, beta):
    y = group_threshold(x, alpha)
    return y / (1 + beta)


# Elastic net loss
def elastic_loss(linear, X, y, lam, alpha, family='multinomial', sample_weight=None):
    weight, bias = list(linear.parameters())
    l1 = lam * alpha * weight.norm(p=1)
    l2 = 0.5 * lam * (1 - alpha) * (weight ** 2).sum()
    if family == 'multinomial':
        if sample_weight is None:
            l = F.cross_entropy(linear(X), y, reduction='mean')
        else:
            l = F.cross_entropy(linear(X), y, reduction='none')
            l = (l * sample_weight).mean()
    elif family == 'gaussian':
        # For some reason, PyTorch mse_loss doesn't take a weight argument
        if sample_weight is None:
            l = 0.5 * F.mse_loss(linear(X), y, reduction='mean')
        else:
            l = 0.5 * F.mse_loss(linear(X), y, reduction='none')
            l = (l * (sample_weight.unsqueeze(1))).mean()
    else:
        raise ValueError(f"Unknown family: {family}")
    return l + l1 + l2


# Elastic net loss given a loader instead
def elastic_loss_loader(linear, loader, lam, alpha, preprocess=None, family='multinomial'):
    loss = 0
    n = 0
    device = linear.weight.device
    if preprocess is not None:
        preprocess_device = get_device(preprocess)
    for batch in loader:
        X, y = batch[0].to(device), batch[1].to(device)
        if preprocess is not None:
            X = preprocess(X)
        bs = X.size(0)
        loss += elastic_loss(linear, X, y, lam, alpha, family=family) * bs
        n += bs
    return loss / n


# Elastic net loss and accuracy
def elastic_loss_and_acc(proj_layer, X, y, factor_z, factor_w, accurate_pred, testresult, uncertain_sample, family='multinomial'):
    # weight, bias = list(linear.parameters())
    # l1 = lam * alpha * weight.norm(p=1)
    # l2 = 0.5 * lam * (1 - alpha) * (weight ** 2).sum()

    # if use_relu:
    # X = F.relu(X)
    # X = F.normalize(X, dim=-1)    # N * D
    # _w = F.relu(linear.weight)
    # _w = F.normalize(_w, dim=-1)  # C * D
    # outputs = torch.mm(X, _w.transpose(1, 0)) + F.normalize(F.relu(linear.bias), dim=-1)

    # outputs = linear(X)

    # outputs, pre_out, _, _, _, _ = proj_layer(X, factor_z, factor_w)

    outputs, pre_out, weibull_lambda, k, weibull_lambda_w, k_w = proj_layer(X, factor_z, factor_w)

    mean_logits = 0
    if uncertain_sample:
        testresult, mean_logits, accurate_pred = batch_uncertain_finetune(outputs, factor_z, factor_w, proj_layer.num_classes, y, X, proj_layer,
                                                                 accurate_pred, testresult)

    if family == 'multinomial':
        l = F.cross_entropy(outputs, y, reduction='mean')
        acc = (outputs.max(1)[1] == y).float().mean()
    elif family == 'gaussian':
        l = 0.5 * F.mse_loss(outputs, y, reduction='mean')
        acc = (outputs == y).float().mean()
    else:
        raise ValueError(f"Unknown family {family}")

    loss = l #+ l1 + l2
    return loss, acc, testresult, mean_logits, accurate_pred, weibull_lambda, k, weibull_lambda_w, k_w


# Elastic net loss given a loader instead
def elastic_loss_and_acc_loader(proj_layer, loader, factor_z, factor_w, uncertain_sample, preprocess=None, family='multinomial'):
    loss, KL_loss = 0, 0
    acc = 0
    n = 0
    device = get_device(proj_layer)

    accurate_pred = torch.zeros([0], dtype=torch.float64).cuda(non_blocking=True)
    testresult = torch.zeros([0], dtype=torch.float64).cuda(non_blocking=True)

    for batch in loader:
        X = batch[0].to(device)
        y = batch[1].to(device)
        if preprocess is not None:
            X = preprocess(X)
        bs = X.size(0)
        l, a, testresult, mean_logits, accurate_pred, weibull_lambda, k, weibull_lambda_w, k_w = elastic_loss_and_acc(proj_layer, X, y, factor_z, factor_w, accurate_pred, testresult, uncertain_sample, family=family)
        loss += l * bs
        acc += a * bs
        n += bs

        gamma_shape = ch.tensor(1.0, dtype=ch.float32, requires_grad=False).cuda(non_blocking=True)
        gamma_scale = ch.tensor(1.0, dtype=ch.float32, requires_grad=False).cuda(non_blocking=True)
        KL = KL_GamWei(gamma_shape, gamma_scale, k, weibull_lambda)
        KL_w = KL_GamWei(gamma_shape, gamma_scale, k_w, weibull_lambda_w)
        KL_loss += 1e-8 * (KL + KL_w)

    pavpus = [0., 0., 0.]
    if uncertain_sample:
        pavpus = uncertain_cal(testresult, mean_logits, accurate_pred)
        print(f'pavpus: {pavpus[0]:.4f}\t {pavpus[1]:.4f}\t {pavpus[2]:.4f}\t')

    return loss / n + KL_loss, acc / n, pavpus


# Train an elastic GLM with proximal gradient as a baseline
def train(linear, X, y, lr, niters, lam, alpha, group=True, verbose=None):
    weight, bias = list(linear.parameters())

    opt = SGD(linear.parameters(), lr=lr)
    for i in range(niters):
        with ch.enable_grad():
            out = linear(X)
            loss = F.cross_entropy(out, y, reduction='mean') + 0.5 * lam * (1 - alpha) * (weight ** 2).sum()
            if verbose and (i % verbose) == 0:
                print(loss.item())

            # gradient step
            opt.zero_grad()
            loss.backward()
            opt.step()

        # proximal step
        if group:
            weight.data = group_threshold(weight, lr * lam * alpha)
        else:
            weight.data = soft_threshold(weight, lr * lam * alpha)


# Train an elastic GLM with stochastic proximal gradient as an even more inaccurate baseline
def train_spg(linear, loader, max_lr, nepochs, lam, alpha, preprocess=None, min_lr=1e-4, group=True, verbose=None):
    weight, bias = list(linear.parameters())

    params = [weight, bias]
    proximal = [True, False]

    device = linear.weight.device

    lrs = ch.logspace(math.log10(max_lr), math.log10(min_lr), nepochs).to(device)

    for t in range(nepochs):
        lr = lrs[t]
        total_loss = 0
        n_ex = 0
        for X, y, idx in loader:
            X, y = X.to(device), y.to(device)
            if preprocess is not None:
                with ch.no_grad():
                    X = preprocess(X)
            with ch.enable_grad():
                out = linear(X)
                # rescaling = X.size(0) / n_ex
                loss = F.cross_entropy(out, y, reduction='mean') + 0.5 * lam * (1 - alpha) * (weight ** 2).sum()
                # loss = F.cross_entropy(out,y, reduction='sum') + 0.5 * lam * (1 - alpha) * (weight**2).sum()
                # print(out.requires_grad, linear.weight.requires_grad)
                loss.backward()

            with ch.no_grad():
                total_loss += loss.item() * X.size(0)
                n_ex += X.size(0)
                for p, prox in zip(params, proximal):
                    # grad = p.grad / X.size(0) * n_ex
                    grad = p.grad

                    # take a step
                    p.data = p.data - lr * grad
                    if prox:
                        if group:
                            p.data = group_threshold(p, lr * lam * alpha)
                        else:
                            p.data = soft_threshold(p, lr * lam * alpha)

            # clean up
            weight.grad.zero_()
            bias.grad.zero_()

        if verbose and (t % verbose) == 0:
            spg_obj = (total_loss / n_ex + lam * alpha * weight.norm(p=1)).item()
            nnz = (weight.abs() > 1e-5).sum().item()
            total = weight.numel()
            print(f"obj {spg_obj} weight nnz {nnz}/{total} ({nnz / total:.4f}) ")
            # print(f"obj {spg_obj} weight nnz {nnz}/{total} ({nnz/total:.4f}) criteria {criteria:.4f} {dw} {db}")


# Train an elastic GLM with proximal SAGA 
# Since SAGA stores a scalar for each example-class pair, either pass 
# the number of examples and number of classes or calculate it with an 
# initial pass over the loaders
def train_saga(proj_layer, opt, loader, test_loader, nepochs, use_relu=True, group=True, verbose=None,
               state=None, table_device=None, n_ex=None, n_classes=None, tol=1e-4,
               preprocess=None, lookbehind=None, family='multinomial', logger=None, ds_name=None):
    if logger is None:
        logger = print

    if ds_name == "places-10":
        ds_name = "places"

    # get total number of examples and initialize scalars
    # for computing the gradients
    if n_ex is None:
        n_ex = sum(tensors[0].size(0) for tensors in loader)
    if n_classes is None:
        if family == 'multinomial':
            n_classes = max(tensors[1].max().item() for tensors in loader) + 1
        elif family == 'gaussian':
            for batch in loader:
                y = batch[1]
                break
            n_classes = y.size(1)

    model_device = get_device(proj_layer)
    obj_best = None
    best_acc_test = -1

    for t in range(nepochs):
        total_loss = 0
        uncertain_sample = True if (t + 1) % 10 == 0 or t == nepochs - 1 else False
        for batch in loader:
            if len(batch) == 3:
                X, y, idx = batch
                w = None
            elif len(batch) == 4:
                X, y, w, idx = batch
            else:
                raise ValueError(
                    f"Loader must return (data, target, index) or (data, target, index, weight) but instead got a tuple of length {len(batch)}")

            if preprocess is not None:
                device = get_device(preprocess)
                with ch.no_grad():
                    X = preprocess(X.to(device))

            X = X.to(model_device)
            proj_layer.train()

            factor_w, factor_z = 0.010, -1 # 0.022091  0.0210
            out, pre_out, weibull_lambda, k, weibull_lambda_w, k_w = proj_layer(X, factor_z, factor_w)

            gamma_shape = ch.tensor(1.0, dtype=ch.float32, requires_grad=False).cuda(non_blocking=True)
            gamma_scale = ch.tensor(1.0, dtype=ch.float32, requires_grad=False).cuda(non_blocking=True)
            KL = KL_GamWei(gamma_shape, gamma_scale, k, weibull_lambda)
            KL_w = KL_GamWei(gamma_shape, gamma_scale, k_w, weibull_lambda_w)

            loss = F.cross_entropy(out, y.to(model_device), reduction='mean')
            total_loss += loss * X.size(0)
            loss += 1e-6 * (KL + KL_w) # 0.00001

            opt.zero_grad()
            loss.backward()
            opt.step()


        nnz = (F.relu(proj_layer.linear.weight - factor_w).abs() > 1e-5).sum().item()
        total = proj_layer.linear.weight.numel()
        h_nnz = (pre_out.abs() > 1e-5).sum().item()
        h_total = pre_out.numel()
        saga_obj = total_loss / n_ex

        logger(
            f"Epoch {t}: obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total}) final_hidden nnz {h_nnz}/{h_total} ({h_nnz / h_total:.4f}) obj_best {obj_best}")

        if test_loader:
            proj_layer.eval()
            with torch.no_grad():
                loss_test, acc_test, pavpus = elastic_loss_and_acc_loader(proj_layer, test_loader, factor_z, factor_w, uncertain_sample,
                                                                  preprocess=preprocess, family=family)
            loss_test, acc_test = loss_test.item(), acc_test.item()
            logger(f"Epoch {t}: loss test {loss_test: .4f}\t acc test {acc_test:.4f}\t "
                  f"pavpus(0.01) {pavpus[0]:.4f}\t pavpus(0.05) {pavpus[1]:.4f}\t pavpus(0.1) {pavpus[2]:.4f}\t")

            if best_acc_test < acc_test:
                best_acc_test = acc_test
                ch.save(proj_layer.state_dict(), f"tmp/{ds_name}/linear_ckpt/best_linear.pth")
        # if (t + 1) % 10 == 0:
        # saga_obj = total_loss / n_ex # + lam * alpha * weight.norm(p=1) + 0.5 * lam * (1 - alpha) * (weight ** 2).sum()
        # # save amount of improvement
        # obj_history.append(saga_obj.item())
        # if obj_best is None or saga_obj.item() + tol < obj_best:
        #     obj_best = saga_obj.item()
        #     nni = 0
        # else:
        #     nni += 1
        #
        # # Stop if no progress for lookbehind iterationsd:])
        # criteria = lookbehind is not None and (nni >= lookbehind)
        #
        # nnz = (F.relu(proj_layer.linear.weight).abs() > 1e-5).sum().item()
        # total = proj_layer.linear.weight.numel()
        # h_nnz = (pre_out.abs() > 1e-5).sum().item()
        # h_total = pre_out.numel()
        # if verbose and (t % verbose) == 0:
        #     if lookbehind is None:
        #         logger(
        #             f"obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total:.4f}) criteria {criteria:.4f} {dw} {db}")
        #     else:
        #         logger(
        #             f"obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total:.4f}) final_hidden nnz {h_nnz}/{h_total} ({h_nnz / h_total:.4f}) obj_best {obj_best}")
        #

    return proj_layer# torch.cat(total_out, dim=0)



def train_saga_early_stop(proj_layer, opt, loader, test_loader, nepochs, factor_w, factor_z, use_relu=True, group=True, verbose=None,
               state=None, table_device=None, n_ex=None, n_classes=None, tol=1e-4,
               preprocess=None, lookbehind=None, family='multinomial', logger=None, ds_name=None):
    if logger is None:
        logger = print



    # get total number of examples and initialize scalars
    # for computing the gradients
    if n_ex is None:
        n_ex = sum(tensors[0].size(0) for tensors in loader)
    if n_classes is None:
        if family == 'multinomial':
            n_classes = max(tensors[1].max().item() for tensors in loader) + 1
        elif family == 'gaussian':
            for batch in loader:
                y = batch[1]
                break
            n_classes = y.size(1)

    model_device = get_device(proj_layer)
    obj_best = None
    best_acc_test = -1
    obj_history = []
    return_model = None

    for t in range(nepochs):
        total_loss, KL_loss = 0, 0
        uncertain_sample = True if (t + 1) % 10 == 0 or t == nepochs - 1 else False
        for batch in loader:
            if len(batch) == 3:
                X, y, idx = batch
                w = None
            elif len(batch) == 4:
                X, y, w, idx = batch
            else:
                raise ValueError(
                    f"Loader must return (data, target, index) or (data, target, index, weight) but instead got a tuple of length {len(batch)}")

            if preprocess is not None:
                device = get_device(preprocess)
                with ch.no_grad():
                    X = preprocess(X.to(device))

            X = X.to(model_device)
            proj_layer.train()

            out, pre_out, weibull_lambda, k, weibull_lambda_w, k_w = proj_layer(X, factor_z, factor_w)
            gamma_shape = ch.tensor(1.0, dtype=ch.float32, requires_grad=False).cuda(non_blocking=True)
            gamma_scale = ch.tensor(1.0, dtype=ch.float32, requires_grad=False).cuda(non_blocking=True)
            KL = KL_GamWei(gamma_shape, gamma_scale, k, weibull_lambda)
            KL_w = KL_GamWei(gamma_shape, gamma_scale, k_w, weibull_lambda_w)


            loss = F.cross_entropy(out, y.to(model_device), reduction='mean')
            total_loss += loss * X.size(0)
            KL_loss += 1e-8 * (KL + KL_w)
            loss += 1e-8 * (KL + KL_w) # 0.00001

            opt.zero_grad()
            loss.backward()
            opt.step()


        nnz = (F.relu(proj_layer.linear.weight - factor_w).abs() > 1e-5).sum().item()
        total = proj_layer.linear.weight.numel()
        h_nnz = (pre_out.abs() > 1e-5).sum().item()
        h_total = pre_out.numel()
        saga_obj = total_loss / n_ex + KL_loss

        print(
            f"Epoch {t}: obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total}) final_hidden nnz {h_nnz}/{h_total} ({h_nnz / h_total:.4f}) obj_best {obj_best}")

        if test_loader:
            proj_layer.eval()
            with torch.no_grad():
                loss_test, acc_test, pavpus = elastic_loss_and_acc_loader(proj_layer, test_loader, factor_z, factor_w, uncertain_sample,
                                                                  preprocess=preprocess, family=family)
            loss_test, acc_test = loss_test.item(), acc_test.item()
            print(f"Epoch {t}: loss test {loss_test: .4f}\t acc test {acc_test:.4f}\t "
                  f"pavpus(0.01) {pavpus[0]:.4f}\t pavpus(0.05) {pavpus[1]:.4f}\t pavpus(0.1) {pavpus[2]:.4f}\t")

            if best_acc_test < acc_test:
                best_acc_test = acc_test
                return_model = copy.deepcopy(proj_layer)
                ch.save(proj_layer.state_dict(), f"tmp/{ds_name}/checkpoint/best_linear.pth")


        obj_history.append(loss_test)
        if obj_best is None or saga_obj.item() + tol < obj_best:
            obj_best = saga_obj.item()
            nni = 0
        else:
            nni += 1

        # Stop if no progress for lookbehind iterationsd:])
        criteria = lookbehind is not None and (nni >= lookbehind)

        nnz = (F.relu(proj_layer.linear.weight - factor_w).abs() > 1e-5).sum().item()
        total = proj_layer.linear.weight.numel()
        h_nnz = (F.relu(pre_out - factor_z).abs() > 1e-5).sum().item()
        h_total = pre_out.numel()
        if verbose and (t % verbose) == 0:
            if lookbehind is None:
                logger(
                    f"obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total:.4f}) criteria {criteria:.4f} {dw} {db}")
            else:
                logger(
                    f"obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total:.4f}) final_hidden nnz {h_nnz}/{h_total} ({h_nnz / h_total:.4f}) obj_best {obj_best}")

        if lookbehind is not None and criteria:
            logger(
                f"obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total:.4f}) obj_best {obj_best} [early stop at {t}]")
            return proj_layer, acc_test #return_model, best_acc_test

    logger(f"did not converge at {nepochs} iterations (criteria {criteria})")
    return proj_layer, acc_test #return_model, best_acc_test# torch.cat(total_out, dim=0)

# Calculate the smallest regularization parameter which results in a
# linear model with all zero weights. Calculation comes from the 
# coordinate descent iteration. 
def maximum_reg(X, y, group=True, family='multinomial'):
    if family == 'multinomial':
        target = ch.eye(y.max() + 1)[y].to(y.device)
    elif family == 'gaussian':
        target = y
    else:
        raise ValueError(f"Unknown family {family}")

    y_bar = target.mean(0)
    y_std = target.std(0)

    y_map = (target - y_bar)

    inner_products = X.t().mm(y_map)

    if group:
        inner_products = inner_products.norm(p=2, dim=1)
    return inner_products.abs().max().item() / X.size(0)


# Same as before, but for a loader instead
def maximum_reg_loader(loader, group=True, preprocess=None, metadata=None, family='multinomial'):
    if metadata is not None:
        return metadata['max_reg']['group'] if group else metadata['max_reg']['nongrouped']

    print("Calculating maximum regularization from dataloader...")
    # calculate number of classes
    y_max = 1
    for batch in loader:
        y = batch[1]
        y_max = max(y_max, y.max().item() + 1)

    if family == 'multinomial':
        eye = ch.eye(y_max).to(y.device)

    y_bar = 0
    n = 0

    # calculate mean
    for batch in loader:
        y = batch[1]

        if family == 'multinomial':
            target = eye[y]
        elif family == 'gaussian':
            target = y
        else:
            raise ValueError(f"Unknown family {family}")

        y_bar += target.sum(0)
        n += y.size(0)
    y_bar = y_bar.float() / n

    # calculate std
    y_std = 0
    for batch in loader:
        y = batch[1]

        if family == 'multinomial':
            target = eye[y]
        elif family == 'gaussian':
            target = y
        else:
            raise ValueError(f"Unknown family {family}")

        y_std += ((target - y_bar) ** 2).sum(0)
    y_std = ch.sqrt(y_std.float() / (n - 1))

    # calculate maximum regularization
    inner_products = 0
    if preprocess is not None:
        device = get_device(preprocess)
    else:
        device = y.device
    for batch in loader:
        X, y = batch[0], batch[1]

        if family == 'multinomial':
            target = eye[y]
        elif family == 'gaussian':
            target = y
        else:
            raise ValueError(f"Unknown family {family}")

        y_map = (target - y_bar)

        if preprocess is not None:
            X = preprocess(X.to(device))
            y_map = y_map.to(device)
            y_std = y_std.to(device)
        inner_products += X.t().mm(y_map)

    if group:
        inner_products = inner_products.norm(p=2, dim=1)
    return inner_products.abs().max().item() / n


def calculate_sparsity(hidden, weight):
    '''
    hidden: N * H
    weight: H * C
    '''
    ids = torch.where(hidden > 1e-5)
    w_ids = torch.unique(ids[1])
    nnz = (weight[w_ids, :] > 1e-5).sum().item()
    total = weight.numel()

    return nnz / total

def nnzs_sparse(proj_layer, loader, max_lr, nepochs, alpha, use_relu, dataset_name,
             table_device=None, preprocess=None, group=False,
             verbose=None, state=None, n_ex=None, n_classes=None,
             tol=1e-4, epsilon=0.001, k=100, checkpoint=None,
             do_zero=True, lr_decay_factor=1, metadata=None,
             val_loader=None, test_loader=None, lookbehind=None,
             family='multinomial', encoder=None):
    if encoder is not None:
        warnings.warn("encoder argument is deprecated; please use preprocess instead", DeprecationWarning)
        preprocess = encoder

    # if preprocess is not None and (get_device(proj_layer) != get_device(preprocess)):
    #     raise ValueError(
    #         "Linear and preprocess must be on same device (got {get_device(linear)} and {get_device(preprocess)})")

    if metadata is not None:
        if n_ex is None:
            n_ex = metadata['X']['num_examples']
        if n_classes is None:
            n_classes = metadata['y']['num_classes']

    path = []
    best_val_loss = float('inf')

    if dataset_name == "places-10":
        dataset_name = "places"
    elif dataset_name == "cifar-10":
        dataset_name = "cifar10"

    if checkpoint is not None:
        os.makedirs(checkpoint, exist_ok=True)

        file_handler = logging.FileHandler(filename=os.path.join(checkpoint, 'output.log'))
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]

        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] %(levelname)s - %(message)s',
            handlers=handlers
        )
        logger = logging.getLogger('glm_saga').info
    else:
        logger = print

    # for i, (lam, lr) in enumerate(zip(lams, lrs)):
    #     start_time = time.time()
    # use relu factor to adjust sparsity
    weight = proj_layer.linear.weight.data.detach().cpu()
    num_classes, num_features = weight.shape
    factor_ws = ch.linspace(weight.min(), weight.max(), 101)[: 1]# .flip(0)
    for lam in factor_ws: # 0.01 ~ 0.1 / 0.03 ~ 0.5
        nw = F.relu(weight - lam)
        nnz = (nw > 1e-5).sum()
        sparsity = nnz / nw.numel()
        print(f'[lam {lam}] sparsity {sparsity}')
    factor_z = 0

    for i, factor_w in enumerate(factor_ws):
        # if i > 50 and i <= 90:
        #     continue
        proj_layer = Proj_Model(num_features, num_classes).to(get_device(proj_layer))
        params = proj_layer.parameters()
        optimizer = SGD(params, lr=0.1) #, momentum=0.9, weight_decay=0.00005)# 0.01 for places10, 0.001 for others

        proj_layer, _ = train_saga_early_stop(proj_layer, optimizer, loader, test_loader, nepochs,  factor_w, factor_z, use_relu,
                           table_device=table_device, preprocess=preprocess, group=group, verbose=verbose,
                           state=state, n_ex=n_ex, n_classes=n_classes, tol=tol, lookbehind=lookbehind,
                           family=family, logger=logger, ds_name=dataset_name)

        with ch.no_grad():
            proj_layer.eval()

            if val_loader:
                loss_val, acc_val, pavpus_val = elastic_loss_and_acc_loader(proj_layer, val_loader, factor_z=factor_z, factor_w=factor_w, uncertain_sample=True,
                                                     preprocess=preprocess, family=family)
                loss_val, acc_val = loss_val.item(), acc_val.item()


            if test_loader:
                loss_test, acc_test, pavpus_test = elastic_loss_and_acc_loader(proj_layer, test_loader, factor_z=factor_z, factor_w=factor_w,
                                                                            uncertain_sample=True,
                                                                            preprocess=preprocess, family=family)
                loss_test, acc_test = loss_test.item(), acc_test.item()


            params = {
                "metrics": {
                    "loss_val": loss_val,
                    "acc_val": acc_val,
                    "loss_test": loss_test,
                    "acc_test": acc_test,
                },
                "proj": proj_layer

            }
            path.append(params)


            nnz = (F.relu(proj_layer.linear.weight - factor_w).abs() > 1e-5).sum().item()
            total = proj_layer.linear.weight.numel()

            logger(
                f" [iteration {i}] [val loss {loss_val:.4f}] [test loss {loss_test:.4f}], weight sparsity {nnz / total} [{nnz}/{total}]\n"
                f"acc test {acc_test}\t acc val {acc_val}\tpavpus(0.01) {pavpus_test[0]:.4f}\t pavpus(0.05) {pavpus_test[1]:.4f}\t pavpus(0.1) {pavpus_test[2]:.4f}\t")

            ch.save(params, f'tmp/{dataset_name}/checkpoint/params{i}.pth')
    return {
        'path': path,
        'state': state
    }

    pass

# Calculate the regularization path of an elastic GLM with proximal SAGA
# Returns a dictionary of <regularization parameter> -> <linear weights and optimizer state>
def glm_saga(proj_layer, optimizer, loader, max_lr, nepochs, alpha, use_relu, dataset_name,
             table_device=None, preprocess=None, group=False,
             verbose=None, state=None, n_ex=None, n_classes=None,
             tol=1e-4, epsilon=0.001, k=100, checkpoint=None,
             do_zero=True, lr_decay_factor=1, metadata=None,
             val_loader=None, test_loader=None, lookbehind=None,
             family='multinomial', encoder=None):
    if encoder is not None:
        warnings.warn("encoder argument is deprecated; please use preprocess instead", DeprecationWarning)
        preprocess = encoder

    # if preprocess is not None and (get_device(proj_layer) != get_device(preprocess)):
    #     raise ValueError(
    #         "Linear and preprocess must be on same device (got {get_device(linear)} and {get_device(preprocess)})")

    if metadata is not None:
        if n_ex is None:
            n_ex = metadata['X']['num_examples']
        if n_classes is None:
            n_classes = metadata['y']['num_classes']

    path = []
    best_val_loss = float('inf')

    if checkpoint is not None:
        os.makedirs(checkpoint, exist_ok=True)

        file_handler = logging.FileHandler(filename=os.path.join(checkpoint, 'output.log'))
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [file_handler, stdout_handler]

        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] %(levelname)s - %(message)s',
            handlers=handlers
        )
        logger = logging.getLogger('glm_saga').info
    else:
        logger = print

    # for i, (lam, lr) in enumerate(zip(lams, lrs)):
    #     start_time = time.time()

    proj_layer = train_saga(proj_layer, optimizer, loader, test_loader, nepochs,  use_relu,
                       table_device=table_device, preprocess=preprocess, group=group, verbose=verbose,
                       state=state, n_ex=n_ex, n_classes=n_classes, tol=tol, lookbehind=lookbehind,
                       family=family, logger=logger, ds_name=dataset_name)

    with ch.no_grad():
        proj_layer.eval()

        loss_val, acc_val = -1, -1
        if val_loader:
            loss_val, acc_val, pavpus_val = elastic_loss_and_acc_loader(proj_layer, val_loader, factor_z=0, factor_w=0, uncertain_sample=True,
                                                 preprocess=preprocess, family=family)
            loss_val, acc_val = loss_val.item(), acc_val.item()

        loss_test, acc_test, best_acc_test = -1, -1, -1
        if test_loader:
            loss_test, acc_test, pavpus_test = elastic_loss_and_acc_loader(proj_layer, test_loader, factor_z=0, factor_w=0,
                                                                        uncertain_sample=True,
                                                                        preprocess=preprocess, family=family)
            loss_test, acc_test = loss_test.item(), acc_test.item()


        params = {
            "metrics": {
                "loss_val": loss_val,
                "acc_val": acc_val,
                "loss_test": loss_test,
                "acc_test": acc_test,
            },
            "proj": proj_layer.parameters()

        }
        path.append(params)


        nnz = (proj_layer.linear.weight.abs() > 1e-5).sum().item()
        total = proj_layer.linear.weight.numel()

        if family == 'multinomial':
            logger(
                f" [val acc {acc_val:.4f}] [test acc {acc_test:.4f}], weight sparsity {nnz / total} [{nnz}/{total}]]")
        elif family == 'gaussian':
            logger(
                f" [val loss {loss_val:.4f}] [test loss {loss_test:.4f}], weight sparsity {nnz / total} [{nnz}/{total}]")

    return {
        'path': path,
        'state': state
    }


# Given a loader, calculate the mean and standard deviation
# for normalization. If a model is provided, calculate the mean and 
# standard deviation of the resulting representation obtained by 
# first passing the example through the model. 
class NormalizedRepresentation(nn.Module):
    def __init__(self, loader, model=None, do_tqdm=True, mean=None, std=None, metadata=None, device='cuda'):
        super(NormalizedRepresentation, self).__init__()

        self.model = model
        if model is not None:
            device = get_device(model)
        self.device = device

        if metadata is not None:
            X_bar = metadata['X']['mean']
            X_std = metadata['X']['std']
        else:
            if mean is None:
                # calculate mean
                X_bar = 0
                n = 0
                it = enumerate(loader)
                if do_tqdm: it = tqdm(it, total=len(loader))

                for _, batch in it:
                    X = batch[0]
                    if model is not None:
                        X = model(X.to(device))

                    X_bar += X.sum(0)
                    n += X.size(0)

                X_bar = X_bar.float() / n
            else:
                X_bar = mean

            if std is None:
                # calculate std
                X_std = 0
                it = enumerate(loader)
                if do_tqdm: it = tqdm(it, total=len(loader))

                for _, batch in it:
                    X = batch[0]
                    if model is not None:
                        X = model(X.to(device))
                    X_std += ((X - X_bar) ** 2).sum(0)
                X_std = ch.sqrt(X_std / (n - 1))
            else:
                X_std = std

        self.mu = X_bar
        self.sigma = X_std

    def forward(self, X):
        if self.model is not None:
            device = get_device(self.model)
            X = self.model(X.to(device))
        return (X - self.mu.to(self.device)) / self.sigma.to(self.device)








