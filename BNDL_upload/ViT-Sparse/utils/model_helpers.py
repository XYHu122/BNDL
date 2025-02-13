import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as sts
import numpy as np


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

    def forward(self, X, factor_z=0, factor_w=0.02, force_sample=False):
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


def KL_GamWei(Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):
    def log_max(input, SMALL=1e-10):
        device = input.device
        input_ = torch.max(input, torch.tensor([SMALL]).to(device))
        return torch.log(input_)

    eulergamma = torch.tensor(0.5772, dtype=torch.float32, requires_grad=False)
    part1 = Gam_shape * log_max(Wei_scale) - eulergamma.to(Wei_scale.device) * Gam_shape * Wei_shape_res + log_max(
        Wei_shape_res)
    part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
    part3 = eulergamma.to(Wei_scale.device) + 1 + Gam_shape * log_max(Gam_scale) - torch.lgamma(Gam_shape)
    KL = part1 + part2 + part3
    return -KL.sum(1).mean()


def batch_uncertain(model_logits,
                    num_classes, target, inp, model,
                    accurate_pred, testresult):
    ## uncertainty estimation
    def two_sample_test_batch(logits, sample_num):
        prob = torch.softmax(logits, 1)
        probmean = torch.mean(prob, 2)
        values, indices = torch.topk(probmean, 2, dim=1)
        aa = logits.gather(1, indices[:, 0].unsqueeze(1).unsqueeze(1).repeat(1, 1, sample_num))
        bb = logits.gather(1, indices[:, 1].unsqueeze(1).unsqueeze(1).repeat(1, 1, sample_num))
        # if True:
        pvalue = sts.ttest_rel(aa.detach().cpu(), bb.detach().cpu(), axis=2).pvalue
        # else:
        # pvalue = np.zeros(shape=(aa.shape[0], aa.shape[1]))
        # for i in range(pvalue.shape[0]):
        #     pvalue[i] = sts.wilcoxon(aa.detach().cpu()[i, 0, :], bb.detach().cpu()[i, 0, :]).pvalue
        return pvalue

    sample_num = 20
    device = target.device
    logits_ii = np.zeros([inp.size(0), num_classes, sample_num])
    logits_greedy = np.zeros([inp.size(0), num_classes])
    logits_greedy[:, :] = model_logits.cpu().data.numpy()


    # sample
    for iii in range(sample_num):  # todo: uncertainty estimation
        # important step !!!!!!
        try:
            outputs = model.module.vit(inp)
            cls_token_embedding = outputs.last_hidden_state[:, 0, :]
            final_outputs = model.module.classifier(cls_token_embedding, force_sample=True)
        except:
            outputs = model.vit(inp)
            cls_token_embedding = outputs.last_hidden_state[:, 0, :]
            final_outputs = model.classifier(cls_token_embedding, force_sample=True)
        model_logits, z_out, weibull_lambda, k, weibull_lambda_w, k_w = final_outputs

        # model_logits = output[0] if (type(output) is tuple) else output
        logits_ii[:, :, iii] = model_logits.cpu().data.numpy()

    mean_logits = F.log_softmax(torch.mean(F.softmax(torch.from_numpy(logits_ii).to(device), dim=1), 2), 1)

    logits_tsam = torch.from_numpy(logits_ii).to(target.device)
    # prob = F.softmax(logits_tsam, 1)
    # ave_prob = torch.mean(prob, 2)
    # prediction = torch.argmax(ave_prob, 1).to(device)
    prediction = torch.argmax(torch.from_numpy(logits_greedy), 1).to(device)
    accurate_pred_i = (prediction == target).type_as(logits_tsam)
    accurate_pred = torch.cat([accurate_pred, accurate_pred_i], 0)
    testresult_i = torch.from_numpy(two_sample_test_batch(logits_tsam, sample_num)).type_as(logits_tsam)
    testresult = torch.cat([testresult, testresult_i], 0)
    # print(f'testresult shape {testresult.shape}')
    # print(testresult)
    return testresult, mean_logits, accurate_pred


def uncertain_cal(testresult, mean_logits, accurate_pred):
    uncertain = (testresult > 0.01).type_as(mean_logits)
    up_1 = uncertain.mean() * 100
    # ucpred_1 = ((uncertain == noise_mask_conca).type_as(mean_logits)).mean() * 100
    ac_1 = (accurate_pred * (1 - uncertain.squeeze())).sum()
    iu_1 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

    ac_prob_1 = ac_1 / (1 - uncertain.squeeze()).sum() * 100
    iu_prob_1 = iu_1 / (1 - accurate_pred).sum() * 100

    uncertain = (testresult > 0.05).type_as(mean_logits)
    up_2 = uncertain.mean() * 100
    # ucpred_2 = (uncertain == noise_mask_conca).type_as(mean_logits).mean() * 100
    ac_2 = (accurate_pred * (1 - uncertain.squeeze())).sum()
    iu_2 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

    ac_prob_2 = ac_2 / (1 - uncertain.squeeze()).sum() * 100
    iu_prob_2 = iu_2 / (1 - accurate_pred).sum() * 100

    uncertain = (testresult > 0.1).type_as(mean_logits)
    up_3 = uncertain.mean() * 100
    # ucpred_3 = (uncertain == noise_mask_conca).type_as(mean_logits).mean() * 100
    ac_3 = (accurate_pred * (1 - uncertain.squeeze())).sum()
    iu_3 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

    ac_prob_3 = ac_3 / (1 - uncertain.squeeze()).sum() * 100
    iu_prob_3 = iu_3 / (1 - accurate_pred).sum() * 100

    base_aic_1 = (ac_1 + iu_1) / accurate_pred.size(0) * 100 # todo: PavPU
    base_aic_2 = (ac_2 + iu_2) / accurate_pred.size(0) * 100
    base_aic_3 = (ac_3 + iu_3) / accurate_pred.size(0) * 100
    base_aic = [base_aic_1, base_aic_2, base_aic_3]  #PavPu

    ac_prob = [ac_prob_1, ac_prob_2, ac_prob_3]
    iu_prob = [iu_prob_1, iu_prob_2, iu_prob_3]
    # ucpred = [ucpred_1, ucpred_2, ucpred_3]

    # uncertainty proportion
    up = [up_1,up_2,up_3]
    return base_aic

