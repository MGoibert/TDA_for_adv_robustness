#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:34:09 2019

@author: m.goibert
"""

import copy
import numpy as np
import torch
from torch.autograd import Variable
import logging
logger = logging.getLogger()


def where(cond, x, y):
    """
    code from :
    https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.double()
    return (cond * x) + ((1 - cond) * y)

# Base class for attacks
class _BaseAttack(object):
    def __init__(self, model, num_iter=None, lims=(-0.5, 0.5)):
        self.model = model
        self.num_iter = num_iter

    def run(data, target, epsilon):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def clamp(self, data):
        return torch.clamp(data, -0.5, 0.5)

# FGSM
class FGSM(_BaseAttack):
    """
    Fast Gradient Sign Method
    """
    def __init__(self, model, loss_func):
        super(FGSM, self).__init__(model)
        self.loss_func = loss_func

    def run(self, data, target, epsilon, num_classes=10, pred=None, retain_graph=True):
        """
        XXX `retain_graph=True` is needed in case caller is calling this
        function in a loop, etc.
        """
        if pred is None:
            pred = self.model(data)
        loss = self.loss_func(pred, target, num_classes)
        self.model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        logger.info(f"L-inf difference between x and x_adv = {torch.abs(self.clamp(data + epsilon * data.grad.data.sign()) - data).max()}")
        return self.clamp(data + epsilon * data.grad.data.sign())

# BIM
class BIM(_BaseAttack):
    """
    BIM (Basic Iterative Method) or PGD method: iterative algorithm based on FGSM
    """
    def __init__(self, model, loss_func, num_iter=20, lims=(-0.5, 0.5)):
        super(BIM, self).__init__(model, num_iter=num_iter, lims=lims)
        self.loss_func = loss_func

    def run(self, data, target, epsilon, num_classes=10, epsilon_iter=None):
        target = target.detach()
        if epsilon_iter is None:
            epsilon_iter = 5 * epsilon / self.num_iter

        x_ori = data.data
        for _ in range(self.num_iter):
            x_adv = Variable(data.data, requires_grad=True)

            # forward pass
            h_adv = self.model(x_adv)
            self.model.zero_grad()
            loss = self.loss_func(h_adv, target, num_classes)

            # backward pass
            loss.backward(retain_graph=True)

            # single-step of FGSM: data <-- x_adv
            #x_adv.grad.sign_()  # x_adv.grad <-- x_adv.grad.sign()
            x_adv = data + epsilon_iter * data.grad.sign()
            eta = torch.clamp(x_adv - x_ori, min=-epsilon, max=epsilon)
            data = self.clamp(x_ori + eta)
            #x_adv = where(x_adv > data + epsilon, data + epsilon, x_adv)
            #x_adv = where(x_adv < data - epsilon, data - epsilon, x_adv)
        x_adv = data

        return x_adv

# Define CW attack then CW
def _to_attack_space(x, lims=(-0.5, 0.5)):
    """
    For C&W attack: transform an input from the model space (]min, max[,
    depending on the data) into  an input of the attack space (]-inf, inf[).
    Take a torch tensor as input.
    """
    # map from [min, max] to [-1, +1]
    a = (lims[0] + lims[1]) / 2
    b = (lims[1] - lims[0]) / 2
    x = (x - a) / b

    # from [-1, +1] to approx. (-1, +1)
    x = x * 0.999999999999999

    # from (-1, +1) to (-inf, +inf)
    x = 1. / 2. * torch.log((1 + x) / (1 - x))

    return x


def _to_model_space(x, lims=(-0.5, 0.5)):
    """
    For C&W attack: transform an input in the attack space (]-inf, inf[) into
    an input of the model space (]min, max[, depending on the data).
    Take a torch tensor as input.
    """

    # from (-inf, +inf) to (-1, +1)
    x = (1 - torch.exp(-2 * x * 0.999)) / (1 + torch.exp(
        -2 * x * 0.999999999999999))

    # map from (-1, +1) to (min, max)
    a = (lims[0] + lims[1]) / 2
    b = (lims[1] - lims[0]) / 2
    x = x * b + a

    return x

# ----------


def _soft_to_logit(softmax_list):
    """
    Maths: if p_i is the softmax corresponding to the logit z_i, then
    z_i = log(p_i) + const. This has not a unique solution, we choose const=0.
    XXX To check
    """
    soft_list = torch.clamp(softmax_list, 1e-250, (1-1e-15))
    return torch.log(soft_list)

# ----------


def _fct_to_min(adv_x, reconstruct_data, target, y_pred, logits, c, confidence=0, lims=(-0.5, 0.5)):
    """
    C&W attack: Objective function to minimize. Untargeted implementation.
    Parameters
    ---------
    adv_x: adversarial exemple (original data in the attack space + perturbation)
    reconstruct_data: almost original data (original after to_attack_space and
                     to_model_space)
    target: tensor, true target class
    logits: logits computed by the model with adv_x as input
    c: constant value tunning the strenght of having the adv. example
        missclassified wrt having an adv example close to the original data.
    confidence: parameter tunning the level of confidence for the adv exemple.
    Technical
    ---------
    We compute the general objective function to minimize in order to obtain a
    "good" adversarial example (misclassified by the model, but close to the
    original data). This function has two part:
        - the distance between the original data and the adv. data: it is the l2
        distance in this setup.
        - the constraint that the adv. example must be misclassified. The model
        does not predict the original/true class iff the the logits of the true
        class is smaller than the largest logit of the other classes, i.e. the
        function computed in is_adv_loss is less than zero.
    Note the confidence value (default=0) tunne how smaller the logit of the
    class must be compared to the highest logit.
    Please refer to Toward Evaluating the Robustness of Neural Networks, Carlini
    and Wagner, 2017 for more information.
    """
    # Logits
    #logits = _soft_to_logit(y_pred)

    # Index of original class
    if False:
        adv_x = adv_x[:1]
        reconstruct_data = reconstruct_data[:1]
        logits = logits[:1]
    #c_min = target.data.numpy()  # .item()

    c_min = target.data # sans numpy implem
    c_max = (torch.stack( [ a != a[i] for a, i in zip(y_pred, target) ] ).double()*y_pred).max(dim=-1)[1]
    i = range(len(logits))

    is_adv_loss = torch.max(logits[i, c_min] - logits[i, c_max] + confidence, torch.zeros_like(logits[i, c_min])).to(adv_x.device)

    # Perturbation size part of the objective function: corresponds to the
    # minimization of the distance between the "true" data and the adv. data.
    scale = (lims[1] - lims[0]) ** 2
    l2_dist = ((adv_x - reconstruct_data) **2).sum(1).sum(1).sum(1) / scale

    # Objective function
    tot_dist = l2_dist.to(adv_x.device) + c.to(adv_x.device) * is_adv_loss.to(adv_x.device)
    return tot_dist

# ----------


def CW_attack(data, target, model, binary_search_steps=15, num_iter=50,
              confidence=0, learning_rate=0.1, initial_c=1, lims=(-0.5, 0.5)):
    """
    Carlini & Wagner attack.
    Untargeted implementation, L2 setup.
    """
    data = data.unsqueeze(0)
    batch_size = 1 if len(data.size()) < 4 else len(data)
    #logger.info(f"batch size = {batch_size}")
    att_original = _to_attack_space(data.detach(), lims=lims)
    #logger.info(f"att_original size = {att_original.size()}")
    reconstruct_original = _to_model_space(att_original, lims=lims)

    c = torch.ones(batch_size) * initial_c
    lower_bound = np.zeros(batch_size)
    upper_bound = np.ones(batch_size) * np.inf
    best_x = data

    for binary_search_step in range(binary_search_steps):
        perturb = [ torch.zeros_like(att_original[t], requires_grad=True)
                   for t in range(batch_size)]
        #logger.info(f"perturb = {perturb}")
        optimizer_CW = [torch.optim.Adam([perturb[t]], lr=learning_rate)
                        for t in range(batch_size)]
        found_adv = torch.zeros(batch_size).byte()
        #logger.info(f"found_adv = {found_adv}")

        for iteration in range(num_iter):
            #logger.info(f"Iteration = {iteration}")
            #logger.info(f"Just before x torch cat = {torch.cat([perturb_.unsqueeze(0) for perturb_ in perturb])} and size = {torch.cat([perturb_.unsqueeze(0) for perturb_ in perturb]).size()}")
            x = _to_model_space(att_original + torch.cat([perturb_.unsqueeze(0) for perturb_ in perturb]) , lims=lims)
            y_pred = model(x)
            #logits = _soft_to_logit(y_pred)
            logits = model(x, presoft=True)
            cost = _fct_to_min(x, reconstruct_original, target, y_pred, logits, c,
                               confidence, lims=lims)
            #logger.info(f"    x = {x}")
            #logger.info(f"    y_pred = {y_pred}")
            #logger.info(f"    logits = {logits}")
            #logger.info(f"    cost = {cost}")
            for t in range(batch_size):
                optimizer_CW[t].zero_grad()
                cost[t].backward(retain_graph=True)
                #logger.info(f"    Gradients = {perturb[t].grad}")
                optimizer_CW[t].step()
                #logger.info(f"    perturb = {perturb} and size = {perturb[0].size()} and unsqueeze version = {perturb[0].unsqueeze(0)} and size unsqueeze = {perturb[0].unsqueeze(0).size()}")
                if logits[t].squeeze().argmax(-1, keepdim=True).item() != target[t]:
                    found_adv[t] = 1
                else:
                    found_adv[t] = 0

        for t in range(batch_size):
            if found_adv[t]:
                upper_bound[t] = c[t]
                best_x[t] = x[t]
            else:
                lower_bound[t] = c[t]
            if upper_bound[t] == np.inf:
                c[t] = 10 * c[t]
            else:
                c[t] = (lower_bound[t] + upper_bound[t]) / 2

    return best_x.squeeze(0)

class CW(_BaseAttack):
    """
    Carlini-Wagner Method
    """
    def __init__(self, model, binary_search_steps=30,
                 num_iter=50, lims=(-0.5, 0.5)):
        _BaseAttack.__init__(self, model, lims=lims)
        self.binary_search_steps = binary_search_steps
        self.num_iter = num_iter
        self.lims = lims

    def run(self, data, target, **kwargs):
        logger.info(f"CW binary search steps = {self.binary_search_steps} and number iterations = {self.num_iter}")
        perturbed_data = CW_attack(
            data, target, self.model, num_iter=self.num_iter,
            binary_search_steps=self.binary_search_steps, lims=self.lims, **kwargs)
        return self.clamp(perturbed_data)



class DeepFool(_BaseAttack):
    def __init__(self, model, num_classes,  num_iter=10):
        super(DeepFool, self).__init__(model, num_iter=num_iter)
        self.num_classes = num_classes
        self.num_iter = num_iter
        self.model = model

    def run(self, image, true_label, epsilon=None):
        #logger.info(f"DeepFool number of iteration = {self.num_iter}")
        self.model.eval()

        #logger.info(f"model = {self.model}")
        #logger.info(f"model modules = {list(self.model.modules())}")
        #newmodel = torch.nn.Sequential(*(list(self.model.modules())[:-1]))
        #logger.info(f"newmodel = {newmodel}")

        nx = torch.unsqueeze(image, 0).detach().cpu().numpy().copy()
        nx = torch.from_numpy(nx)
        nx.requires_grad = True
        eta = torch.zeros(nx.shape)

        out = self.model(nx+eta, presoft=True)
        #logger.info(f"newmodel pred = {newmodel(nx)}")
        #logger.info(f"out = {out}")
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.num_iter:
            out[0, py].backward(retain_graph=True)
            #logger.info(f"out[0, py] = {out[0, py]}")
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(self.num_classes):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                #logger.info(f"out[0, i] = {out[0, i]}")
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                if np.linalg.norm(wi.numpy().flatten())==0.0: logger.info(f"Denominator = 0 for i = {i} and iter = {i_iter}")
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())
                #logger.info(f"wi val = {wi}")
                #logger.info(f"fi val = {fi}")
                #logger.info(f"value i val = {value_i}")

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi
                    #logger.info(f"r_i val = {ri}")

            eta += ri.clone() if type(ri) != type(None) else 0
            nx.grad.data.zero_()
            out = _soft_to_logit(self.model(nx+eta))
            py = out.max(1)[1].item()
            i_iter += 1
        
        #logger.info(f"DeepFool total run = {i_iter}")
        #if i_iter == self.num_iter:
        #    logger.info(f"eta (perturbation) after {i_iter} run = {eta} and norm = {np.linalg.norm(eta.detach().numpy().flatten())}")
        x_adv = nx + eta
        #x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)

        #logger.info(f"x_adv = {x_adv} and size = {x_adv.size()}")
        return self.clamp(x_adv)
