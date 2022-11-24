from src.models import resnet18_10cls
from src.utils import *
import torch
import torch.nn as nn
from advertorch.attacks import LinfPGDAttack, PGDAttack
import os
from functools import partial


def perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,
                      l1_sparsity=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(xvar + delta)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)

        elif ord == 1:
            grad = delta.grad.data
            abs_grad = torch.abs(grad)

            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if l1_sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(
                    int(np.round((1 - l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + batch_multiply(eps_iter, grad)

            delta.data = batch_l1_proj(delta.data.cpu(), eps)
            if xvar.is_cuda:
                delta.data = delta.data.cuda()
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv, delta

class MyPGDAttack(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False):
        """
        Create an instance of the PGDAttack.

        """
        super(MyPGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval, self.delta = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )

        return rval.data


class UnNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
class AttackRunner():
    def __init__(self, config, exp):
        self.exp            = exp
        self.config         = config
        self.model          = self.config.model
        self.data_info      = self.config.data_info
        self.checkpoint     = self.config.checkpoint
        self.epsilon        = self.config.epsilon
        self.dir            = self.config.dir
        self.step_size      = self.config.step_size
        self.num_steps      = self.config.num_steps
        self.print_freq     = 10
        
    def run(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        safe_mkdir(f"{self.dir}/{self.exp}")     
        print = partial(lprint, dir=os.path.join(self.dir, self.exp))

        data_loader = self.data_info["dataloader"]["valid"]
        self.model.load_state_dict(torch.load(self.checkpoint))
        self.model = self.model.to(self.device)
        self.model.eval()
        epsilon = self.epsilon / 255.0
        step_size = self.step_size / 255.0  
        
        print('using linf PGD attack')
        adversary = MyPGDAttack(predict=self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, nb_iter=self.num_steps, eps_iter=step_size,rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)

        self.generate_adversarial_example(data_loader=data_loader, adversary=adversary, img_path=os.path.join(self.dir, self.exp))
    
    def generate_adversarial_example(self, data_loader, adversary, img_path, unorm=True, save_perturb=True):
        print = partial(lprint, dir=os.path.join(self.dir, self.exp))
        self.model.eval()
        for batch_idx, (inputs, true_class) in enumerate(data_loader):
            inputs, true_class = inputs.to(self.device), true_class.to(self.device)
            if unorm:
                unorm = UnNormalize()
                unorm(inputs)
            inputs_adv = adversary.perturb(inputs, true_class)
            img_list = [f"img_bcidx{batch_idx}_idx{x}_cls{true_class[x]}.png" for x in range(len(true_class))]
            save_images(inputs_adv, img_list, img_path)
            img_list_ori = [f"img_bcidx{batch_idx}_idx{x}_cls{true_class[x]}_ori.png" for x in range(len(true_class))]
            save_images(inputs, img_list_ori, img_path)
            if save_perturb:
                img_list_delta = [f"img_bcidx{batch_idx}_idx{x}_cls{true_class[x]}_delta.png" for x in range(len(true_class))]
                safe_mkdir(os.path.join(img_path, "purturb"))
                save_images(adversary.delta, img_list_delta, os.path.join(img_path, "purturb"))
            if batch_idx % self.print_freq == 0:
                print('generating: [{0}/{1}]'.format(batch_idx, len(data_loader)))
        
   

