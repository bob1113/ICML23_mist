# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import rand_init_delta


def perturb_iterative(
    xvar,
    yvar,
    predict,
    nb_iter,
    eps,
    eps_iter,
    loss_fn,
    delta_init=None,
    minimize=False,
    ord=np.inf,
    clip_min=0.0,
    clip_max=1.0,
    l1_sparsity=None,
    mask=None,
):
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
        if mask is None:
            outputs = predict(xvar + delta)
        else:
            outputs = predict(xvar + delta * mask)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
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
                vals, idx = view.topk(int(np.round((1 - l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + batch_multiply(eps_iter, grad)

            delta.data = batch_l1_proj(delta.data.cpu(), eps)
            delta.data = delta.data.to(xvar.device)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()
    if mask is None:
        x_adv = clamp(xvar + delta, clip_min, clip_max)
    else:
        x_adv = clamp(xvar + delta * mask, clip_min, clip_max)
    return x_adv


class PGDAttack(Attack, LabelMixin):
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
        self,
        predict,
        loss_fn=None,
        eps=0.3,
        nb_iter=40,
        eps_iter=0.01,
        rand_init=True,
        clip_min=0.0,
        clip_max=1.0,
        ord=np.inf,
        l1_sparsity=None,
        targeted=False,
    ):
        """
        Create an instance of the PGDAttack.
        """
        super(PGDAttack, self).__init__(predict, loss_fn, clip_min, clip_max)
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

    def perturb(self, x, y=None, mask=None):
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
            rand_init_delta(delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = perturb_iterative(
            x,
            y,
            self.predict,
            nb_iter=self.nb_iter,
            eps=self.eps,
            eps_iter=self.eps_iter,
            loss_fn=self.loss_fn,
            minimize=self.targeted,
            ord=self.ord,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            delta_init=delta,
            l1_sparsity=self.l1_sparsity,
            mask=mask,
        )

        return rval.data


class L2PGDAttack(PGDAttack):
    """
    PGD Attack with order=L2
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
        self,
        predict,
        loss_fn=None,
        eps=0.3,
        nb_iter=40,
        eps_iter=0.01,
        rand_init=True,
        clip_min=0.0,
        clip_max=1.0,
        targeted=False,
    ):
        ord = 2
        super(L2PGDAttack, self).__init__(
            predict=predict,
            loss_fn=loss_fn,
            eps=eps,
            nb_iter=nb_iter,
            eps_iter=eps_iter,
            rand_init=rand_init,
            clip_min=clip_min,
            clip_max=clip_max,
            targeted=targeted,
            ord=ord,
        )


class LinfPGDAttack(PGDAttack):
    """
    PGD Attack with order=Linf
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False):
        ord = np.inf
        super(LinfPGDAttack, self).__init__(
            predict=predict,
            loss_fn=loss_fn,
            eps=eps,
            nb_iter=nb_iter,
            eps_iter=eps_iter,
            rand_init=rand_init,
            clip_min=clip_min,
            clip_max=clip_max,
            targeted=targeted,
            ord=ord,
        )


class MIFGSM(Attack, LabelMixin):
    """
    Momentum Iterative Attack (Dong et al. 2017).
    Similar to PGD but integrates momentum into the gradient update.
    Reference: https://arxiv.org/pdf/1710.06081.pdf
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, nb_iter=40, decay_factor=1.0, eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False, ord=np.inf):
        super(MIFGSM, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.decay_factor = decay_factor
        self.eps_iter = eps_iter
        self.targeted = targeted
        self.ord = ord
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None, mask=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack strength of eps, using momentum in gradient updates.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :param mask: optional mask tensor, same shape as x, indicating which pixels
                     are allowed to be perturbed.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        # Initialize the perturbation
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)

        # If you want a random start similar to PGD, uncomment these lines:
        rand_init_delta(delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
        delta.data = clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x

        # Momentum buffer
        g = torch.zeros_like(x)

        for i in range(self.nb_iter):
            delta.requires_grad_()
            if mask is None:
                adv_x = x + delta
            else:
                adv_x = x + delta * mask

            outputs = self.predict(adv_x)
            loss = self.loss_fn(outputs, y)
            if self.targeted:
                loss = -loss

            loss.backward()
            grad = delta.grad.data

            # Normalize gradient
            if self.ord == np.inf:
                # For MIFGSM in L_inf, typically the grad is normalized by its L1 norm
                # as per the original paper. However, you can also try L2 normalization.
                grad_norm = torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True) + 1e-10
                g = self.decay_factor * g + grad / grad_norm
                # Update delta: sign of accumulated gradient
                delta.data = delta.data + self.eps_iter * g.sign()
                # Projection onto L_inf ball
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)
                # Clip to valid range
                delta.data = clamp(x + delta.data, self.clip_min, self.clip_max) - x

            elif self.ord == 2:
                # For L2 MIFGSM (less common), normalize by L2
                grad_norm = torch.sqrt((grad**2).sum(dim=[1, 2, 3], keepdim=True)) + 1e-10
                g = self.decay_factor * g + grad / grad_norm
                # Update delta in direction of g
                norm_g = torch.sqrt((g**2).sum(dim=[1, 2, 3], keepdim=True))
                scaled_g = g / (norm_g + 1e-10)
                delta.data = delta.data + self.eps_iter * scaled_g
                # Project onto L2 ball
                norm_delta = torch.sqrt((delta.data**2).sum(dim=[1, 2, 3], keepdim=True))
                factor = torch.min(torch.tensor(1.0, device=x.device), self.eps / norm_delta)
                delta.data = delta.data * factor
                # Clip to valid range
                delta.data = clamp(x + delta.data, self.clip_min, self.clip_max) - x

            else:
                raise NotImplementedError("MIFGSM currently only supports ord=inf or ord=2")

            delta.grad.zero_()

        if mask is None:
            x_adv = clamp(x + delta, self.clip_min, self.clip_max)
        else:
            x_adv = clamp(x + delta * mask, self.clip_min, self.clip_max)

        return x_adv.data


class NIFGSM(Attack, LabelMixin):
    """
    NIFGSM: Nesterov Iterative FGSM Attack
    Reference:
    - "Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks"
      [https://arxiv.org/abs/1908.06281]

    Similar to MIFGSM but computes the gradient at a "look-ahead" point:
    x_nes = x + delta + decay_factor * g, where g is momentum.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, nb_iter=40, decay_factor=1.0, eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False, ord=np.inf):
        super(NIFGSM, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.decay_factor = decay_factor
        self.eps_iter = eps_iter
        self.targeted = targeted
        self.ord = ord
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None, mask=None):
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x, requires_grad=True)
        g = torch.zeros_like(x)  # momentum

        for i in range(self.nb_iter):
            # Nesterov step: Evaluate gradient at the look-ahead point
            x_nes = x + delta + self.decay_factor * g

            if mask is None:
                outputs = self.predict(x_nes)
            else:
                outputs = self.predict(x + (delta + self.decay_factor * g) * mask)

            loss = self.loss_fn(outputs, y)
            if self.targeted:
                loss = -loss

            loss.backward()
            grad = delta.grad.data

            # Normalize and accumulate momentum
            grad_norm = torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True) + 1e-10
            g = self.decay_factor * g + grad / grad_norm

            # Update delta
            delta.data = delta.data + self.eps_iter * g.sign()

            # Project onto L∞ ball
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)

            # Clip to [clip_min, clip_max]
            if mask is None:
                delta.data = clamp(x + delta.data, self.clip_min, self.clip_max) - x
            else:
                delta.data = clamp(x + delta.data * mask, self.clip_min, self.clip_max) - x

            delta.grad.zero_()

        if mask is None:
            x_adv = clamp(x + delta, self.clip_min, self.clip_max)
        else:
            x_adv = clamp(x + delta * mask, self.clip_min, self.clip_max)

        return x_adv.data


def input_diversity(x, prob=0.5, low=224, high=256):
    """
    Example input diversity function:
    With probability `prob`, resize the input to a random size between [low, high]
    and then pad it back to the original size.

    Adjust as needed:
    - The original input is assumed to be square for simplicity.
    - The transformations can be more complex: random crops, flips, color jitter, etc.
    """
    if np.random.rand() > prob:
        return x
    orig_size = x.shape[-1]
    rnd = np.random.randint(low, high)
    # Resize
    x_resize = nn.functional.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    # Pad
    pad_size = orig_size - rnd
    pad_left = np.random.randint(0, pad_size + 1)
    pad_right = pad_size - pad_left
    pad_top = np.random.randint(0, pad_size + 1)
    pad_bottom = pad_size - pad_top
    x_pad = nn.functional.pad(x_resize, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    return x_pad


class VMIFGSM(Attack, LabelMixin):
    """
    VMIFGSM: Variance-Reduced Momentum Iterative FGSM
    Introduced in:
    - "Enhancing Adversarial Defense by k-Winner-Takes-All Activations" (some variants discussed)
    - Another relevant reference is "Variance Reduced Attacks" and "Adversarial examples in the
      physical world" that propose input transformations.

    The key idea: At each iteration, generate multiple transformed versions of the adversarial
    example and average their gradients to reduce variance.
    """

    def __init__(
        self, predict, loss_fn=None, eps=0.3, nb_iter=40, decay_factor=1.0, eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False, ord=np.inf, num_samples=5, diversity_prob=0.5, low=224, high=256
    ):
        """
        :param num_samples: number of diverse samples to average the gradient over.
        :param diversity_prob: probability of applying the diversity transformation.
        :param low, high: range of sizes for the random resize in input_diversity.
        """
        super(VMIFGSM, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.decay_factor = decay_factor
        self.eps_iter = eps_iter
        self.targeted = targeted
        self.ord = ord
        self.num_samples = num_samples
        self.diversity_prob = diversity_prob
        self.low = low
        self.high = high
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None, mask=None):
        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x, requires_grad=True)
        g = torch.zeros_like(x)  # momentum

        for i in range(self.nb_iter):
            # Compute averaged gradient over multiple transformed samples
            grads = []
            for _ in range(self.num_samples):
                if mask is None:
                    adv_x = x + delta
                else:
                    adv_x = x + delta * mask

                # Apply input diversity
                adv_x_div = input_diversity(adv_x, prob=self.diversity_prob, low=self.low, high=self.high)

                outputs = self.predict(adv_x_div)
                loss = self.loss_fn(outputs, y)
                if self.targeted:
                    loss = -loss

                delta.grad = None
                loss.backward(retain_graph=True)  # We retain_graph to accumulate multiple samples

                grads.append(delta.grad.data.clone())

            # Average the gradients
            mean_grad = torch.mean(torch.stack(grads), dim=0)

            # Momentum update
            grad_norm = torch.mean(torch.abs(mean_grad), dim=[1, 2, 3], keepdim=True) + 1e-10
            g = self.decay_factor * g + mean_grad / grad_norm

            # Update delta
            delta.data = delta.data + self.eps_iter * g.sign()

            # Project onto L∞ ball
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)

            # Clip to [clip_min, clip_max]
            if mask is None:
                delta.data = clamp(x + delta.data, self.clip_min, self.clip_max) - x
            else:
                delta.data = clamp(x + delta.data * mask, self.clip_min, self.clip_max) - x

        if mask is None:
            x_adv = clamp(x + delta, self.clip_min, self.clip_max)
        else:
            x_adv = clamp(x + delta * mask, self.clip_min, self.clip_max)

        return x_adv.data

class VNIFGSM(Attack, LabelMixin):
    """
    VNIFGSM: Variance-Reduced Nesterov Iterative FGSM
    Combines NIFGSM (Nesterov acceleration) with VMIFGSM (variance-reduced gradients via input diversity).

    Steps:
    1. Compute a look-ahead point x_nes = x + delta + decay_factor * g.
    2. At each iteration, for `num_samples` transformed versions of x_nes, compute gradient and average.
    3. Update momentum and delta using the averaged gradient.
    4. Project onto L∞ ball and clamp final output.
    """

    def __init__(
        self, predict, loss_fn=None, eps=0.3, nb_iter=40, decay_factor=1.0, 
        eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False, ord=np.inf,
        num_samples=5, diversity_prob=0.5, low=224, high=256
    ):
        super(VNIFGSM, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.decay_factor = decay_factor
        self.eps_iter = eps_iter
        self.targeted = targeted
        self.ord = ord
        self.num_samples = num_samples
        self.diversity_prob = diversity_prob
        self.low = low
        self.high = high

        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None, mask=None):
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x, requires_grad=True)
        g = torch.zeros_like(x)  # momentum buffer

        for i in range(self.nb_iter):
            # Nesterov step: Compute look-ahead point
            if mask is None:
                x_nes = x + delta + self.decay_factor * g
            else:
                x_nes = x + (delta + self.decay_factor * g) * mask

            # Compute averaged gradient over multiple transformed samples
            grads = []
            for _ in range(self.num_samples):
                adv_x_div = input_diversity(x_nes, prob=self.diversity_prob, low=self.low, high=self.high)

                outputs = self.predict(adv_x_div)
                loss = self.loss_fn(outputs, y)
                if self.targeted:
                    loss = -loss

                delta.grad = None
                loss.backward(retain_graph=True)
                grads.append(delta.grad.data.clone())

            # Average gradients to reduce variance
            mean_grad = torch.mean(torch.stack(grads), dim=0)

            # Momentum update
            grad_norm = torch.mean(torch.abs(mean_grad), dim=[1,2,3], keepdim=True) + 1e-10
            g = self.decay_factor * g + mean_grad / grad_norm

            # Update delta
            delta.data = delta.data + self.eps_iter * g.sign()

            # Project onto L∞ ball
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)

            # Clip final adversarial image
            if mask is None:
                delta.data = clamp(x + delta.data, self.clip_min, self.clip_max) - x
            else:
                delta.data = clamp(x + delta.data * mask, self.clip_min, self.clip_max) - x

        if mask is None:
            x_adv = clamp(x + delta, self.clip_min, self.clip_max)
        else:
            x_adv = clamp(x + delta * mask, self.clip_min, self.clip_max)

        return x_adv.data
