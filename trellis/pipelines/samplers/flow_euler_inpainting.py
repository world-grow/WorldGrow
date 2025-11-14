from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from ...modules import sparse as sp
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from .flow_euler import FlowEulerSampler


class FlowEulerInpaintingSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        super().__init__(sigma_min)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        x_t_no_mask = x_t[:, :pred_v.shape[1]]
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t_no_mask, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        x_t_no_mask = x_t[:, :pred_v.shape[1]]
        pred_x_prev = x_t_no_mask - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        x_0: torch.Tensor,
        masked_x_0: torch.Tensor,
        mask: torch.Tensor,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        strength: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        t_pairs = t_pairs[int(len(t_pairs) * (1 - strength)):]
        if isinstance(noise, torch.Tensor):
            cat = torch.cat
        elif isinstance(noise, sp.SparseTensor):
            cat = sp.sparse_cat
        else:
            raise NotImplementedError(f"Sample type {type(noise)}, {type(mask)} and {type(masked_x_0)} cannot be catted.")
        if len(t_pairs) == 0:
            return x_0
        pred_x_prev = t_pairs[0][0] * noise + (1 - t_pairs[0][0]) * x_0
        sample = cat([pred_x_prev, mask, masked_x_0], dim=1)
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            pred_x_prev = out.pred_x_prev * mask + (t_prev * noise + (1.0 - t_prev) * masked_x_0) * (1 - mask)
            sample = cat([pred_x_prev, mask, masked_x_0], dim=1)
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample[:, :x_0.shape[1]]
        return ret


class FlowEulerInpaintingCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerInpaintingSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        x_0: torch.Tensor,
        masked_x_0: torch.Tensor,
        mask: torch.Tensor,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        strength: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, x_0, masked_x_0, mask, cond, steps, rescale_t, strength, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerInpaintingGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerInpaintingSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        x_0: torch.Tensor,
        masked_x_0: torch.Tensor,
        mask: torch.Tensor,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        strength: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, x_0, masked_x_0, mask, cond, steps, rescale_t, strength, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
