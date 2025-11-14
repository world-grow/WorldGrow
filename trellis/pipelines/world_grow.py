from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from copy import deepcopy
from transformers import CLIPTextModel, AutoTokenizer
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..utils import postprocessing_utils


class WorldGrowPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
        text_prompt (str): The text prompt used for conditioning.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
        text_prompt: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)
        self.text_prompt = text_prompt

    @staticmethod
    def from_pretrained(path: str) -> "WorldGrowPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(WorldGrowPipeline, WorldGrowPipeline).from_pretrained(path)
        new_pipeline = WorldGrowPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        new_pipeline.text_prompt = args['text_prompt']

        return new_pipeline

    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state

        return embeddings

    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure_coarse(
        self,
        cond: dict,
        world_size: Tuple[int] = (3, 3),
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            world_size (Tuple[int]): The size of the world (H, W).
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_coarse_flow_model']
        ss_encoder = self.models['sparse_structure_encoder']
        ss_decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution
        breso = reso * (2 ** (len(ss_encoder.channels) - 1))
        coarse_world = torch.zeros((1, 1, breso // 2 * ((world_size[0] + 1) // 2 + 1), breso // 2 * ((world_size[1] + 1) // 2 + 1), breso)).cuda()
        coarse_world_latents = torch.zeros((1, flow_model.out_channels, reso // 2 * ((world_size[0] + 1) // 2 + 1), reso // 2 * ((world_size[1] + 1) // 2 + 1), reso)).cuda()
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        for i in range((world_size[0] + 1) // 2):
            for j in range((world_size[1] + 1) // 2):
                x_0 = coarse_world[:, :, i*breso//2:i*breso//2+breso, j*breso//2:j*breso//2+breso, :].clone().detach()
                x_0[x_0 > 0] = 1
                x_0[x_0 < 0] = 0
                z_0 = coarse_world_latents[:, :, i*reso//2:i*reso//2+reso, j*reso//2:j*reso//2+reso, :].clone().detach()
                noise = torch.randn(1, flow_model.out_channels, reso, reso, reso).cuda()
                mask = torch.ones(1, 1, breso, breso, breso).cuda()
                if i == 0 and j == 0:
                    pass
                elif i == 0:
                    mask[:, :, :, :breso * 3 // 8, :] = 0
                elif j == 0:
                    mask[:, :, :breso * 3 // 8, :, :] = 0
                else:
                    mask[:, :, :, :breso * 3 // 8, :] = 0
                    mask[:, :, :breso * 3 // 8, :, :] = 0
                masked_x_0 = x_0 * (1 - mask)
                latent = ss_encoder(masked_x_0, sample_posterior=False)
                masked_z_0 = latent.clone().detach().float()
                rescaled_mask = F.interpolate(mask, size=z_0.shape[-3:])
                z_s = self.sparse_structure_sampler.sample(
                    flow_model,
                    noise,
                    z_0,
                    masked_z_0,
                    rescaled_mask,
                    **cond,
                    **sampler_params,
                    strength=1.0,
                    verbose=True,
                ).samples
                coarse_world_latents[:, :, i*reso//2:i*reso//2+reso, j*reso//2:j*reso//2+reso, :] = z_s
                coarse_world[:, :, i*breso//2:i*breso//2+breso, j*breso//2:j*breso//2+breso, :] = ss_decoder(z_s)

        return coarse_world

    def sample_sparse_structure_fine(
        self,
        cond: dict,
        coarse_world: torch.Tensor,
        world_size: Tuple[int] = (3, 3),
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coarse_world (torch.Tensor): The coarse sparse structures.
            world_size (Tuple[int]): The size of the world (H, W).
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_fine_flow_model']
        ss_encoder = self.models['sparse_structure_encoder']
        ss_decoder = self.models['sparse_structure_decoder']
        reso = flow_model.resolution
        breso = reso * (2 ** (len(ss_encoder.channels) - 1))
        fine_world = F.interpolate(coarse_world, size=(breso // 2 * (world_size[0] + 1), breso // 2 * (world_size[1] + 1), breso * 2))[:, :, :, :, breso//2:breso//2*3]
        fine_world[fine_world > 0] = 1
        fine_world[fine_world < 0] = 0
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        for i in range(world_size[0]):
            for j in range(world_size[1]):
                x_0 = fine_world[:, :, i*breso//2:i*breso//2+breso, j*breso//2:j*breso//2+breso, :].clone().detach()
                z_0 = ss_encoder(x_0, sample_posterior=False)
                noise = torch.randn(1, flow_model.out_channels, reso, reso, reso).cuda()
                mask = torch.ones(1, 1, breso, breso, breso).cuda()
                if i == 0 and j == 0:
                    pass
                elif i == 0:
                    mask[:, :, :, :breso * 3 // 8, :] = 0
                elif j == 0:
                    mask[:, :, :breso * 3 // 8, :, :] = 0
                else:
                    mask[:, :, :, :breso * 3 // 8, :] = 0
                    mask[:, :, :breso * 3 // 8, :, :] = 0
                masked_x_0 = x_0 * (1 - mask)
                latent = ss_encoder(masked_x_0, sample_posterior=False)
                masked_z_0 = latent.clone().detach().float()
                rescaled_mask = F.interpolate(mask, size=z_0.shape[-3:])
                z_s = self.sparse_structure_sampler.sample(
                    flow_model,
                    noise,
                    z_0,
                    masked_z_0,
                    rescaled_mask,
                    **cond,
                    **sampler_params,
                    strength=0.3,
                    verbose=True,
                ).samples
                fine_world[:, :, i*breso//2:i*breso//2+breso, j*breso//2:j*breso//2+breso, :] = ss_decoder(z_s)
                fine_world[fine_world > 0] = 1
                fine_world[fine_world < 0] = 0

        return fine_world


    def decode_slat(
        self,
        world: torch.Tensor,
        world_feats: torch.Tensor,
        world_size: Tuple[int] = (3, 3),
        formats: List[str] = ['gaussian', 'mesh'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            world (torch.Tensor): The coordinates of the sparse structure.
            world_feats (torch.Tensor): The features of structured latent.
            world_size (Tuple[int]): The size of the world (H, W).
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        breso = world.shape[-1]
        std = torch.tensor(self.slat_normalization['std'])[None].to(world_feats.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(world_feats.device)
        ret = {}
        gaussians = []
        glb_scene = trimesh.Scene()

        for i in range(world_size[0]):
            for j in range(world_size[1]):
                ss = world[:, :, i*breso//2:i*breso//2+breso, j*breso//2:j*breso//2+breso, :]
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
                slat_0 = world_feats[i*breso//2:i*breso//2+breso, j*breso//2:j*breso//2+breso, :].clone().detach()
                feats = []
                for coord in coords:
                    _, x, y, z = coord
                    feats.append(slat_0[x, y, z])
                if len(feats) == 0:
                    continue
                feats = torch.stack(feats).cuda()
                slat = sp.SparseTensor(
                    feats=feats,
                    coords=coords,
                )
                slat = slat * std + mean
                if 'gaussian' in formats:
                    gaussian = self.models['slat_decoder_gs'](slat)[0]
                    if 'mesh' in formats:
                        mesh = self.models['slat_decoder_mesh'](slat)[0]
                        with torch.enable_grad():
                            glb = postprocessing_utils.to_glb(gaussian, mesh, simplify=0.9, texture_size=1024)
                        trans = np.eye(4)
                        trans[0, 3] = i / 2
                        trans[1, 3] = 0
                        trans[2, 3] = -j / 2
                        glb.apply_transform(trans)
                        glb_scene.add_geometry(glb)
                    gaussian.trim_edge(0.15)
                    trans = np.eye(4)
                    trans[0, 3] = i / 2
                    trans[1, 3] = j / 2
                    trans[2, 3] = 0
                    gaussian.transform(trans)
                    gaussians.append(gaussian)

        if 'gaussian' in formats:
            xyzs = []
            features_dcs = []
            opacities = []
            scalings = []
            rotations = []
            for gaussian in gaussians:
                xyzs.append(gaussian._xyz)
                features_dcs.append(gaussian._features_dc)
                opacities.append(gaussian._opacity)
                scalings.append(gaussian._scaling)
                rotations.append(gaussian._rotation)
            gaussian_all = deepcopy(gaussians[0])
            gaussian_all._xyz = torch.cat(xyzs)
            gaussian_all._features_dc = torch.cat(features_dcs)
            gaussian_all._opacity = torch.cat(opacities)
            gaussian_all._scaling = torch.cat(scalings)
            gaussian_all._rotation = torch.cat(rotations)
            ret['gaussian'] = gaussian_all
            if 'mesh' in formats:
                ret['mesh'] = glb_scene

        return ret

    def sample_slat(
        self,
        cond: dict,
        world: torch.Tensor,
        world_size: Tuple[int] = (3, 3),
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            world (torch.Tensor): The coordinates of the sparse structure.
            world_size (Tuple[int]): The size of the world (H, W).
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        breso = world.shape[-1]
        world_feats = torch.zeros((breso // 2 * (world_size[0] +1), breso // 2 * (world_size[1] +1), breso, flow_model.out_channels)).cuda()
        sampler_params = {**self.slat_sampler_params, **sampler_params}

        for i in range(world_size[0]):
            for j in range(world_size[1]):
                voxels = world[:, :, i*breso//2:i*breso//2+breso, j*breso//2:j*breso//2+breso, :]
                coords = torch.argwhere(voxels > 0)[:, [0, 2, 3, 4]].int()
                if coords.numel() == 0:
                    continue
                slat_0 = world_feats[i*breso//2:i*breso//2+breso, j*breso//2:j*breso//2+breso, :].clone().detach()
                feats = []
                for coord in coords:
                    _, x, y, z = coord
                    feats.append(slat_0[x, y, z])
                feats = torch.stack(feats).cuda()
                noisy_feats = torch.randn_like(feats).cuda()
                mask = torch.ones(feats.shape[0], 1).cuda()
                if i == 0 and j == 0:
                    pass
                elif i == 0:
                    mask[coords[:, 2] < breso * 3 // 8] = 0
                elif j == 0:
                    mask[coords[:, 1] < breso * 3 // 8] = 0
                else:
                    mask[coords[:, 2] < breso * 3 // 8] = 0
                    mask[coords[:, 1] < breso * 3 // 8] = 0
                masked_feats = feats * (1 - mask)
                slat = sp.SparseTensor(
                    feats=feats,
                    coords=coords,
                )
                noisy_slat = sp.SparseTensor(
                    feats=noisy_feats,
                    coords=coords,
                )
                masked_slat = sp.SparseTensor(
                    feats=masked_feats,
                    coords=coords,
                )
                slat_mask = sp.SparseTensor(
                    feats=mask,
                    coords=coords,
                )

                slat = self.slat_sampler.sample(
                    flow_model,
                    noisy_slat,
                    slat,
                    masked_slat,
                    slat_mask,
                    **cond,
                    **sampler_params,
                    strength=1.0,
                    verbose=True,
                ).samples
                for idx, coord in enumerate(coords):
                    _, x, y, z = coord
                    world_feats[i*breso//2 + x, j*breso//2 + y, z] = slat.feats[idx].clone().detach()

        return world_feats

    @torch.no_grad()
    def run(
        self,
        world_size: Tuple[int] = (3, 3),
        seed: Optional[int] = None,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['gaussian', 'mesh'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            world_size (Tuple[int]): The size of the world (H, W).
            seed (Optional[int]): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([self.text_prompt])
        if seed is not None:
            torch.manual_seed(seed)

        coarse_world = self.sample_sparse_structure_coarse(cond, world_size, sparse_structure_sampler_params)
        fine_world = self.sample_sparse_structure_fine(cond, coarse_world, world_size, sparse_structure_sampler_params)
        world_feats = self.sample_slat(cond, fine_world, world_size, slat_sampler_params)

        return self.decode_slat(fine_world, world_feats, world_size, formats)
