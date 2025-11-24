"""Library of data augmentations for training adapted from deepsat"""
import math
import random
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# --- Add somewhere near your other transforms ---
class StashLabelBeforeDownsample:
    """
    Copy the current label tensor (before any spatial downsampling) into sample['labels_raw'].
    Place this transform immediately BEFORE any transform that changes label resolution.
    """
    def __init__(self, key_in: str = "labels", key_out: str = "labels_raw"):
        self.key_in = key_in
        self.key_out = key_out

    def __call__(self, sample):
        # sample could be dict or namespace; handle dict safely
        if isinstance(sample, dict) and self.key_in in sample and self.key_out not in sample:
            # save a detached clone so later transforms can't mutate it
            x = sample[self.key_in]
            try:
                sample[self.key_out] = x.clone()
            except AttributeError:
                # if not tensor, best-effort shallow copy
                sample[self.key_out] = x
        return sample


class DebugTapOnce:
    _done = False
    def __init__(self, func):
        self.func = func
    def __call__(self, sample):
        if not DebugTapOnce._done:
            try:
                self.func(sample)
            finally:
                DebugTapOnce._done = True
        return sample

def band_order_probe(sample):
    # sample["inputs"]: [T, C, H, W] in MS-CLIP order after reorder, reflectance scale
    x = sample["inputs"]                      # [T,C,H,W]
    t, c, h, w = x.shape
    xr = x.reshape(-1, c, h, w)               # [T, C, H, W] → [N, C, H, W]

    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = [xr[:, i] for i in range(10)]

    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    def corr2(a,b):
        am = a.mean(dim=(1,2), keepdim=True); bm = b.mean(dim=(1,2), keepdim=True)
        an = a - am; bn = b - bm
        num = (an*bn).mean(dim=(1,2))
        den = (an.pow(2).mean(dim=(1,2)).sqrt() * bn.pow(2).mean(dim=(1,2)).sqrt() + 1e-6)
        return (num/den).mean().item()

    print("[DEBUG] Reflectance (pre-normalize)")
    print("NDVI mean/std/min/max:",
          ndvi.mean().item(), ndvi.std().item(), ndvi.min().item(), ndvi.max().item())
    print("corr(B8,B8A):", corr2(B8,B8A))
    print("corr(B11,B12):", corr2(B11,B12),
          "  corr(B2,B11):", corr2(B2,B11), "  corr(B2,B12):", corr2(B2,B12))
    ch_means = xr.mean(dim=(0,2,3))
    print("Per-channel reflectance means (B2..B12):", [round(v.item(),2) for v in ch_means])

class StatsTap(object):
    def __init__(self, tag: str):
        self.tag = tag

    def __call__(self, sample):
        x = sample["inputs"]  # tensor with shape [T,C,H,W] or [T,H,W,C] or [C,H,W]

        # Standardize to [T, C, H, W] for stats
        if x.ndim == 3:               # [C,H,W]
            t = x.unsqueeze(0)        # [1,C,H,W]
        elif x.ndim == 4:
            # If channel-last (THWC), permute to TCHW
            if x.shape[-1] <= 20 and x.shape[1] in (224, 256):   # [T,H,W,C]
                t = x.permute(0, 3, 1, 2).contiguous()
            else:
                t = x  # already [T,C,H,W]
        else:
            raise ValueError(f"[DebugStats] Unexpected shape: {x.shape}")

        # Reduce over time and spatial dims
        cmins = torch.amin(t, dim=(0, 2, 3))
        cmaxs = torch.amax(t, dim=(0, 2, 3))
        cmean = t.mean(dim=(0, 2, 3))
        cstd  = t.std(dim=(0, 2, 3), unbiased=False)

        # Print first 10 channels to keep logs readable
        def tolist(v): 
            return [round(float(z), 3) for z in v[:10]]
        print(f"[{self.tag}] shape={tuple(x.shape)}",
              f"min[:10]={tolist(cmins)} max[:10]={tolist(cmaxs)}",
              f"mean[:10]={tolist(cmean)} std[:10]={tolist(cstd)}")
        return sample



class DOSApproxL1CtoL2A:
    """
    Dark Object Subtraction (DOS) per time-step on reflectance tensors (pre-normalization).
    Expects sample['inputs'] in [T, C, H, W] with MS-CLIP order:
    [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]
    """
    def __init__(self, p=1.0, scales=None):

        default_scales = torch.tensor([1.00, 0.85, 0.70, 0.10, 0.05, 0.05, 0.10, 0.08, 0.02, 0.02])
        self.p = p
        self.scales = None if scales is None else torch.tensor(scales, dtype=torch.float32)
        self.default_scales = default_scales

    def __call__(self, sample):
        x = sample["inputs"]  # [T, C, H, W], reflectance units
        device = x.device if isinstance(x, torch.Tensor) else None
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        T, C, H, W = x.shape
        s = (self.scales.to(x.device) if self.scales is not None else self.default_scales.to(x.device)).view(1, C, 1, 1)

        x_ = x.view(T, C, -1)
        k = max(1, int((self.p/100.0) * x_.shape[-1]))
        haze_vals, _ = torch.kthvalue(x_[:, 0, :], k=k, dim=-1)   # [T]
        haze = haze_vals.view(T, 1, 1, 1) * s                      # [T, C, 1, 1] scaled per band

        x = (x - haze).clamp_min_(0.0)

        sample["inputs"] = x
        return sample


class ToTensorMSCLIP(object):
    """
    Convert S2 arrays to Tensors in MS-CLIP style.
    Keeps raw reflectance values (no min-max normalization),
    converts to float32, and permutes to [C, H, W].
    """

    def __init__(self, with_loc: bool = False):
        self.with_loc = with_loc

    def _to_chw(self, arr: np.ndarray) -> torch.Tensor:
        # Handle single or multi-band arrays
        if arr.ndim == 2:
            arr = arr[None, :, :]  # [1, H, W]
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))  # [C, H, W]
        return torch.from_numpy(arr.astype(np.float32))

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        tensor_sample = {}
        for key in ["10x", "20x", "60x"]:
            if sample.get(key) is not None:
                tensor_sample[key] = self._to_chw(sample[key])
            else:
                tensor_sample[key] = None

        tensor_sample["labels"] = torch.tensor(sample["labels"], dtype=torch.float32).unsqueeze(0)
        tensor_sample["doy"] = torch.tensor(sample["doy"], dtype=torch.float32)
        if self.with_loc:
            tensor_sample["loc"] = torch.tensor(sample["loc"], dtype=torch.float32)
        return tensor_sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, with_loc: bool = False):
        self.with_loc = with_loc

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        tensor_sample = {}
        tensor_sample["10x"] = torch.tensor(sample["10x"]).to(torch.float32) if sample["10x"] is not None else None
        tensor_sample["20x"] = torch.tensor(sample["20x"]).to(torch.float32) if sample["20x"] is not None else None
        tensor_sample["60x"] = torch.tensor(sample["60x"]).to(torch.float32) if sample["60x"] is not None else None
        tensor_sample["labels"] = torch.tensor(sample["labels"]).to(torch.float32).unsqueeze(0)  # 1, H, W
        tensor_sample["doy"] = torch.tensor(sample["doy"]).to(torch.float32)

        if self.with_loc:
            tensor_sample["loc"] = torch.tensor(sample["loc"]).to(torch.float32)  # Like labels 2, H, W

        return tensor_sample


class EnvToTensor(object):
    """
    Rescale the image in a sample to a given square side
    items in  : modis11, modis13_15, era5, cds, doy, (labels)
    items out : modis11, modis13_15, era5, cds, doy, (labels)
    """

    def __init__(self, with_loc: bool = False):
        self.with_loc = with_loc

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        tensor_sample = {}
        tensor_sample["modis11"] = (
            torch.tensor(sample["modis11"]).to(torch.float32) if sample["modis11"] is not None else None
        )
        tensor_sample["modis11_mask"] = (
            torch.tensor(sample["modis11_mask"]).to(torch.float32) if sample["modis11_mask"] is not None else None
        )
        tensor_sample["modis13_15"] = (
            torch.tensor(sample["modis13_15"]).to(torch.float32) if sample["modis13_15"] is not None else None
        )
        tensor_sample["modis13_15_mask"] = (
            torch.tensor(sample["modis13_15_mask"]).to(torch.float32) if sample["modis13_15_mask"] is not None else None
        )
        tensor_sample["era5"] = torch.tensor(sample["era5"]).to(torch.float32) if sample["era5"] is not None else None
        tensor_sample["era5_mask"] = (
            torch.tensor(sample["era5_mask"]).to(torch.float32) if sample["era5_mask"] is not None else None
        )
        tensor_sample["cds"] = torch.tensor(sample["cds"]).to(torch.float32) if sample["cds"] is not None else None
        tensor_sample["cds_mask"] = (
            torch.tensor(sample["cds_mask"]).to(torch.float32) if sample["cds_mask"] is not None else None
        )
        tensor_sample["tab_doy"] = torch.tensor(sample["tab_doy"]).to(torch.float32)

        if "labels" in sample:
            tensor_sample["labels"] = torch.tensor(sample["labels"]).to(torch.float32).unsqueeze(0)

        if self.with_loc:
            raise NotImplementedError("Location data not available for Environment Canada data")

        return tensor_sample


class Rescale(object):
    """
    Rescale the image in a sample to a given square side
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, output_size: Tuple[int, int]):
        assert isinstance(output_size, (tuple,))
        self.new_h, self.new_w = output_size

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for inputc in ["20x", "60x"]:
            if sample[inputc] is not None:
                sample[inputc] = self.rescale_3d_map(sample[inputc], mode="bilinear")

        return sample

    def rescale_3d_map(self, img: torch.Tensor, mode: str) -> torch.Tensor:
        """Rescale Img"""
        img = F.interpolate(img, size=(self.new_h, self.new_w), mode=mode)
        return img


class EnvRescale(object):
    """
    Rescale the env tile in a sample to a given square side
    items in  : modis11, modis13_15, era5, cds, doy
    items out : modis11, modis13_15, era5, cds, doy
    """

    def __init__(self, mid_size: int, low_size: int):
        self.mid_size = mid_size
        self.low_size = low_size

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for inputc in ["modis11_mask"]:
            if sample[inputc] is not None:
                sample[inputc] = self.rescale_3d_map(sample[inputc], self.mid_size, mode="nearest")
                sample[inputc] = sample[inputc].to(torch.bool)

        for inputc in ["modis11"]:
            if sample[inputc] is not None:
                if torch.isnan(sample[inputc]).any():
                    sample[inputc] = torch.where(
                        torch.isnan(sample[inputc]), torch.tensor(0.0, device=sample[inputc].device), sample[inputc]
                    )
                    sample[inputc] = self.rescale_3d_map(sample[inputc], self.mid_size, mode="bilinear")
                    sample[inputc][sample[inputc + "_mask"]] = float("nan")
                else:
                    sample[inputc] = self.rescale_3d_map(sample[inputc], self.mid_size, mode="bilinear")

        for inputc in ["cds_mask"]:
            if sample[inputc] is not None:
                sample[inputc] = self.rescale_3d_map(sample[inputc], self.low_size, mode="nearest")
                sample[inputc] = sample[inputc].to(torch.bool)

        for inputc in ["cds"]:
            if sample[inputc] is not None:
                if torch.isnan(sample[inputc]).any():
                    sample[inputc] = torch.where(
                        torch.isnan(sample[inputc]), torch.tensor(0.0, device=sample[inputc].device), sample[inputc]
                    )
                    sample[inputc] = self.rescale_3d_map(sample[inputc], self.low_size, mode="bilinear")
                    sample[inputc][sample[inputc + "_mask"]] = float("nan")

                else:
                    sample[inputc] = self.rescale_3d_map(sample[inputc], self.low_size, mode="bilinear")

        return sample

    def rescale_3d_map(self, img: torch.Tensor, size: int, mode: str):
        """Rescale Img"""
        img = F.interpolate(img, size=(size, size), mode=mode)
        return img


class Concat(object):
    """
    Concat all inputs
    items in  : x10, x20, x60, day, year, labels
    items out : inputs, labels
    """

    def __init__(self, concat_keys: List[str]):
        self.concat_keys = concat_keys

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        try:
            conc_keys = [key for key in self.concat_keys if sample[key] is not None]
            inputs = torch.cat([sample[key] for key in conc_keys], dim=1)
            sample["inputs"] = inputs
            sample = {key: sample[key] for key in sample.keys() if key not in self.concat_keys}
        except:
            print([("conc", key, sample[key].shape) for key in sample.keys()])
        return sample


class EnvConcat(object):
    """
    Concat all inputs
    items in  : modis11, modis13_15, era5, cds, doy
    items out : mid_inputs, low_inputs
    """

    def __init__(self, mid_keys: List[str], low_keys: List[str]):
        self.mid_keys = mid_keys
        self.low_keys = low_keys

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        try:
            mid_keys = [key for key in self.mid_keys if sample[key] is not None]
            mid_mask_keys = [f"{key}_mask" for key in self.mid_keys if sample[key] is not None]
            low_keys = [key for key in self.low_keys if sample[key] is not None]
            low_mask_keys = [f"{key}_mask" for key in self.low_keys if sample[key] is not None]

            mid_inputs = torch.cat([sample[key] for key in mid_keys], dim=1)
            mid_inputs_mask = torch.cat([sample[key].to(torch.bool) for key in mid_mask_keys], dim=1)
            low_inputs = torch.cat([sample[key] for key in low_keys], dim=1)
            low_inputs_mask = torch.cat([sample[key].to(torch.bool) for key in low_mask_keys], dim=1)
            sample["mid_inputs"] = mid_inputs
            sample["mid_inputs_mask"] = mid_inputs_mask
            sample["low_inputs"] = low_inputs
            sample["low_inputs_mask"] = low_inputs_mask

            sample = {
                key: sample[key]
                for key in sample.keys()
                if key not in mid_keys + low_keys + mid_mask_keys + low_mask_keys
            }
        except:
            print([("conc", key, sample[key].shape) for key in sample.keys()])

        return sample


class Normalize(object):
    """
    Normalize inputs as in https://arxiv.org/pdf/1802.02080.pdf
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, mean: float, std: float, eps: float = 1e-7):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample["inputs"] = (sample["inputs"] - self.mean) / self.std
        sample["doy"] = sample["doy"] / (366 + self.eps)  # We do have bisextile year in our data
        return sample

class NormalizeMSCLIP(object):
    def __init__(self, mean: float, std: float,eps: float = 1e-7):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.eps = eps

    def __call__(self, sample):
        sample["inputs"] = (sample["inputs"] - self.mean) / self.std
        sample["doy"] = sample["doy"] / (366 + self.eps)
        return sample

class Crop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, img_size: int, crop_size: int, random: bool =False, ground_truths: List[str] = [], with_loc: bool = False):
        self.img_size = img_size
        self.crop_size = crop_size
        self.random = random
        if not random:
            self.top = int((img_size - crop_size) / 2)
            self.left = int((img_size - crop_size) / 2)
        self.ground_truths = ground_truths
        self.with_loc = with_loc

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.random:
            top = torch.randint(self.img_size - self.crop_size, (1,))[0]
            left = torch.randint(self.img_size - self.crop_size, (1,))[0]
        else:  # center
            top = self.top
            left = self.left
        sample["inputs"] = sample["inputs"][:, :, top : top + self.crop_size, left : left + self.crop_size]
        for gt in self.ground_truths:
            sample[gt] = sample[gt][:, top : top + self.crop_size, left : left + self.crop_size]

        if self.with_loc:
            sample["loc"] = sample["loc"][:, top : top + self.crop_size, left : left + self.crop_size]

        return sample


class ResizedCrop(object):
    """Resized crop the image in a sample."""

    def __init__(self, out_size: int, scale: float, prob: float, ground_truths: List[str] = [], with_loc: bool = False):
        self.out_size = out_size
        self.scale = scale
        self.prob = prob
        self.ground_truths = ground_truths
        self.with_loc = with_loc

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if random.random() < self.prob:
            # Cropping
            scale = random.uniform(self.scale[0], self.scale[1])
            crop_size = int(sample["inputs"].shape[2] * scale)
            img_size = sample["inputs"].shape[2]
            sample = Crop(img_size=img_size, crop_size=crop_size, random=True, ground_truths=self.ground_truths)(sample)

            # Resizing
            sample["inputs"] = F.interpolate(sample["inputs"], size=(self.out_size, self.out_size), mode="bilinear")
            sample["labels"] = F.interpolate(
                sample["labels"].unsqueeze(0), size=(self.out_size, self.out_size), mode="nearest-exact"
            ).squeeze(0)

            if self.with_loc:
                sample["loc"] = F.interpolate(
                    sample["loc"].unsqueeze(0), size=(self.out_size, self.out_size), mode="bilinear"
                ).squeeze(0)

        return sample


class HVFlip(object):
    """
    random horizontal, vertical flip
    items in  : inputs, labels
    items out : inputs, labels
    """

    def __init__(self, hflip_prob: float, vflip_prob: float, with_loc: bool = False):
        assert isinstance(hflip_prob, (float,))
        assert isinstance(vflip_prob, (float,))
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.with_loc = with_loc

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:

        # Extract the image and environmental data
        if isinstance(sample, (list, tuple)):
            img_sample = sample[0]
            env_sample = sample[1]
        else:
            if "inputs" in sample:
                img_sample = sample
                env_sample = None
            else:
                img_sample = None
                env_sample = sample

        # Apply the flip
        if random.random() < self.hflip_prob:
            if img_sample is not None:
                img_sample["inputs"] = torch.flip(img_sample["inputs"], (3,))
                img_sample["labels"] = torch.flip(img_sample["labels"], (2,))
                if self.with_loc:
                    img_sample["loc"] = torch.flip(img_sample["loc"], (2,))
            if env_sample is not None:
                env_sample["mid_inputs"] = torch.flip(env_sample["mid_inputs"], (3,))
                env_sample["low_inputs"] = torch.flip(env_sample["low_inputs"], (3,))
                env_sample["mid_inputs_mask"] = torch.flip(env_sample["mid_inputs_mask"], (3,))
                env_sample["low_inputs_mask"] = torch.flip(env_sample["low_inputs_mask"], (3,))
                if "labels" in env_sample:
                    env_sample["labels"] = torch.flip(env_sample["labels"], (2,))

        if random.random() < self.vflip_prob:

            if img_sample is not None:
                img_sample["inputs"] = torch.flip(img_sample["inputs"], (2,))
                img_sample["labels"] = torch.flip(img_sample["labels"], (1,))
                if self.with_loc:
                    img_sample["loc"] = torch.flip(img_sample["loc"], (1,))
            if env_sample is not None:
                env_sample["mid_inputs"] = torch.flip(env_sample["mid_inputs"], (2,))
                env_sample["low_inputs"] = torch.flip(env_sample["low_inputs"], (2,))
                env_sample["mid_inputs_mask"] = torch.flip(env_sample["mid_inputs_mask"], (2,))
                env_sample["low_inputs_mask"] = torch.flip(env_sample["low_inputs_mask"], (2,))
                if "labels" in env_sample:
                    env_sample["labels"] = torch.flip(env_sample["labels"], (1,))

        if env_sample is None:
            return img_sample
        if img_sample is None:
            return env_sample
        return img_sample, env_sample


class MixHVFlip(object):
    """
    random horizontal, vertical flip
    items in  : inputs, env_inputs, labels
    items out : inputs, env_inputs, labels
    """

    def __init__(self, hflip_prob: float, vflip_prob: float, with_loc: bool = False):
        assert isinstance(hflip_prob, (float,))
        assert isinstance(vflip_prob, (float,))
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.with_loc = with_loc

    def __call__(self, img_sample: Dict[str, torch.Tensor], env_sample: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    # Attention here we have data in the format THWC and not TCHW because applied after ToTHWC

        # Apply the flip
        if random.random() < self.hflip_prob:
            if img_sample is not None:
                img_sample["inputs"] = torch.flip(img_sample["inputs"], (2,))
                img_sample["labels"] = torch.flip(img_sample["labels"], (1,))
                if self.with_loc:
                    img_sample["loc"] = torch.flip(img_sample["loc"], (1,))
            if env_sample is not None:
                env_sample["mid_inputs"] = torch.flip(env_sample["mid_inputs"], (2,))
                env_sample["low_inputs"] = torch.flip(env_sample["low_inputs"], (2,))
                env_sample["mid_inputs_mask"] = torch.flip(env_sample["mid_inputs_mask"], (2,))
                env_sample["low_inputs_mask"] = torch.flip(env_sample["low_inputs_mask"], (2,))
                if "labels" in env_sample:
                    env_sample["labels"] = torch.flip(env_sample["labels"], (1,))

        if random.random() < self.vflip_prob:

            if img_sample is not None:
                img_sample["inputs"] = torch.flip(img_sample["inputs"], (1,))
                img_sample["labels"] = torch.flip(img_sample["labels"], (0,))
                if self.with_loc:
                    img_sample["loc"] = torch.flip(img_sample["loc"], (0,))
            if env_sample is not None:
                env_sample["mid_inputs"] = torch.flip(env_sample["mid_inputs"], (1,))
                env_sample["low_inputs"] = torch.flip(env_sample["low_inputs"], (1,))
                env_sample["mid_inputs_mask"] = torch.flip(env_sample["mid_inputs_mask"], (1,))
                env_sample["low_inputs_mask"] = torch.flip(env_sample["low_inputs_mask"], (1,))
                if "labels" in env_sample:
                    env_sample["labels"] = torch.flip(env_sample["labels"], (0,))

        return img_sample, env_sample


class GaussianNoise(object):
    """Add gaussian noise to the inputs"""

    def __init__(self, var_limit: float, p: float, per_channel: bool = True):
        self.var_limit = var_limit
        self.p = p
        self.per_channel = per_channel

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            if "inputs" in sample:
                noise = torch.randn_like(sample["inputs"]) * (var**0.5)
                sample["inputs"] = sample["inputs"] + noise
            if "mid_inputs" in sample:
                noise = torch.randn_like(sample["mid_inputs"]) * (var**0.5)
                sample["mid_inputs"] = sample["mid_inputs"] + noise
            if "low_inputs" in sample:
                noise = torch.randn_like(sample["low_inputs"]) * (var**0.5)
                sample["low_inputs"] = sample["low_inputs"] + noise
        return sample


class DownSampleLab(object):
    """Downsample the labels"""

    def __init__(self, out_H: int, out_W: int):
        self.out_H = out_H
        self.out_W = out_W

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        kernel_size = (round(sample["labels"].shape[1] / self.out_H), round(sample["labels"].shape[2] / self.out_W))
        sample["labels"] = F.max_pool2d(sample["labels"].unsqueeze(0), kernel_size=kernel_size, padding=1).squeeze(0)
        return sample


class RandomTemporalDrop(object):
    """Drop Timesteps randomly except the first one"""

    def __init__(self, drop_prob: float):
        self.drop_prob = drop_prob

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.drop_prob:
            timesteps = sample["inputs"].shape[0]
            drop_idx = random.randint(1, timesteps - 1)
            sample["inputs"] = torch.cat([sample["inputs"][:drop_idx], sample["inputs"][drop_idx + 1 :]], dim=0)
            sample["doy"] = torch.cat([sample["doy"][:drop_idx], sample["doy"][drop_idx + 1 :]], dim=0)
        return sample


class ReorderBands(object):
    def __init__(self, order):
        self.order = order
    def __call__(self, sample):
        x = sample["inputs"]
        if x.ndim == 3:
            x = x[self.order, ...]
        elif x.ndim == 4:
            x = x[:, self.order, ...]
        else:
            raise ValueError(f"Unexpected shape {x.shape}")
        sample["inputs"] = x
        return sample


class ToTHWC(object):
    """
    Convert Tensors to THWC.
    items in  : inputs, unk_masks, labels
    items out : inputs, unk_masks, labels
    """

    def __call__(Felf, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = sample["inputs"]
        if x.ndim == 4: 
            if x.shape[1] in (224, 256): 
                x = x.permute(0, 3, 1, 2).contiguous()  # [T,H,W,C] → [T,C,H,W]

        elif x.ndim == 5:  
            if x.shape[2] in (224, 256): 
                x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B,T,H,W,C] → [B,T,C,H,W]
        sample["inputs"] = x
        
        #sample["labels"] = sample["labels"].permute(1, 2, 0)
        #sample["unk_masks"] = sample["unk_masks"].permute(1, 2, 0)
        return sample


class EnvToTHWC(object):
    """
    Convert Tensors to THWC.
    items in  : mid_inputs, low_inputs, labels
    items out : mid_inputs, low_inputs, labels
    """

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample["mid_inputs"] = sample["mid_inputs"].permute(0, 2, 3, 1)
        sample["low_inputs"] = sample["low_inputs"].permute(0, 2, 3, 1)
        sample["mid_inputs_mask"] = sample["mid_inputs_mask"].permute(0, 2, 3, 1)
        sample["low_inputs_mask"] = sample["low_inputs_mask"].permute(0, 2, 3, 1)

        if "labels" in sample:
            sample["labels"] = sample["labels"].permute(1, 2, 0)

        return sample


class TabToTensor(object):
    """
    Convert tabular ndarrays to Tensors.
    """

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensor_sample = {}
        tensor_sample["tab_inputs"] = torch.tensor(sample["tab"]).to(
            torch.float32
        )  # Size is T*C with C = N Cols and T = TS length
        tensor_sample["tab_doy"] = (
            torch.tensor(np.array(sample["tab_doy"])).unsqueeze(-1).to(torch.float32)
        )  # Size is T*1
        tensor_sample["mask"] = torch.tensor(sample["mask"]).to(
            torch.float32
        )  # Size is T*C with C = N Cols and T = TS length
        return tensor_sample


class TabNormalize(object):
    """
    Normalize tabular inputs.
    """

    def __init__(self, mean: float, std: float, eps: float = 1e-7):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample: Dict[str, torch.Tensor]):
        sample["tab_inputs"] = (sample["tab_inputs"] - self.mean) / self.std
        sample["tab_doy"] = sample["tab_doy"] / (366 + self.eps)  # We do have bisextile year in our data
        return sample


class EnvNormalize(object):
    """
    Normalize Environmental inputs.
    """

    def __init__(self, mid_mean: float, mid_std: float, low_mean: float, low_std: float, eps: float = 1e-7):
        self.mid_mean = mid_mean
        self.mid_std = mid_std
        self.low_mean = low_mean
        self.low_std = low_std
        self.eps = eps

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample["mid_inputs"] = (sample["mid_inputs"] - self.mid_mean) / self.mid_std
        sample["low_inputs"] = (sample["low_inputs"] - self.low_mean) / self.low_std
        sample["tab_doy"] = sample["tab_doy"] / (366 + self.eps)
        return sample


class TabTileDates(object):
    """
    Concatenate inputs, and tab_doy
    """

    def __call__(self, sample: Dict[str, torch.Tensor]):
        sample["tab_inputs"] = torch.cat((sample["tab_inputs"], sample["tab_doy"]), dim=1)  # T*(C + 1)
        del sample["tab_doy"]
        return sample


class EnvTileDates(object):
    """
    Tile a 1d array of tab_doy to height (H), width (W) of env tiles.
    """

    def __init__(self, mid_H: int, mid_W: int, low_H: int, low_W: int, doy_bins: List[int] = None):
        assert isinstance(mid_H, (int,))
        assert isinstance(mid_W, (int,))
        assert isinstance(low_H, (int,))
        assert isinstance(low_W, (int,))

        self.mid_H = mid_H
        self.mid_W = mid_W
        self.low_H = low_H
        self.low_W = low_W
        self.doy_bins = doy_bins

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mid_doy, low_doy = self.repeat(sample["tab_doy"], binned=self.doy_bins is not None)
        sample["mid_inputs"] = torch.cat((sample["mid_inputs"], mid_doy), dim=1)
        sample["low_inputs"] = torch.cat((sample["low_inputs"], low_doy), dim=1)
        del sample["tab_doy"]
        return sample

    def repeat(self, tensor: torch.Tensor, binned: bool = False):
        if binned:
            mid_out = tensor.unsqueeze(1).unsqueeze(1).repeat(1, self.mid_H, self.mid_W, 1)  # .permute(0, 2, 3, 1)
            low_out = tensor.unsqueeze(1).unsqueeze(1).repeat(1, self.low_H, self.low_W, 1)  # .permute(0, 2, 3, 1)
        else:
            mid_out = tensor.repeat(1, self.mid_H, self.mid_W, 1).permute(3, 0, 1, 2)
            low_out = tensor.repeat(1, self.low_H, self.low_W, 1).permute(3, 0, 1, 2)
        return mid_out, low_out


class CutOrPad(object):
    """
    Pad series with zeros (matching series elements) to a max sequence length or cut sequential parts
    items in  : inputs, *inputs_backward, labels
    items out : inputs, *inputs_backward, labels, seq_lengths
    """

    def __init__(self, max_seq_len: int, sampling_type: str, mode: str = "image"):
        assert isinstance(max_seq_len, (int, tuple))
        self.max_seq_len = max_seq_len
        self.sampling_type = sampling_type
        self.mode = mode

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if self.mode == "image":
            seq_len = deepcopy(sample["inputs"].shape[0])
            sample["inputs"],sample["doy"] = self.pad_or_cut(sample)

            if "inputs_backward" in sample:
                sample["inputs_backward"] = self.pad_or_cut(sample["inputs_backward"])
            if seq_len > self.max_seq_len:
                seq_len = self.max_seq_len
            sample["seq_lengths"] = seq_len
        elif self.mode == "env":
            sample["mid_inputs"], sample["mid_inputs_mask"] = self.pad_or_cut(
                sample["mid_inputs"], sample["mid_inputs_mask"]
            )
            sample["low_inputs"], sample["low_inputs_mask"] = self.pad_or_cut(
                sample["low_inputs"], sample["low_inputs_mask"]
            )
        elif self.mode == "tab":
            sample["tab_inputs"], sample["mask"] = self.pad_or_cut(sample["tab_inputs"], sample["mask"])

        else:
            raise ValueError("mode must be either 'image' or 'env'")
        
        return sample

    def pad_or_cut(self, sample: dict, mask_tensor: torch.Tensor = None, dtype=torch.float32) -> Tuple[torch.Tensor]:
        """Pad or Cut the input tensor to the target size"""
        tensor = sample["inputs"]
        tensorDoy = sample["doy"]
        seq_len = tensor.shape[0]
        diff = self.max_seq_len - seq_len
        if diff > 0:
            tsize = list(tensor.shape)
            tsizeDoy = list(tensorDoy.shape)
            if len(tsize) == 1:
                pad_shape = [diff]
                pad_shapeDoy = [diff]
            else:
                pad_shape = [diff] + tsize[1:]
                pad_shapeDoy = [diff] + tsizeDoy[1:]
            tensor = torch.cat(
                (tensor, torch.zeros(pad_shape, dtype=dtype)), dim=0
            )
            #print("=============>",diff, pad_shape, tensorDoy.shape, tensor.shape)
            tensorDoy = torch.cat(
                (tensorDoy, torch.zeros(pad_shapeDoy, dtype=dtype)), dim=0
            )
            mask_tensor = (
                torch.cat((mask_tensor, torch.zeros(pad_shape, dtype=dtype)), dim=0)
                if mask_tensor is not None
                else None
            )
        elif diff < 0:
            if self.sampling_type == "random":
                random_idx = self.random_subseq(seq_len)
                tensor = tensor[random_idx]
                tensorDoy = tensorDoy[random_idx]
                mask_tensor = mask_tensor[random_idx] if mask_tensor is not None else None
            elif self.sampling_type == "start":
                tensor = tensor[-self.max_seq_len :]
                tensorDoy = tensorDoy[-self.max_seq_len :]
                mask_tensor = mask_tensor[-self.max_seq_len :] if mask_tensor is not None else None
            elif self.sampling_type == "uniform":
                uni_idx = self.uniform_subseq(seq_len)
                tensor = tensor[uni_idx]
                tensorDoy = tensorDoy[uni_idx]
                mask_tensor = mask_tensor[uni_idx] if mask_tensor is not None else None
            else:
                start_idx = torch.randint(seq_len - self.max_seq_len, (1,))[0]
                tensor = tensor[start_idx - self.max_seq_len + 1 : start_idx + 1]
                mask_tensor = (
                    mask_tensor[start_idx - self.max_seq_len + 1 : start_idx + 1] if mask_tensor is not None else None
                )

        if mask_tensor is None:
            return tensor, tensorDoy

        else:
            return tensor, tensorDoy, mask_tensor

    def random_subseq(self, seq_len: int):
        random_integers = torch.randperm(seq_len - 1)[: self.max_seq_len - 1].sort()[0]
        return torch.cat((random_integers, torch.tensor([seq_len - 1])))

    def uniform_subseq(self, seq_len: int):
        """Uniformaly sample between the firs and last tile"""

        # Always include the first and last indices
        indices = [0]

        # Calculate the number of intermediate points to sample
        num_intermediate_points = self.max_seq_len - 2
        step_size = (seq_len - 1) / (num_intermediate_points + 1)

        # Generate the intermediate indices
        for i in range(1, num_intermediate_points + 1):
            intermediate_index = int(round(i * step_size))
            indices.append(intermediate_index)

        # Add the last index
        indices.append(seq_len - 1)
        return torch.tensor(indices, dtype=torch.long)

class TileLocs(object):
    """Add Loc to inputs as radian"""

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        T = sample["inputs"].shape[0]
        loc = torch.tensor(math.pi) * sample["loc"] / 180
        loc = loc.unsqueeze(0)
        loc = loc.repeat(T, 1, 1, 1)
        sample["inputs"] = torch.cat((sample["inputs"], loc), dim=1)
        del sample["loc"]
        return sample
