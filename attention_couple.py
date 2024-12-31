# Heavily derived work for attention couple calculation and patch - GPL-3.0 license
# Credits to https://github.com/laksjdjf/cgem156-ComfyUI/tree/main/scripts/attention_couple
# and https://github.com/Haoming02/sd-forge-couple

import torch
import torch.nn.functional as F
import math
from base64 import b64decode as decode
from io import BytesIO as bIO
from PIL import Image
import numpy as np
from functools import reduce


def repeat_div(value: int, iterations: int) -> int:
    for _ in range(iterations):
        value = math.ceil(value / 2)

    return value


def get_mask(mask, batch_size, num_tokens, original_shape):
    image_width: int = original_shape[3]
    image_height: int = original_shape[2]

    scale = math.ceil(math.log2(math.sqrt(image_height * image_width / num_tokens)))
    size = (repeat_div(image_height, scale), repeat_div(image_width, scale))

    num_conds = mask.shape[0]
    mask_downsample = F.interpolate(mask, size=size, mode="nearest")
    mask_downsample = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(
        batch_size, dim=0
    )

    return mask_downsample


def lcm(a, b):
    return a * b // math.gcd(a, b)


def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm


def b64image2tensor(img: str, width: int, height: int) -> torch.Tensor:
    image_bytes = decode(img)
    image = Image.open(bIO(image_bytes)).convert("L")

    if image.width != width or image.height != height:
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)

    return image


class AttentionCoupleRegion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "mask": ("MASK",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("ATTENTION_COUPLE_REGION",)
    RETURN_NAMES = ("region",)
    FUNCTION = "attention_couple_region"
    CATEGORY = "A8R8"

    def attention_couple_region(self, cond, mask, weight):
        return ({"cond": cond, "mask": mask, "weight": weight},)


class AttentionCoupleRegions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "region_1": ("ATTENTION_COUPLE_REGION", ),
                "region_2": ("ATTENTION_COUPLE_REGION", ),
            },

        }

    RETURN_TYPES = ("ATTENTION_COUPLE_REGION",)
    RETURN_NAMES = ("regions",)

    FUNCTION = "attention_couple_regions"
    CATEGORY = "A8R8"

    def attention_couple_regions(self, inputcount, **kwargs):
        regions = kwargs.get("regions")

        if regions:
            assert isinstance(
                regions, list
            ), "Regions has to be a list of regions, a single item was passed to regions."

        regions_first = kwargs["region_1"]
        regions = [kwargs.get(f"region_{i}") for i in range(1, inputcount+1)] + (
            regions if regions else []
        )

        flattened_regions = reduce(
            lambda acc, region: acc + region
            if isinstance(region, list)
            else acc + [region]
            if region
            else acc,
            regions,
            [],
        )
        return (flattened_regions,)


class AttentionCouple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "base_prompt": ("CONDITIONING",),
                "global_prompt_weight": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Base prompt strength.",
                    },
                ),
                "regions": (
                    "ATTENTION_COUPLE_REGION",
                    {
                        "tooltip": "Accepts Attention Couple Regions or a single Attention Couple Region directly."
                    },
                ),
                "width": ("INT", {"default": 1024, "min": 8, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "step": 8}),
            },
        }

    # INPUT_IS_LIST = True #(False, False, True,)
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "attention_couple"
    CATEGORY = "A8R8"

    def attention_couple(
        self,
        model,
        global_prompt_weight,
        base_prompt,
        height,
        width,
        regions,
        **kwargs,
    ):
        base_mask = torch.zeros((height, width)).unsqueeze(0)
        global_mask = (torch.ones((height, width)) * global_prompt_weight).unsqueeze(0)

        new_model = model.clone()

        if not isinstance(regions, list):
            regions = [regions]

        num_conds = len(regions) + 1

        mask = [base_mask] + [
            global_mask
            if i == 0
            else F.interpolate(
                regions[i - 1]["mask"].unsqueeze(0),
                size=(height, width),
                mode="nearest-exact",
            ).squeeze(0)
            * regions[i - 1]["weight"]
            for i in range(0, num_conds)
        ]
        mask = torch.stack(mask, dim=0)
        assert mask.sum(dim=0).min() > 0, "There are areas that are zero in all masks."
        self.mask = mask / mask.sum(dim=0, keepdim=True)

        self.conds = [
            base_prompt[0][0] if i == 0 else regions[i - 1]["cond"][0][0]
            for i in range(0, num_conds)
        ]
        num_tokens = [cond.shape[1] for cond in self.conds]

        num_conds += 1

        def attn2_patch(q, k, v, extra_options):
            assert k.mean() == v.mean(), "k and v must be the same."
            device, dtype = q.device, q.dtype

            if self.conds[0].device != device:
                self.conds = [cond.to(device, dtype=dtype) for cond in self.conds]
            if self.mask.device != device:
                self.mask = self.mask.to(device, dtype=dtype)

            cond_or_unconds = extra_options["cond_or_uncond"]
            num_chunks = len(cond_or_unconds)
            self.batch_size = q.shape[0] // num_chunks
            q_chunks = q.chunk(num_chunks, dim=0)
            k_chunks = k.chunk(num_chunks, dim=0)
            lcm_tokens = lcm_for_list(num_tokens + [k.shape[1]])
            conds_tensor = torch.cat(
                [
                    cond.repeat(self.batch_size, lcm_tokens // num_tokens[i], 1)
                    for i, cond in enumerate(self.conds)
                ],
                dim=0,
            )

            qs, ks = [], []
            for i, cond_or_uncond in enumerate(cond_or_unconds):
                k_target = k_chunks[i].repeat(1, lcm_tokens // k.shape[1], 1)
                if cond_or_uncond == 1:  # uncond
                    qs.append(q_chunks[i])
                    ks.append(k_target)
                else:
                    qs.append(q_chunks[i].repeat(num_conds, 1, 1))
                    ks.append(torch.cat([k_target, conds_tensor], dim=0))

            qs = torch.cat(qs, dim=0)
            ks = torch.cat(ks, dim=0).to(k)

            if qs.size(0) % 2 == 1:
                empty = torch.zeros_like(qs[0]).unsqueeze(0)
                qs = torch.cat((qs, empty), dim=0)

                empty2 = torch.zeros_like(ks[0]).unsqueeze(0)
                ks = torch.cat((ks, empty2), dim=0)

            return qs, ks, ks

        def attn2_output_patch(out, extra_options):
            cond_or_unconds = extra_options["cond_or_uncond"]
            mask_downsample = get_mask(
                self.mask,
                self.batch_size,
                out.shape[1],
                extra_options["original_shape"],
            )
            outputs = []
            pos = 0
            for cond_or_uncond in cond_or_unconds:
                if cond_or_uncond == 1:  # uncond
                    outputs.append(out[pos : pos + self.batch_size])
                    pos += self.batch_size
                else:
                    masked_output = (
                        out[pos : pos + num_conds * self.batch_size] * mask_downsample
                    ).view(num_conds, self.batch_size, out.shape[1], out.shape[2])
                    masked_output = masked_output.sum(dim=0)
                    outputs.append(masked_output)
                    pos += num_conds * self.batch_size
            return torch.cat(outputs, dim=0)

        new_model.set_model_attn2_patch(attn2_patch)
        new_model.set_model_attn2_output_patch(attn2_output_patch)

        return (new_model,)
