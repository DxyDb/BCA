("rfft")
import time
import pdb
import math
import warnings
from typing import Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

# from args import *
# args_l = get_args()


def get_circ_index(rc, cc, k, sym=False):
    # print(rc,cc,k)
    if rc % k != 0:
        rc += k-rc % k
    if cc % k != 0:
        cc += k-cc % k
    assert rc % k == 0, f"rc={rc}, k={k}"
    assert cc % k == 0, f"cc={cc}, k={k}"
    rc = int((rc+k-1)/k) * k
    cc = int((cc+k-1)/k) * k
    i = np.arange(0,k,1).reshape([1,k])
    j = np.arange(0,-k,-1).reshape([k,1])
    # to follow the caffe implementation
    #indx = i + j
    indx = (i + j).T
    indx = (indx + k) % k
    m = np.tile(indx, [int(rc/k), int(cc/k)])
    offset = np.arange(0,rc*cc)
    i = (offset // cc) // k
    j = (offset % cc) // k
    offset = (i * cc + j * k).reshape([rc,cc])
    return (m + offset).astype(np.int64)


def block_circ_mv(x, circ_w):  # note this input is circ_w not fft_w
    """
    circ_w: [1, rows, cols, circ]
    x: [batch, in_features]
    rows: number of circulant block per row in weight
    cols: number of circulant block per col in weight
    circ: circulant block size
    """
    #print("circ_w", circ_w.shape)
    #print("x", x.shape)
    _, rows, cols, circ = circ_w.shape
    fft_w = torch.fft.rfft(circ_w, n=circ)
    fft_x = torch.fft.rfft(x.view([-1, 1, cols, circ]), n=circ)
    ty = fft_w * fft_x
    ifft_ty = torch.fft.irfft(ty, n=circ)
    output = ifft_ty.sum(dim=-2).view([-1, rows*circ])
    return output


# Inherit from Function
class BlockCircMV(torch.autograd.Function):


    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(x, w, bias):
        """
        fft_w: [1, r, s, circ]
        fft_x: [b, s * circ]
        circ: circulant block size
        """
        _, r, s, circ = w.shape
        fft_x = torch.fft.rfft(x.view([-1, 1, s, circ]))
        fft_w = torch.fft.rfft(w)
        output = torch.fft.irfft(fft_w * fft_x)
        output = output.sum(dim=2).view([-1, r * circ])
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output


    @staticmethod
    def setup_context(ctx, inputs, output):  # save memory
        x, w, bias = inputs
        _, r, s, circ = w.shape
        fft_x = torch.fft.rfft(x.view([-1, 1, s, circ]))
        fft_w = torch.fft.rfft(w)
        ctx.save_for_backward(fft_x, fft_w, bias)


    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        fft_x, fft_w, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        _, rows, cols, _ = fft_w.shape
        circ = grad_output.shape[1] // rows
        fft_o = torch.fft.rfft(grad_output.view([-1, rows, 1, circ]))
        # y = x @ mat.T = IDFT @ diag(DFT(v)) @ DFT @ x = IDFT @ diag(DFT(x)) @ DFT @ v
        # dy/dx = IDFT @ diag(DFT(v[0,n-1,...1])) @ DFT @ dl/dy
        # dy/dv = IDFT @ diag(DFT(x[0,n-1,...1])) @ DFT @ dl/dy
        if ctx.needs_input_grad[0]:
            grad_input = torch.fft.irfft(torch.conj(fft_w) * fft_o).sum(dim=1)
            grad_input = grad_input.view([-1, cols * circ])
        if ctx.needs_input_grad[1]:
            grad_weight = torch.fft.irfft(torch.conj(fft_x) * fft_o)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

        
class CirculantLayer(BaseTunerLayer):
    adapter_layer_names = ["circulant"]
    other_param_names = ("block_size",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        
        self.base_layer = base_layer
        self.block_size = {}
        self.circulant = nn.ParameterDict({})
        self.indices = {}
        self.scale = {}

        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features


    def update_layer(self, adapter_name, block_size, scale, init_circulant_weights=None):
        # print('\033 args in layer \033[0m', args)
        if block_size <= 0:
            raise ValueError(f"`block_size` should be a positive integer value but the value passed is {block_size}")
        self.block_size[adapter_name] = block_size
        self.scale[adapter_name] = scale
        if block_size > 0:
            # if args_l.set_bias:
            #     d = self.in_features
            #     center_frequency = args_l.fc  # D_0 
            #     width = args_l.width   # W
            #     order = 2 # 2n
            #     rows, cols = np.ogrid[:d, :d]
            #     distance = np.sqrt((rows - d / 2)**2 + (cols - d / 2)**2)
            #     mask_gs = torch.tensor(np.exp(-(distance * width / (distance**2 - center_frequency**2))**(-2)))
            #     mask_gs = FFT_SHIFT(mask_gs)
            #     samples = torch.multinomial(mask_gs.view(-1),1000, replacement=True)
            #     samples = torch.stack([samples // d, samples % d], dim=1).T
            #     self.indices[adapter_name] = samples
            #     print('\033[32m Using frequency bias... \033[0m')
            
            ind = get_circ_index(self.out_features, self.in_features, block_size)
            size = len(np.unique(ind))
            self.rows = (self.out_features + self.out_features%block_size) // block_size # out_features
            self.cols = (self.in_features + self.in_features%block_size) // block_size # in_features
            # self.indices[adapter_name] = torch.from_numpy(ind.astype(np.int64))
            w = getattr(self.get_base_layer(), "weight", None)
            if w is not None:
                print("init from existing weight")
                # in case model in bfloat16 which is not supported in numpy
                # we convert to float first
                #w = w.detach().cpu().numpy()
                if (w.shape[0] < ind.shape[0]) or (w.shape[1] < ind.shape[1]):
                    repeated_w = w.repeat((ind.shape[0]//w.shape[0]+1), (ind.shape[1]//w.shape[1]+1))
                    w = repeated_w[:ind.shape[0], :ind.shape[1]]
                w = w.detach().cpu().float().numpy()
                circ_w = np.bincount(ind.flatten(), weights=w.flatten()) / float(block_size)
                assert circ_w.shape[0] == size, f"{circ_w.shape[0]} != {size}"
                self.circulant[adapter_name] = nn.Parameter(torch.from_numpy(circ_w), requires_grad=True)
            else:
                print("init from random")
                self.circulant[adapter_name] = nn.Parameter(torch.randn(size), requires_grad=True)
                torch.nn.init.kaiming_normal_(
                    self.circulant[adapter_name].view([self.rows, self.cols, block_size])
                )
            
            print(
                f"adapter_name={adapter_name},"
                f"out_features={self.out_features},"
                f"in_features={self.in_features},"
                f"block_size={block_size},"
                f"param={size}"
            )

        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)


        self.set_adapter(self.active_adapters)

    def reset_circulant_parameters(self, adapter_name, init_circulant_weights):

        print("[reset_circulant_parameters] liaosiyu test")
        
        if init_circulant_weights is False:
            return

        if adapter_name in self.circulant.keys():
            if init_circulant_weights is True:
                # # initialize A the same way as the default for nn.Linear and B to zero
                # # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                # nn.init.kaiming_uniform_(self.circulant[adapter_name].weight, a=math.sqrt(5))

                torch.nn.init.kaiming_normal_(
                    self.circulant[adapter_name].weight.view([self.rows, self.cols, self.circ])
                )

            elif init_circulant_weights.lower() == "gaussian":
                nn.init.normal_(self.circulant[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_circulant_weights=}")
        if adapter_name in self.circulant.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.circulant[adapter_name])



# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------



class Linear(nn.Module, CirculantLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        block_size: int = 0,
        scale: float = 0.1,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_circulant_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        CirculantLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, block_size, scale, init_circulant_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.circulant.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.circulant.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.circulant[adapter].device
        dtype = self.circulant[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        circulant = self.circulant[adapter]
        indices = self.indices[adapter].to(circulant.device)# * self.scale[adapter]
        weight = circulant[indices]
        
        if cast_to_fp32:
            weight = weight.float()

        output_tensor = weight

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            # cast back the weights
            # self.weight = weight.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.circulant.keys():
                    continue
                
                circulant = self.circulant[active_adapter]


                #print(f"[forward] circulant={torch.sum(circulant)}, circulant_device={circulant.device}, x_device={x.device}")


                # # do not add scale
                # dense_s = circulant[indices]
                # if circulant.dtype == torch.bfloat16:
                #     dense_s = dense_s.to(torch.float16)
                # delta_w = dense_s #* scale
                # x, delta_w = x.to(circulant.dtype), delta_w.to(circulant.dtype)
                # result += torch.einsum('ijk,kl->ijl', x, delta_w)


                block_size = self.block_size[active_adapter]
                scale = self.scale[active_adapter]
                batch, seq_len, _ = x.shape

                if circulant.dtype == torch.bfloat16:
                    #bf16 not supported by rfft yet
                    # https://github.com/pytorch/pytorch/issues/70664
                    tmp_circulant = circulant.to(torch.float32)
                    x = x.to(tmp_circulant.dtype)

                    y = BlockCircMV.apply(
                        x.view([batch * seq_len, -1]),
                        tmp_circulant.view([1, self.rows, self.cols, block_size]),
                        None,
                    )

                    y = y.to(result.dtype)
                    
                else:
                    # fft_w = torch.fft.rfft(
                    #     circulant.view([1, self.rows, self.cols, block_size]),
                    #     n = block_size
                    # )
                    # x = x.to(circulant.dtype)
                    # y = block_circ_matmul(fft_w, x.view([batch * seq_len, -1]), self.rows, self.cols, block_size)
                    y = BlockCircMV.apply(
                        x.view([batch * seq_len, -1]),
                        circulant.view([1, self.rows, self.cols, block_size]),
                        None,
                    )
                y = y[ :, :self.out_features].view([batch, seq_len, -1])
                result += y * scale

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "circulant." + rep
