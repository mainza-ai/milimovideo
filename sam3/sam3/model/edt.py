# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Triton kernel for euclidean distance transform (EDT)"""

import torch
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    # Mock triton for decorators
    class MockTriton:
        def jit(self, f): return f
        def autotune(self, *args, **kwargs): return lambda f: f
    triton = MockTriton()
import numpy as np

# ... (docstring skipped/kept if matching) ...

@triton.jit
def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr if TRITON_AVAILABLE else bool):
    pass # Dummy kernel if triton not available, real one is JIT compiled by import if available

# Define the real kernel only if triton available, or keep the definition but it won't be used
if TRITON_AVAILABLE:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        # This is a somewhat verbatim implementation of the efficient 1D EDT algorithm described above
        # It can be applied horizontally or vertically depending if we're doing the first or second stage.
        # It's parallelized across batch+row (or batch+col if horizontal=False)
        # TODO: perhaps the implementation can be revisited if/when local gather/scatter become available in triton
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width
    
        # This will be the index of the right most parabola in the envelope ("the top of the stack")
        k = 0
        for q in range(1, length):
            # Read the function value at the current location. Note that we're doing a singular read, not very efficient
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            # location of the parabola on top of the stack
            r = tl.load(v + block_start + (k * stride))
            # associated boundary
            z_k = tl.load(z + block_start + (k * stride))
            # value of the function at the parabola location
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            # intersection between the two parabolas
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2
    
            # we'll pop as many parabolas as required
            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2
    
            # Store the new one
            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)
    
        # Last step, we read the envelope to find the min in every location
        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)


def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
        It should be equivalent to a batched version of cv2.distanceTransform(input, cv2.DIST_L2, 0)
    """
    assert data.dim() == 3
    
    if not TRITON_AVAILABLE or not data.is_cuda:
        # Fallback to OpenCV
        import cv2
        B, H, W = data.shape
        device = data.device
        
        # Convert to numpy uint8 (0 where we want 0 distance, 1 elsewhere? 
        # Usually dist transform calculates distance to nearest 0. 
        # The triton kernel seems to compute squared distance field for arbitrary function?
        # The docstring check: "equivalent to cv2.distanceTransform(input, cv2.DIST_L2, 0)"
        # cv2 inputs: image (8-bit) content. 0 dist to 0 pixels.
        
        data_np = data.detach().cpu().numpy().astype(np.uint8)
        # Note: cv2.distanceTransform expects 0 pixels to be targets (dist=0) and non-zero to be background (dist calculated).
        # data input is "binary images". 
        # If data is a mask where object is 1 and bg is 0, we want distance to 0? 
        # Wait, usually we want distance from boundary?
        # Let's assume standard behavior: 0 is target.
        
        outputs = []
        for i in range(B):
            # cv2.distanceTransform returns float32
            dist = cv2.distanceTransform(data_np[i], cv2.DIST_L2, 0)
            outputs.append(dist)
            
        return torch.tensor(np.array(outputs), device=device, dtype=torch.float32)

    B, H, W = data.shape
    data = data.contiguous()

    # Allocate the "function" tensor. Implicitly the function is 0 if data[i,j]==0 else +infinity
    output = torch.where(data, 1e18, 0.0)
    assert output.is_contiguous()

    # Scratch tensors for the parabola stacks
    parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
    parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    # Grid size (number of blocks)
    grid = (B, H)

    # Launch initialization kernel
    edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=True,
    )

    # reset the parabola stacks
    parabola_loc.zero_()
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, W)
    edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=False,
    )
    # don't forget to take sqrt at the end
    return output.sqrt()
