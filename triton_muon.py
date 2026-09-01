import torch
import triton
import triton.language as tl


@triton.jit
def ns_kernel_1(a_ptr,
        out_ptr,
        M,
        K,
        stride_am,
        stride_ak,
        stride_cm,
        stride_cn,
        BLOCK_M : tl.constexpr,
        BLOCK_K : tl.constexpr):
    """
    Compute X @ X.transpose
    X.shape = (M,K)
    out.shape = (M,M)
    """
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)

    offset_m = pid_m * BLOCK_M + tl.arange(0,BLOCK_M)
    offset_n = pid_n * BLOCK_M + tl.arange(0,BLOCK_M)
    offset_k = tl.arange(0,BLOCK_K)

    a_ptrs = a_ptr + offset_m[:,None] * stride_am + offset_k[None,:] * stride_ak
    b_ptrs = a_ptr + offset_k[:,None] * stride_ak + offset_n[None,:] * stride_am
    acc = tl.zeros((BLOCK_M,BLOCK_M), dtype = tl.float32)

    for k in range(0,K,BLOCK_K):
        k_mask = offset_k + k < K

        a = tl.load(a_ptrs, mask = (offset_m[:,None] < M) & (k_mask[None,:]), other = 0.0)
        b = tl.load(b_ptrs, mask = (k_mask[:,None]) & (offset_n[None,:] < M), other = 0.0)
        acc = tl.dot(a,b,acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_ak

    out_ptrs = out_ptr + offset_m[:,None] * stride_cm + offset_n[None,:] * stride_cn
    mask = (offset_m[:,None] < M) & (offset_n[None,:] < M)
    tl.store(out_ptrs, acc, mask = mask)

def solve_ns_kernel_1(matrix: torch.Tensor, out: torch.Tensor):
    if matrix.ndim != 2:
        raise AssertionError("Muon only supports 2D tensors")
    M, K = matrix.shape

    BLOCK_M = 64
    BLOCK_K = 32
    GROUP_M = 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(M, BLOCK_M))
    ns_kernel_1[grid](
        a_ptr=matrix,
        out_ptr=out,
        M=M,
        K=K,
        stride_am=matrix.stride(0),
        stride_ak=matrix.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )


@triton.jit
def muon_step(): ...


def solve_muon_step(
    param: torch.Tensor,
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    weight_decay: float,
    nesterov: bool,
    ns_coefficients: tuple[float],
    ns_step: int,
    eps: float,
    lr: float,
): ...
