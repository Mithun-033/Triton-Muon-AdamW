import triton
import triton.language as tl

import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=["N"],
)
@triton.jit
def adamw_step(
    p_ptr,
    g_ptr,
    m_ptr,
    v_ptr,
    N: int,
    BLOCK_SIZE: tl.constexpr,
    lr: float,
    weight_decay: tl.constexpr,
    bias1 ,
    bias2,
    beta1: tl.constexpr = 0.9,
    beta2: tl.constexpr = 0.99,
    eps: tl.constexpr = 1e-8,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid_n * BLOCK_SIZE

    ptrs = pid_m * N + offset + cols
    mask = offset + cols < N

    params = tl.load(p_ptr + ptrs, mask=mask, other=0.0)
    g = tl.load(g_ptr + ptrs, mask=mask, other=0.0)
    m = tl.load(m_ptr + ptrs, mask=mask, other=0.0)
    v = tl.load(v_ptr + ptrs, mask=mask, other=0.0)

    m_t = beta1 * m + (1 - beta1) * g
    v_t = beta2 * v + (1 - beta2) * (g * g)
    m_c = m_t / bias1
    v_c = v_t / bias2

    param_decay = (1 - lr * weight_decay) * params
    param_new = param_decay - lr * (m_c / (tl.sqrt(v_c) + eps))

    tl.store(p_ptr + ptrs, param_new, mask=mask)
    tl.store(m_ptr + ptrs, m_t, mask=mask)
    tl.store(v_ptr + ptrs, v_t, mask=mask)


def solve_adamw_step(
    matrix: torch.tensor,
    gradient: torch.tensor,
    m: torch.Tensor,
    v: torch.tensor,
    lr : float,
    bias1,
    bias2,
    beta1: float,
    beta2: float,
    weight_decay,
    eps : float
):
    if matrix.ndim==1:
        matrix=matrix.unsqueeze(0)
        gradient=gradient.unsqueeze(0)
        m=m.unsqueeze(0)
        v=v.unsqueeze(0)

    M, N = matrix.shape
    grid=lambda meta:(M,triton.cdiv(N,meta['BLOCK_SIZE']))

    adamw_step[grid](matrix, gradient, m, v, N=N, lr=lr, weight_decay=weight_decay, bias1=bias1, bias2=bias2, beta1=beta1, beta2=beta2, eps=eps)


class TritonAdamW:
    def __init__(self, model, beta1, beta2, lr, weight_decay, eps):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.step_num = 0
        self.m = {}
        self.v = {}

        for param in self.model.parameters():
            self.m[param] = torch.zeros_like(param, dtype = torch.float32)
            self.v[param] = torch.zeros_like(param, dtype = torch.float32)

    @torch.no_grad()
    def step(self):
        self.step_num += 1
        for param in self.model.parameters():
            if param.grad is None:
                continue
            solve_adamw_step(
                param,
                param.grad,
                self.m[param],
                self.v[param],
                self.lr,
                1 - self.beta1 ** self.step_num,
                1 - self.beta2 ** self.step_num,
                self.beta1,
                self.beta2,
                self.weight_decay,
                self.eps
            )

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

