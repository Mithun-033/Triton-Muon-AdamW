import triton
import triton.language as tl

import torch

@triton.jit
def adamw_step(
    p_ptr,
    g_ptr,
    m_ptr,
    v_ptr,
    N: int,
    BLOCK_SIZE: int,
    lr: float,
    weight_decay: float,
    step: int,
    beta1: int = 0.9,
    beta2: int = 0.99,
    eps: float = 1e-8,
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)

    ptrs = pid * N + tl.arange(BLOCK_SIZE)
    mask = cols < N

    params = tl.load(p_ptr + ptrs, mask=mask, other=0.0)
    g = tl.load(g_ptr + ptrs, mask=mask, other=0.0)
    m = tl.load(m_ptr + ptrs, mask=mask, other=0.0)
    v = tl.load(v_ptr + ptrs, mask=mask, other=0.0)

    m_t = beta1 * m + (1 - beta1) * g
    v_t = beta2 * v + (1 - beta2) * (g * g)
    m_c = m_t / (1 - beta1**step)
    v_c = v_t / (1 - beta2**step)

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
    step: int,
    beta1: float,
    beta2: float,
    weight_decay,
    eps : float
):
    M, N = matrix.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    adamw_step[grid](matrix, gradient, m, v, N, BLOCK_SIZE, lr, weight_decay, step, beta1, beta2, eps )


class TritonAdamW:
    def __init__(self, model, beta1, beta2, lr, weight_decay, eps):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.step = 0
        self.m = {}
        self.v = {}

        for param in self.model.parameters():
            self.m[param] = torch.zeros_like(param)
            self.v[param] = torch.zeros_like(param)

    def step(self):
        self.step += 1
        for param in self.model.parameters():
            solve_adamw_step(
                param,
                param.grad,
                self.m[param],
                self.v[param],
                self.lr,
                self.step,
                self.beta1,
                self.beta2,
                self.weight_decay,
                self.eps
            )

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

