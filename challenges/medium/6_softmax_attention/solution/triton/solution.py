import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q, K, V, output,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_om, stride_od,
    M, N, d,
    sm_scale,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.float32))
    K = K.to(tl.pointer_type(tl.float32))
    V = V.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    offsets_d = tl.arange(0, BLOCK_SIZE_D)
    mask_q = offsets_d < d

    q = tl.load(Q + pid * stride_qm + offsets_d * stride_qd, mask=mask_q, other=0.0)

    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
    for start_n in range(0, N, BLOCK_SIZE_N):
        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_k = (offsets_n[:, None] < N) & (offsets_d[None, :] < d)
        k = tl.load(K + offsets_n[:, None] * stride_kn + offsets_d[None, :] * stride_kd,
                    mask=mask_k, other=0.0)
        qk = tl.sum(q[None, :] * k, axis=1)
        qk *= sm_scale
        qk = tl.where(offsets_n < N, qk, -float('inf'))

        m_prev = m_i
        block_max = tl.max(qk, axis=0)
        m_i = tl.maximum(m_prev, block_max)
        alpha = tl.exp(m_prev - m_i)
        p = tl.exp(qk - m_i)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        mask_v = (offsets_n[:, None] < N) & (offsets_d[None, :] < d)
        v = tl.load(V + offsets_n[:, None] * stride_vn + offsets_d[None, :] * stride_vd,
                    mask=mask_v, other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

    acc = acc / l_i
    tl.store(output + pid * stride_om + offsets_d * stride_od, acc, mask=mask_q)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    sm_scale = d ** -0.5
    grid = (M,)
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = triton.next_power_of_2(d)
    attention_kernel[grid](
        Q, K, V, output,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        M, N, d,
        sm_scale,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_D=triton.next_power_of_2(d),
        num_warps=4,
        num_stages=2
    )
