from typing import List, Optional
import torch
import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)


_TORCH_TO_GL_DTYPE = {
    torch.float8_e4m3fn: gl.float8e4nv,
    torch.float8_e5m2: gl.float8e5,
    torch.bfloat16: gl.bfloat16,
    torch.float16: gl.float16,
    torch.float32: gl.float32,
}


def _gl_dtype(t: torch.Tensor):
    try:
        return _TORCH_TO_GL_DTYPE[t.dtype]
    except KeyError as e:
        raise TypeError(f"Unsupported tensor dtype: {t.dtype}") from e


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    m_reps = triton.cdiv(BLOCK_M, m)
    n_reps = triton.cdiv(num_warps, m_reps)
    max_n = max(BLOCK_N // n_reps, 8)
    n = 256
    while n > max_n or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n


@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 256 // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )


@aggregate
class Config:
    B: gl.constexpr
    M_aligned: gl.constexpr
    N: gl.constexpr
    K: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    TILE_ORDER: gl.constexpr
    num_warps: gl.constexpr
    num_stages: gl.constexpr
    num_sms: gl.constexpr
    # Derived: tile counts.
    num_m_tiles: gl.constexpr
    num_n_tiles: gl.constexpr
    num_k_blocks: gl.constexpr
    num_tiles_per_batch: gl.constexpr
    num_tiles: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, B, M_aligned, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TILE_ORDER, num_warps, num_stages, num_sms):
        self.B = gl.constexpr(B)
        self.M_aligned = gl.constexpr(M_aligned)
        self.N = gl.constexpr(N)
        self.K = gl.constexpr(K)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.TILE_ORDER = gl.constexpr(TILE_ORDER)
        self.num_warps = gl.constexpr(num_warps)
        self.num_stages = gl.constexpr(num_stages)
        self.num_sms = gl.constexpr(num_sms)
        num_m = M_aligned // BLOCK_M
        num_n = N // BLOCK_N
        self.num_m_tiles = gl.constexpr(num_m)
        self.num_n_tiles = gl.constexpr(num_n)
        self.num_k_blocks = gl.constexpr(K // BLOCK_K)
        self.num_tiles_per_batch = gl.constexpr(num_m * num_n)
        self.num_tiles = gl.constexpr(B * num_m * num_n)


@aggregate
class BarrierCounter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def increment(self):
        if self.num_barriers == 1:
            return BarrierCounter(gl.to_tensor(0), self.phase ^ 1, self.num_barriers)
        next_index = self.index + 1
        rollover = next_index == self.num_barriers
        index = gl.where(rollover, 0, next_index)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return BarrierCounter(index, phase, self.num_barriers)


@aggregate
class Channel:
    x_smem: gl.shared_memory_descriptor
    y_smem: gl.shared_memory_descriptor
    xs_smem: gl.shared_memory_descriptor
    ready_bars: gl.shared_memory_descriptor
    empty_bars: gl.shared_memory_descriptor
    num_stages: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, x_smem, y_smem, xs_smem, ready_bars, empty_bars, num_stages):
        self.x_smem = x_smem
        self.y_smem = y_smem
        self.xs_smem = xs_smem
        self.ready_bars = ready_bars
        self.empty_bars = empty_bars
        self.num_stages = gl.constexpr(num_stages)

    @gluon.jit
    def alloc(
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr,
        x_dtype: gl.constexpr,
        x_layout: gl.constexpr,
        y_dtype: gl.constexpr,
        y_layout: gl.constexpr,
        xs_dtype: gl.constexpr,
        xs_layout: gl.constexpr,
        num_stages: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        # x: 3D box [1, BLOCK_M, BLOCK_K] (x is permuted/non-contig at the global level).
        # y, xs: 2D boxes (callers see flat 2D global views, see wrapper).
        x_smem = gl.allocate_shared_memory(x_dtype, [num_stages, 1, BLOCK_M, BLOCK_K], x_layout)
        y_smem = gl.allocate_shared_memory(y_dtype, [num_stages, BLOCK_N, BLOCK_K], y_layout)
        xs_smem = gl.allocate_shared_memory(xs_dtype, [num_stages, 1, BLOCK_M], xs_layout)
        ready_bars = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
        empty_bars = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(num_stages):
            mbarrier.init(ready_bars.index(i), count=1)
            mbarrier.init(empty_bars.index(i), count=1)
            mbarrier.arrive(empty_bars.index(i), count=1)
        return Channel(x_smem, y_smem, xs_smem, ready_bars, empty_bars, num_stages)

    @gluon.jit
    def release(self):
        self.x_smem._keep_alive()
        self.y_smem._keep_alive()
        self.xs_smem._keep_alive()
        for i in gl.static_range(self.num_stages):
            mbarrier.invalidate(self.ready_bars.index(i))
            mbarrier.invalidate(self.empty_bars.index(i))


@gluon.jit
def get_tile(tile_id, config):
    # TILE_ORDER: 0 = horizontal (N fastest within batch — favours x reuse across N sweep)
    #             1 = vertical   (M fastest within batch — favours y reuse across M sweep)
    batch_id = tile_id // config.num_tiles_per_batch
    local_id = tile_id % config.num_tiles_per_batch
    if config.TILE_ORDER == 0:
        m_tile_id = local_id // config.num_n_tiles
        n_tile_id = local_id % config.num_n_tiles
    else:
        n_tile_id = local_id // config.num_m_tiles
        m_tile_id = local_id % config.num_m_tiles
    return batch_id, m_tile_id, n_tile_id


@gluon.jit
def compute_partition(channel, config, tensors):
    x_desc, y_desc, xs_desc, z_desc, ys_ptr = tensors
    start_pid = gl.program_id(0)
    counter = BarrierCounter(index=gl.to_tensor(0), phase=gl.to_tensor(0), num_barriers=config.num_stages)
    mma_layout: gl.constexpr = pick_wgmma_layout(x_desc.dtype, config.BLOCK_M, config.BLOCK_N, num_warps=config.num_warps)
    z_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, config.BLOCK_M, config.BLOCK_N], z_desc.dtype)
    z_smem = gl.allocate_shared_memory(z_desc.dtype, [1, config.BLOCK_M, config.BLOCK_N], z_smem_layout)

    slice_m_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)

    for tile_id in range(start_pid, config.num_tiles, config.num_sms):
        batch_id, m_tile_id, n_tile_id = get_tile(tile_id, config)
        m_start = m_tile_id * config.BLOCK_M
        n_start = n_tile_id * config.BLOCK_N
        # ys layout matches the scale grid (N/BLOCK_N, K/BLOCK_K); one scale per (n_tile, k_block).
        ys_base = (batch_id * config.num_n_tiles + n_tile_id) * config.num_k_blocks

        partial_zero = gl.zeros((config.BLOCK_M, config.BLOCK_N), dtype=gl.float32, layout=mma_layout)
        acc = gl.zeros((config.BLOCK_M, config.BLOCK_N), dtype=gl.float32, layout=mma_layout)

        for k in range(0, config.K, config.BLOCK_K):
            k_block_idx = k // config.BLOCK_K
            index, phase = counter.index, counter.phase
            x_slot = channel.x_smem.index(index)   # [1, BLOCK_M, BLOCK_K]
            y_slot = channel.y_smem.index(index)   # [1, BLOCK_N, BLOCK_K]
            xs_slot = channel.xs_smem.index(index)  # [1, 1, BLOCK_M]
            ready_bar = channel.ready_bars.index(index)
            empty_bar = channel.empty_bars.index(index)
            mbarrier.wait(ready_bar, phase)

            x = x_slot.reshape((config.BLOCK_M, config.BLOCK_K))
            y = y_slot
            xs_1d = xs_slot.reshape((config.BLOCK_M,))

            x_s = xs_1d.load(slice_m_layout)
            y_s = gl.load(ys_ptr + ys_base + k_block_idx)
            xy_s = x_s * y_s

            y_t = y.permute((1, 0))
            partial_async = warpgroup_mma(x, y_t, partial_zero, use_acc=False, is_async=True)
            partial = warpgroup_mma_wait(num_outstanding=0, deps=(partial_async,))

            acc = acc + partial * xy_s[:, None]

            mbarrier.arrive(empty_bar)
            counter = counter.increment()

        acc_out = acc.to(z_desc.dtype)
        tma.store_wait(pendings=0)
        z_smem.reshape((config.BLOCK_M, config.BLOCK_N)).store(acc_out)
        fence_async_shared()
        tma.async_copy_shared_to_global(z_desc, [batch_id, m_start, n_start], z_smem)

    tma.store_wait(pendings=0)


@gluon.jit
def load_partition(channel, config, tensors):
    x_desc, y_desc, xs_desc, z_desc, ys_ptr = tensors
    start_pid = gl.program_id(0)
    counter = BarrierCounter(index=gl.to_tensor(0), phase=gl.to_tensor(0), num_barriers=config.num_stages)

    # x: fp8 (1 B/elem), y: fp8 (1 B/elem), xs: fp32 (4 B/elem)
    nbytes: gl.constexpr = (
        config.BLOCK_M * config.BLOCK_K
        + config.BLOCK_N * config.BLOCK_K
        + 1 * config.BLOCK_M * 4
    )

    for tile_id in range(start_pid, config.num_tiles, config.num_sms):
        batch_id, m_tile_id, n_tile_id = get_tile(tile_id, config)
        m_start = m_tile_id * config.BLOCK_M
        n_start = n_tile_id * config.BLOCK_N
        # 2D flat coords: y as (B*N, K), xs as (B*num_k_blocks, M_aligned)
        y_row = batch_id * config.N + n_start
        xs_row_base = batch_id * config.num_k_blocks

        for k in range(0, config.K, config.BLOCK_K):
            k_block_idx = k // config.BLOCK_K
            index, phase = counter.index, counter.phase
            x_slot = channel.x_smem.index(index)
            y_slot = channel.y_smem.index(index)
            xs_slot = channel.xs_smem.index(index)
            ready_bar = channel.ready_bars.index(index)
            empty_bar = channel.empty_bars.index(index)
            mbarrier.wait(empty_bar, phase)

            mbarrier.expect(ready_bar, nbytes)
            tma.async_copy_global_to_shared(x_desc, [batch_id, m_start, k], ready_bar, x_slot)
            tma.async_copy_global_to_shared(y_desc, [y_row, k], ready_bar, y_slot)
            tma.async_copy_global_to_shared(xs_desc, [xs_row_base + k_block_idx, m_start], ready_bar, xs_slot)

            counter = counter.increment()


@triton.autotune(
    configs=[
        triton.Config({"TILE_ORDER": tile_order}, num_warps=nw, num_stages=ns)
        for nw in (4, 8)
        for ns in (4, 6, 8)
        for tile_order in (0, 1)  # 0=horizontal (n fastest), 1=vertical (m fastest)
    ],
    key=["B", "M_aligned", "N", "K"],
)
@gluon.jit
def w8a8_block_fp8_bmm_kernel(
    x_desc,
    y_desc,
    xs_desc,
    z_desc,
    ys_ptr,
    B: gl.constexpr,
    M_aligned: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    TILE_ORDER: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr,
    num_sms: gl.constexpr,
):
    config = Config(
        B=B,
        M_aligned=M_aligned,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        TILE_ORDER=TILE_ORDER,
        num_warps=num_warps,
        num_stages=num_stages,
        num_sms=num_sms,
    )
    tensors = (x_desc, y_desc, xs_desc, z_desc, ys_ptr)
    channel = Channel.alloc(
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        x_dtype=x_desc.dtype,
        x_layout=gl.constexpr(x_desc.layout),
        y_dtype=y_desc.dtype,
        y_layout=gl.constexpr(y_desc.layout),
        xs_dtype=xs_desc.dtype,
        xs_layout=gl.constexpr(xs_desc.layout),
        num_stages=num_stages,
        num_warps=num_warps,
    )

    gl.warp_specialize(
        [(compute_partition, (channel, config, tensors)),
         (load_partition, (channel, config, tensors))],
        [1],
        [24],
    )

    channel.release()


def w8a8_block_fp8_bmm(
    x: torch.Tensor,
    y: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    block_size: List[int] = [128, 128],
    z: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
):
    # x: [B, M, K]  fp8
    # y: [B, N, K]  fp8
    # xs: [B, M, K // block_k]      f32
    # ys: [B, N // block_n, K // block_k]  f32
    # z:  [B, M, N]  out_dtype
    assert len(block_size) == 2
    BLOCK_N_S, BLOCK_K_S = block_size
    assert BLOCK_N_S == 128 and BLOCK_K_S == 128, "this kernel assumes 128x128 block-wise FP8 scales"

    assert x.ndim == 3 and y.ndim == 3 and xs.ndim == 3 and ys.ndim == 3
    assert x.shape[0] == y.shape[0] == xs.shape[0] == ys.shape[0]
    assert x.shape[-1] == y.shape[-1]
    assert x.shape[:-1] == xs.shape[:-1]
    assert x.stride(-1) == 1 and y.stride(-1) == 1

    device = x.device
    B, M, K = x.shape
    _, N, _ = y.shape
    assert K % BLOCK_K_S == 0 and N % BLOCK_N_S == 0
    num_kb = K // BLOCK_K_S

    if z is None:
        z = torch.empty((B, M, N), device=device, dtype=output_dtype)
    else:
        assert z.shape == (B, M, N) and z.device == device and z.dtype == output_dtype
        assert z.stride(-1) == 1

    BLOCK_M = 64
    BLOCK_N = BLOCK_N_S  # 128
    BLOCK_K = BLOCK_K_S  # 128

    M_aligned = triton.cdiv(M, BLOCK_M) * BLOCK_M
    xs_transformed = torch.empty((B, num_kb, M_aligned), device=device, dtype=torch.float32)
    xs_transformed[:, :, :M].copy_(xs.transpose(1, 2))

    x_gl_dtype = _gl_dtype(x)
    y_gl_dtype = _gl_dtype(y)
    xs_gl_dtype = _gl_dtype(xs_transformed)
    z_gl_dtype = _gl_dtype(z)

    x_layout = gl.NVMMASharedLayout.get_default_for([1, BLOCK_M, BLOCK_K], x_gl_dtype)
    x_desc = TensorDescriptor.from_tensor(x, block_shape=[1, BLOCK_M, BLOCK_K], layout=x_layout)

    assert y.is_contiguous(), "y must be contiguous so it can be viewed as (B*N, K)"
    y_flat = y.view(B * N, K)
    y_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], y_gl_dtype)
    y_desc = TensorDescriptor.from_tensor(y_flat, block_shape=[BLOCK_N, BLOCK_K], layout=y_layout)

    xs_flat = xs_transformed.view(B * num_kb, M_aligned)
    xs_layout = gl.NVMMASharedLayout.get_default_for([1, BLOCK_M], xs_gl_dtype)
    xs_desc = TensorDescriptor(
        xs_flat,
        shape=[B * num_kb, M],
        strides=[M_aligned, 1],
        block_shape=[1, BLOCK_M],
        layout=xs_layout,
    )

    z_layout = gl.NVMMASharedLayout.get_default_for([1, BLOCK_M, BLOCK_N], z_gl_dtype)
    z_desc = TensorDescriptor.from_tensor(z, block_shape=[1, BLOCK_M, BLOCK_N], layout=z_layout)

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    w8a8_block_fp8_bmm_kernel[(num_sms,)](
        x_desc, y_desc, xs_desc, z_desc, ys,
        B=B, M_aligned=M_aligned, N=N, K=K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_sms=num_sms,
    )

    # Print autotune best config once per unique shape key.
    best = w8a8_block_fp8_bmm_kernel.best_config
    if best is not None:
        key = (B, M_aligned, N, K)
        if _printed_best_configs.get(key) != repr(best):
            _printed_best_configs[key] = repr(best)
            print(
                f"[w8a8_block_fp8_bmm] B={B} M={M} (M_aligned={M_aligned}) N={N} K={K}  "
                f"best_config={best}"
            )

    return z


_printed_best_configs: dict = {}
