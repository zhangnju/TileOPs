import torch

str2dtype = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    "int32": torch.int32
}

dtype2str = {v: k for k, v in str2dtype.items()}


@torch.compile
def reduce_on_dim0(x: torch.Tensor) -> torch.Tensor:
    """Reduce a tensor on dimension 0.

    Arguments:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Reduced tensor.
    """
    return x[0] if x.size(0) == 1 else x.sum(dim=0)


@torch.compile
def zero_pad(x: torch.Tensor, pad_size: int, dim: int) -> torch.Tensor:
    """Pad a tensor with 0 to a be divisible by `pad_size` along a specified dimension.

    Arguments:
        x (torch.Tensor): Input tensor.
        pad_size (int): The size to pad to be divisible by.
        dim (int): The dimension to pad.

    Returns:
        torch.Tensor: Padded tensor.
    """
    if x.size(dim) % pad_size == 0:
        return x
    pad_len = (pad_size - x.size(dim) % pad_size)
    assert 0 < pad_len < pad_size

    zero_shape = list(x.shape)
    zero_shape[dim] = pad_len
    zero_shape = tuple(zero_shape)
    zeros = torch.zeros(zero_shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, zeros), dim=dim)


def ensure_contiguous(func: callable) -> callable:
    """Decorator to ensure that all tensor arguments are contiguous before calling the function.

    Arguments:
        func (callable): The function to decorate.

    Returns:
        callable: The decorated function.
    """

    def wrapper(*args, **kwargs):
        args = [arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapper


def _is_rocm() -> bool:
    """Return True when running on an AMD GPU via ROCm/HIP."""
    return getattr(torch.version, 'hip', None) is not None


def is_hopper() -> bool:
    """Return True iff the current GPU is NVIDIA Hopper (SM90).

    Always returns False on AMD/ROCm devices so that Hopper-exclusive
    kernel paths (WGMMA, warp-specialization) are never selected there.
    """
    if _is_rocm():
        return False
    return torch.cuda.get_device_capability() == (9, 0)


def get_sm_version() -> int:
    """Return an NVIDIA SM version integer for the current GPU.

    NVIDIA only: major * 10 + minor  (e.g. SM80→80, SM89→89, SM90→90)
    Do not call this on AMD/ROCm devices; use get_amd_gfx_version() instead.
    """
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def get_amd_gfx_version() -> int:
    """Return the AMD GFX series number for the current GPU.

    AMD only: parses the gcnArchName (e.g. 'gfx950') and returns the
    numeric suffix (e.g. 950).  Use this in supported_amd_archs checks
    instead of get_sm_version(), which is reserved for NVIDIA SM IDs.

    Raises:
        RuntimeError: if called on a non-ROCm device.
        ValueError: if the GFX architecture name cannot be parsed.
    """
    if not _is_rocm():
        raise RuntimeError("get_amd_gfx_version() must only be called on AMD/ROCm devices")
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    gcn_arch = prop.gcnArchName  # e.g. 'gfx950', 'gfx942'
    # Strip optional ISA suffix like ':sramecc+:xnack-'
    base = gcn_arch.split(':')[0]
    if not base.startswith('gfx'):
        raise ValueError(f"Unexpected gcnArchName format: {gcn_arch!r}")
    return int(base[3:])
