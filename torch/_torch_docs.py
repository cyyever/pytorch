# mypy: allow-untyped-defs
"""Adds docstrings to functions defined in the torch._C module."""

import re



def parse_kwargs(desc):
    r"""Map a description of args to a dictionary of {argname: description}.

    Input:
        ('    weight (Tensor): a weight tensor\n' +
         '        Some optional description')
    Output: {
        'weight': \
        'weight (Tensor): a weight tensor\n        Some optional description'
    }
    """
    # Split on exactly 4 spaces after a newline
    regx = re.compile(r"\n\s{4}(?!\s)")
    kwargs = [section.strip() for section in regx.split(desc)]
    kwargs = [section for section in kwargs if len(section) > 0]
    return {desc.split(" ")[0]: desc for desc in kwargs}


def merge_dicts(*dicts):
    """Merge dictionaries into a single dictionary."""
    return {x: d[x] for d in dicts for x in d}


common_args = parse_kwargs(
    """
    input (Tensor): the input tensor.
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned tensor. Default: ``torch.preserve_format``.
"""
)

reduceops_common_args = merge_dicts(
    common_args,
    parse_kwargs(
        """
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
"""
    ),
    {
        "opt_keepdim": """
    keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.
"""
    },
)

multi_dim_common = merge_dicts(
    reduceops_common_args,
    parse_kwargs(
        """
    dim (int or tuple of ints): the dimension or dimensions to reduce.
"""
    ),
    {
        "keepdim_details": """
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).
"""
    },
    {
        "opt_dim": """
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
"""
    },
    {
        "opt_dim_all_reduce": """
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.
"""
    },
)

single_dim_common = merge_dicts(
    reduceops_common_args,
    parse_kwargs(
        """
    dim (int): the dimension to reduce.
"""
    ),
    {
        "opt_dim": """
    dim (int, optional): the dimension to reduce.
"""
    },
    {
        "opt_dim_all_reduce": """
    dim (int, optional): the dimension to reduce.
        If ``None``, all dimensions are reduced.
"""
    },
    {
        "opt_dim_without_none": """
    dim (int, optional): the dimension to reduce. If omitted, all dimensions are reduced. Explicit ``None`` is not supported.
"""
    },
    {
        "keepdim_details": """If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensor having 1 fewer dimension than :attr:`input`."""
    },
)

factory_common_args = merge_dicts(
    common_args,
    parse_kwargs(
        """
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.contiguous_format``.
    check_invariants (bool, optional): If sparse tensor invariants are checked.
        Default: as returned by :func:`torch.sparse.check_sparse_tensor_invariants.is_enabled`,
        initially False.
"""
    ),
    {
        "sparse_factory_device_note": """\
.. note::

   If the ``device`` argument is not specified the device of the given
   :attr:`values` and indices tensor(s) must match. If, however, the
   argument is specified the input Tensors will be converted to the
   given device and in turn determine the device of the constructed
   sparse tensor."""
    },
)

factory_like_common_args = parse_kwargs(
    """
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.
"""
)

factory_data_common_args = parse_kwargs(
    """
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
"""
)

tf32_notes = {
    "tf32_note": """This operator supports :ref:`TensorFloat32<tf32_on_ampere>`."""
}

rocm_fp16_notes = {
    "rocm_fp16_note": """On certain ROCm devices, when using float16 inputs this module will use \
:ref:`different precision<fp16_on_mi200>` for backward."""
}

reproducibility_notes: dict[str, str] = {
    "forward_reproducibility_note": """This operation may behave nondeterministically when given tensors on \
a CUDA device. See :doc:`/notes/randomness` for more information.""",
    "backward_reproducibility_note": """This operation may produce nondeterministic gradients when given tensors on \
a CUDA device. See :doc:`/notes/randomness` for more information.""",
    "cudnn_reproducibility_note": """In some circumstances when given tensors on a CUDA device \
and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is \
undesirable, you can try to make the operation deterministic (potentially at \
a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. \
See :doc:`/notes/randomness` for more information.""",
}

sparse_support_notes = {
    "sparse_beta_warning": """
.. warning::
    Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
    or may not have autograd support. If you notice missing functionality please
    open a feature request.""",
}
