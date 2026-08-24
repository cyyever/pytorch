from torchgen.dest.native_functions import (
    compute_native_function_declaration as compute_native_function_declaration,
)
from torchgen.dest.register_dispatch_key import (
    gen_registration_headers as gen_registration_headers,
    gen_registration_helpers as gen_registration_helpers,
    RegisterDispatchKey as RegisterDispatchKey,
)
from torchgen.dest.ufunc import (
    compute_ufunc_cpu as compute_ufunc_cpu,
    compute_ufunc_cpu_kernel as compute_ufunc_cpu_kernel,
    compute_ufunc_cuda as compute_ufunc_cuda,
)
