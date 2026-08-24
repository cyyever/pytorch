import torch


class WrapperModule:
    """Wraps the instance of wrapped_type.
    Randomaly initializes num_params tensors with single float element.
    Args:
        wrapped_type:
            - Object type to be wrapped.
                Expects the wrapped_type to:
                   - be constructed with pt_fn specified in module_config.
                   - provide forward method that takes module_config.num_params args.
        module_config:
            - Specified pt_fn to construct wrapped_type with, and number of
              parameters wrapped_type's forward method takes.
    """

    def __init__(self, wrapped_type, module_config):
        pt_fn = module_config.pt_fn
        self.module = wrapped_type(pt_fn)
        self.tensor_inputs = []
        self.module_name = wrapped_type.__name__
        for _ in range(module_config.num_params):
            self.tensor_inputs.append(torch.randn(1))
        print(f"Benchmarking module {self.module_name} with fn {pt_fn.__name__}")

    def forward(self, niters):
        with torch.no_grad():
            for _ in range(niters):
                self.module.forward(*self.tensor_inputs)
