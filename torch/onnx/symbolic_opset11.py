from typing import Any, Never


_UNSUPPORTED_MESSAGE = (
    "ONNX support is not included in this PyTorch build. "
    "Use a PyTorch distribution built with ONNX support."
)


def _unsupported(*args: Any, **kwargs: Any) -> Never:
    raise RuntimeError(_UNSUPPORTED_MESSAGE)


unsqueeze = _unsupported
squeeze = _unsupported
select = _unsupported
