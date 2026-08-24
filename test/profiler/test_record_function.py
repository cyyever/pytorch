# Owner(s): ["oncall: profiler"]
# ruff: noqa: F841

from typing import Any

import torch
import torch.optim
import torch.utils.data
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import ChromiumEventLogger
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.autograd.profiler import profile as _profile
from torch.profiler import kineto_available, record_function
from torch.testing._internal.common_utils import run_tests, TestCase


# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

Json = dict[str, Any]


class TestRecordFunction(TestCase):
    def _record_function_with_param(self):
        u = torch.randn(3, 4, 5, requires_grad=True)
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            with record_function("## TEST 1 ##", "1, 2, 3"):
                rf_handle = _record_function_with_args_enter(
                    "## TEST 2 ##", 1, False, 2.5, [u, u], "hello", u
                )
                _record_function_with_args_exit(rf_handle)
            with record_function("## TEST 3 ##"):
                rf_handle = _record_function_with_args_enter("## TEST 4 ##")
                _record_function_with_args_exit(rf_handle)
        return prof

    def test_record_function(self):
        prof_result = self._record_function_with_param()
        found_test_1 = False
        found_test_2 = False
        found_test_3 = False
        found_test_4 = False
        for e in prof_result.function_events:
            if "## TEST 1 ##" == e.name:
                found_test_1 = True
                self.assertTrue(e.input_shapes == [[]])
            elif "## TEST 2 ##" == e.name:
                found_test_2 = True
                self.assertTrue(e.input_shapes == [[], [], [], [], [], [3, 4, 5]])
            elif "## TEST 3 ##" == e.name:
                found_test_3 = True
                self.assertTrue(e.input_shapes == [])
            elif "## TEST 4 ##" == e.name:
                found_test_4 = True
                self.assertTrue(e.input_shapes == [])
        self.assertTrue(found_test_1)
        self.assertTrue(found_test_2)
        self.assertTrue(found_test_3)
        self.assertTrue(found_test_4)

    def test_python_dispatch_mode_record_function(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class TestDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with _profile() as prof:
            with enable_python_dispatcher():
                with TestDispatchMode():
                    x = torch.randn(3, 4)
                    y = torch.sin(x)

        found_python_dispatch_mode = False
        for e in prof.function_events:
            if e.name == "PythonDispatchMode":
                found_python_dispatch_mode = True
                break
        self.assertTrue(
            found_python_dispatch_mode,
            "PythonDispatchMode record function not found in profiler events",
        )

    def test_python_subclass_record_function(self):
        class TestTensorSubclass(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                def unwrap(x):
                    return x.elem if isinstance(x, TestTensorSubclass) else x

                def wrap(x):
                    return TestTensorSubclass(x) if isinstance(x, torch.Tensor) else x

                unwrapped_args = tuple(unwrap(arg) for arg in args)
                unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
                result = func(*unwrapped_args, **unwrapped_kwargs)

                if isinstance(result, torch.Tensor):
                    return TestTensorSubclass(result)
                return result

        with _profile() as prof:
            with enable_python_dispatcher():
                x = TestTensorSubclass(torch.randn(3, 4))
                y = torch.sin(x)

        found_python_subclass = False
        for e in prof.function_events:
            if e.name == "PythonSubclass":
                found_python_subclass = True
                break
        self.assertTrue(
            found_python_subclass,
            "PythonSubclass record function not found in profiler events",
        )

    def test_chromium_event_logger_profiler_integration(self):
        """Verify ChromiumEventLogger emits RecordFunction events to profiler"""

        with _profile(use_kineto=kineto_available()) as prof:
            logger = ChromiumEventLogger()
            logger.log_event_start("test_compile_event", 0, {})
            logger.log_event_end(
                "test_compile_event", 1000, {}, 0, log_pt2_compile_event=False
            )

        found = False
        for e in prof.function_events:
            if e.name == "test_compile_event":
                found = True
                break
        self.assertTrue(
            found,
            "ChromiumEventLogger event not found in profiler output",
        )

    def test_chromium_logger_reset_cleans_open_events(self):
        """Verify reset() cleans up lingering record functions on exception"""
        from torch._dynamo.utils import ChromiumEventLogger

        with _profile(use_kineto=kineto_available()):
            logger = ChromiumEventLogger()

            logger.log_event_start("incomplete_event", 0, {})

            rf_dict = logger.get_record_functions()
            self.assertIn("incomplete_event", rf_dict)

            logger.reset()

            rf_dict = logger.get_record_functions()
            self.assertEqual(len(rf_dict), 0, "reset() failed to clean up open events")

    def test_chromium_logger_no_overhead_when_profiler_disabled(self):
        """Verify zero overhead when profiler is not enabled"""
        from torch._dynamo.utils import ChromiumEventLogger

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        logger = ChromiumEventLogger()

        logger.log_event_start("test_event", 0, {})
        logger.log_event_end("test_event", 1000, {}, 0, log_pt2_compile_event=False)

        rf_dict = logger.get_record_functions()
        self.assertEqual(
            len(rf_dict), 0, "RecordFunction created when profiler disabled"
        )


if __name__ == "__main__":
    run_tests()
