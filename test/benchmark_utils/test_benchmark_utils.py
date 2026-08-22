# Owner(s): ["module: unknown"]

import os
import re
import textwrap
import timeit
import unittest
from typing import Any

import expecttest
import numpy as np

import torch
import torch.utils.benchmark as benchmark_utils
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)


class TestBenchmarkUtils(TestCase):
    def regularizeAndAssertExpectedInline(
        self, x: Any, expect: str, indent: int = 12
    ) -> None:
        x_str: str = re.sub(
            "object at 0x[0-9a-fA-F]+>",
            "object at 0xXXXXXXXXXXXX>",
            x if isinstance(x, str) else repr(x),
        )
        if "\n" in x_str:
            # Indent makes the reference align at the call site.
            x_str = textwrap.indent(x_str, " " * indent)

        self.assertExpectedInline(x_str, expect, skip=1)

    def test_timer(self):
        timer = benchmark_utils.Timer(
            stmt="torch.ones(())",
        )
        sample = timer.timeit(5).median
        self.assertIsInstance(sample, float)

        median = timer.blocked_autorange(min_run_time=0.01).median
        self.assertIsInstance(median, float)

        # We set a very high threshold to avoid flakiness in CI.
        # The internal algorithm is tested in `test_adaptive_timer`
        median = timer.adaptive_autorange(threshold=0.5).median

        # Test that multi-line statements work properly.
        median = (
            benchmark_utils.Timer(
                stmt="""
                with torch.no_grad():
                    y = x + 1""",
                setup="""
                x = torch.ones((1,), requires_grad=True)
                for _ in range(5):
                    x = x + 1.0""",
            )
            .timeit(5)
            .median
        )
        self.assertIsInstance(sample, float)

    @slowTest
    @unittest.skipIf(IS_SANDCASTLE, "C++ timing is OSS only.")
    @unittest.skipIf(True, "Failing on clang, see 74398")
    def test_timer_tiny_fast_snippet(self):
        timer = benchmark_utils.Timer(
            "auto x = 1;(void)x;",
            timer=timeit.default_timer,
            language=benchmark_utils.Language.CPP,
        )
        median = timer.blocked_autorange().median
        self.assertIsInstance(median, float)

    @slowTest
    @unittest.skipIf(IS_SANDCASTLE, "C++ timing is OSS only.")
    @unittest.skipIf(True, "Failing on clang, see 74398")
    def test_cpp_timer(self):
        timer = benchmark_utils.Timer(
            """
                #ifndef TIMER_GLOBAL_CHECK
                static_assert(false);
                #endif

                torch::Tensor y = x + 1;
            """,
            setup="torch::Tensor x = torch::empty({1});",
            global_setup="#define TIMER_GLOBAL_CHECK",
            timer=timeit.default_timer,
            language=benchmark_utils.Language.CPP,
        )
        t = timer.timeit(10)
        self.assertIsInstance(t.median, float)

    class _MockTimer:
        _seed = 0

        _timer_noise_level = 0.05
        _timer_cost = 100e-9  # 100 ns

        _function_noise_level = 0.05
        _function_costs = (
            ("pass", 8e-9),
            ("cheap_fn()", 4e-6),
            ("expensive_fn()", 20e-6),
            ("with torch.no_grad():\n    y = x + 1", 10e-6),
        )

        def __init__(self, stmt, setup, timer, globals):
            self._random_state = np.random.RandomState(seed=self._seed)
            self._mean_cost = dict(self._function_costs)[stmt]

        def sample(self, mean, noise_level):
            return max(self._random_state.normal(mean, mean * noise_level), 5e-9)

        def timeit(self, number):
            return sum(
                [
                    # First timer invocation
                    self.sample(self._timer_cost, self._timer_noise_level),
                    # Stmt body
                    self.sample(self._mean_cost * number, self._function_noise_level),
                    # Second timer invocation
                    self.sample(self._timer_cost, self._timer_noise_level),
                ]
            )

    def test_adaptive_timer(self):
        class MockTimer(benchmark_utils.Timer):
            _timer_cls = self._MockTimer

        class _MockCudaTimer(self._MockTimer):
            # torch.cuda.synchronize is much more expensive than
            # just timeit.default_timer
            _timer_cost = 10e-6

            _function_costs = (
                self._MockTimer._function_costs[0],
                self._MockTimer._function_costs[1],
                # GPU should be faster once there is enough work.
                ("expensive_fn()", 5e-6),
            )

        class MockCudaTimer(benchmark_utils.Timer):
            _timer_cls = _MockCudaTimer

        m = MockTimer("pass").blocked_autorange(min_run_time=10)
        self.regularizeAndAssertExpectedInline(
            m,
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            pass
              Median: 7.98 ns
              IQR:    0.52 ns (7.74 to 8.26)
              125 measurements, 10000000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer("pass").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            pass
              Median: 7.86 ns
              IQR:    0.71 ns (7.63 to 8.34)
              6 measurements, 1000000 runs per measurement, 1 thread""",
        )

        # Check against strings so we can reuse expect infra.
        self.regularizeAndAssertExpectedInline(m.mean, """8.0013658357956e-09""")
        self.regularizeAndAssertExpectedInline(m.median, """7.983151323215967e-09""")
        self.regularizeAndAssertExpectedInline(len(m.times), """125""")
        self.regularizeAndAssertExpectedInline(m.number_per_run, """10000000""")

        self.regularizeAndAssertExpectedInline(
            MockTimer("cheap_fn()").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            cheap_fn()
              Median: 3.98 us
              IQR:    0.27 us (3.85 to 4.12)
              252 measurements, 10000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer("cheap_fn()").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            cheap_fn()
              Median: 4.16 us
              IQR:    0.22 us (4.04 to 4.26)
              4 measurements, 1000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer("expensive_fn()").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            expensive_fn()
              Median: 19.97 us
              IQR:    1.35 us (19.31 to 20.65)
              501 measurements, 1000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer("expensive_fn()").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            expensive_fn()
              Median: 20.79 us
              IQR:    1.09 us (20.20 to 21.29)
              4 measurements, 1000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("pass").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            pass
              Median: 7.92 ns
              IQR:    0.43 ns (7.75 to 8.17)
              13 measurements, 100000000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("pass").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            pass
              Median: 7.75 ns
              IQR:    0.57 ns (7.56 to 8.13)
              4 measurements, 10000000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("cheap_fn()").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            cheap_fn()
              Median: 4.04 us
              IQR:    0.30 us (3.90 to 4.19)
              25 measurements, 100000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("cheap_fn()").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            cheap_fn()
              Median: 4.09 us
              IQR:    0.38 us (3.90 to 4.28)
              4 measurements, 100000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("expensive_fn()").blocked_autorange(min_run_time=10),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            expensive_fn()
              Median: 4.98 us
              IQR:    0.31 us (4.83 to 5.13)
              20 measurements, 100000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockCudaTimer("expensive_fn()").adaptive_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            expensive_fn()
              Median: 5.01 us
              IQR:    0.28 us (4.87 to 5.15)
              4 measurements, 10000 runs per measurement, 1 thread""",
        )

        # Make sure __repr__ is reasonable for
        # multi-line / label / sub_label / description, but we don't need to
        # check numerics.
        multi_line_stmt = """
        with torch.no_grad():
            y = x + 1
        """

        self.regularizeAndAssertExpectedInline(
            MockTimer(multi_line_stmt).blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            stmt:
              with torch.no_grad():
                  y = x + 1

              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer(multi_line_stmt, sub_label="scalar_add").blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            stmt: (scalar_add)
              with torch.no_grad():
                  y = x + 1

              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer(
                multi_line_stmt,
                label="x + 1 (no grad)",
                sub_label="scalar_add",
            ).blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            x + 1 (no grad): scalar_add
              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer(
                multi_line_stmt,
                setup="setup_fn()",
                sub_label="scalar_add",
            ).blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            stmt: (scalar_add)
              with torch.no_grad():
                  y = x + 1

            setup: setup_fn()
              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 1 thread""",
        )

        self.regularizeAndAssertExpectedInline(
            MockTimer(
                multi_line_stmt,
                setup="""
                    x = torch.ones((1,), requires_grad=True)
                    for _ in range(5):
                        x = x + 1.0""",
                sub_label="scalar_add",
                description="Multi-threaded scalar math!",
                num_threads=16,
            ).blocked_autorange(),
            """\
            <torch.utils.benchmark.utils.common.Measurement object at 0xXXXXXXXXXXXX>
            stmt: (scalar_add)
              with torch.no_grad():
                  y = x + 1

            Multi-threaded scalar math!
            setup:
              x = torch.ones((1,), requires_grad=True)
              for _ in range(5):
                  x = x + 1.0

              Median: 10.06 us
              IQR:    0.54 us (9.73 to 10.27)
              20 measurements, 1000 runs per measurement, 16 threads""",
        )

    def test_compare(self):
        # Simulate several approaches.
        costs = (
            # overhead_optimized_fn()
            (1e-6, 1e-9),
            # compute_optimized_fn()
            (3e-6, 5e-10),
            # special_case_fn()  [square inputs only]
            (1e-6, 4e-10),
        )

        sizes = (
            (16, 16),
            (16, 128),
            (128, 128),
            (4096, 1024),
            (2048, 2048),
        )

        # overhead_optimized_fn()
        class _MockTimer_0(self._MockTimer):
            _function_costs = tuple(
                (f"fn({i}, {j})", costs[0][0] + costs[0][1] * i * j) for i, j in sizes
            )

        class MockTimer_0(benchmark_utils.Timer):
            _timer_cls = _MockTimer_0

        # compute_optimized_fn()
        class _MockTimer_1(self._MockTimer):
            _function_costs = tuple(
                (f"fn({i}, {j})", costs[1][0] + costs[1][1] * i * j) for i, j in sizes
            )

        class MockTimer_1(benchmark_utils.Timer):
            _timer_cls = _MockTimer_1

        # special_case_fn()
        class _MockTimer_2(self._MockTimer):
            _function_costs = tuple(
                (f"fn({i}, {j})", costs[2][0] + costs[2][1] * i * j)
                for i, j in sizes
                if i == j
            )

        class MockTimer_2(benchmark_utils.Timer):
            _timer_cls = _MockTimer_2

        results = []
        for i, j in sizes:
            results.append(
                MockTimer_0(
                    f"fn({i}, {j})",
                    label="fn",
                    description=f"({i}, {j})",
                    sub_label="overhead_optimized",
                ).blocked_autorange(min_run_time=10)
            )

            results.append(
                MockTimer_1(
                    f"fn({i}, {j})",
                    label="fn",
                    description=f"({i}, {j})",
                    sub_label="compute_optimized",
                ).blocked_autorange(min_run_time=10)
            )

            if i == j:
                results.append(
                    MockTimer_2(
                        f"fn({i}, {j})",
                        label="fn",
                        description=f"({i}, {j})",
                        sub_label="special_case (square)",
                    ).blocked_autorange(min_run_time=10)
                )

        def rstrip_lines(s: str) -> str:
            # VSCode will rstrip the `expected` string literal whether you like
            # it or not. So we have to rstrip the compare table as well.
            return "\n".join([i.rstrip() for i in s.splitlines(keepends=False)])

        compare = benchmark_utils.Compare(results)
        self.regularizeAndAssertExpectedInline(
            rstrip_lines(str(compare).strip()),
            """\
            [------------------------------------------------- fn ------------------------------------------------]
                                         |  (16, 16)  |  (16, 128)  |  (128, 128)  |  (4096, 1024)  |  (2048, 2048)
            1 threads: --------------------------------------------------------------------------------------------
                  overhead_optimized     |    1.3     |     3.0     |     17.4     |     4174.4     |     4174.4
                  compute_optimized      |    3.1     |     4.0     |     11.2     |     2099.3     |     2099.3
                  special_case (square)  |    1.1     |             |      7.5     |                |     1674.7

            Times are in microseconds (us).""",
        )

        compare.trim_significant_figures()
        self.regularizeAndAssertExpectedInline(
            rstrip_lines(str(compare).strip()),
            """\
            [------------------------------------------------- fn ------------------------------------------------]
                                         |  (16, 16)  |  (16, 128)  |  (128, 128)  |  (4096, 1024)  |  (2048, 2048)
            1 threads: --------------------------------------------------------------------------------------------
                  overhead_optimized     |     1      |     3.0     |      17      |      4200      |      4200
                  compute_optimized      |     3      |     4.0     |      11      |      2100      |      2100
                  special_case (square)  |     1      |             |       8      |                |      1700

            Times are in microseconds (us).""",
        )

        compare.colorize()
        columnwise_colored_actual = rstrip_lines(str(compare).strip())
        columnwise_colored_expected = textwrap.dedent(
            """\
            [------------------------------------------------- fn ------------------------------------------------]
                                         |  (16, 16)  |  (16, 128)  |  (128, 128)  |  (4096, 1024)  |  (2048, 2048)
            1 threads: --------------------------------------------------------------------------------------------
                  overhead_optimized     |     1      |  \x1b[92m\x1b[1m   3.0   \x1b[0m\x1b[0m  |  \x1b[2m\x1b[91m    17    \x1b[0m\x1b[0m  |      4200      |  \x1b[2m\x1b[91m    4200    \x1b[0m\x1b[0m
                  compute_optimized      |  \x1b[2m\x1b[91m   3    \x1b[0m\x1b[0m  |     4.0     |      11      |  \x1b[92m\x1b[1m    2100    \x1b[0m\x1b[0m  |      2100
                  special_case (square)  |  \x1b[92m\x1b[1m   1    \x1b[0m\x1b[0m  |             |  \x1b[92m\x1b[1m     8    \x1b[0m\x1b[0m  |                |  \x1b[92m\x1b[1m    1700    \x1b[0m\x1b[0m

            Times are in microseconds (us)."""
        )

        compare.colorize(rowwise=True)
        rowwise_colored_actual = rstrip_lines(str(compare).strip())
        rowwise_colored_expected = textwrap.dedent(
            """\
            [------------------------------------------------- fn ------------------------------------------------]
                                         |  (16, 16)  |  (16, 128)  |  (128, 128)  |  (4096, 1024)  |  (2048, 2048)
            1 threads: --------------------------------------------------------------------------------------------
                  overhead_optimized     |  \x1b[92m\x1b[1m   1    \x1b[0m\x1b[0m  |  \x1b[2m\x1b[91m   3.0   \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    17    \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    4200    \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    4200    \x1b[0m\x1b[0m
                  compute_optimized      |  \x1b[92m\x1b[1m   3    \x1b[0m\x1b[0m  |     4.0     |  \x1b[2m\x1b[91m    11    \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    2100    \x1b[0m\x1b[0m  |  \x1b[31m\x1b[1m    2100    \x1b[0m\x1b[0m
                  special_case (square)  |  \x1b[92m\x1b[1m   1    \x1b[0m\x1b[0m  |             |  \x1b[31m\x1b[1m     8    \x1b[0m\x1b[0m  |                |  \x1b[31m\x1b[1m    1700    \x1b[0m\x1b[0m

            Times are in microseconds (us)."""
        )

        def print_new_expected(s: str) -> None:
            print(f'{"":>12}"""\\', end="")
            for l in s.splitlines(keepends=False):
                print("\n" + textwrap.indent(repr(l)[1:-1], " " * 12), end="")
            print('"""\n')

        if expecttest.ACCEPT:
            # expecttest does not currently support non-printable characters,
            # so these two entries have to be updated manually.
            if columnwise_colored_actual != columnwise_colored_expected:
                print("New columnwise coloring:\n")
                print_new_expected(columnwise_colored_actual)

            if rowwise_colored_actual != rowwise_colored_expected:
                print("New rowwise coloring:\n")
                print_new_expected(rowwise_colored_actual)

        self.assertEqual(columnwise_colored_actual, columnwise_colored_expected)
        self.assertEqual(rowwise_colored_actual, rowwise_colored_expected)

    @unittest.skipIf(
        IS_WINDOWS and os.getenv("VC_YEAR") == "2019", "Random seed only accepts int32"
    )
    def test_fuzzer(self):
        fuzzer = benchmark_utils.Fuzzer(
            parameters=[
                benchmark_utils.FuzzedParameter(
                    "n", minval=1, maxval=16, distribution="loguniform"
                )
            ],
            tensors=[benchmark_utils.FuzzedTensor("x", size=("n",))],
            seed=0,
        )

        expected_results = [
            (0.7821, 0.0536, 0.9888, 0.1949, 0.5242, 0.1987, 0.5094),
            (0.7166, 0.5961, 0.8303, 0.005),
        ]

        for i, (tensors, _, _) in enumerate(fuzzer.take(2)):
            x = tensors["x"]
            self.assertEqual(x, torch.tensor(expected_results[i]), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    run_tests()
