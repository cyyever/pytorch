# Owner(s): ["module: functorch"]

import random

import torch
import torch.fx as fx
from functorch import make_fx
from torch._functorch.compile_utils import fx_graph_cse
from torch.testing._internal.common_utils import run_tests, TestCase


HAS_CUDA = torch.cuda.is_available()


# check if the CSE modified graph of f has delta less nodes, and do not reduce the number of nodes further on a second pass.
# delta is an integer >= -1. If delta = -1, only check if the new graph
#   has less or equal number of nodes
def check(f, t, delta, check_val=True, graph_input=False):
    if graph_input:
        fx_g = f
    else:
        fx_g = make_fx(f)(t)
    new_graph = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_graph)

    # the number of nodes decrease/ or stay the same
    old_num_nodes = len(fx_g.graph.nodes)
    new_num_nodes = len(new_graph.nodes)
    if delta == -1:
        if old_num_nodes < new_num_nodes:
            raise AssertionError(
                f"number of nodes increased {old_num_nodes}, {new_num_nodes}"
            )
    else:
        if old_num_nodes != new_num_nodes + delta:
            raise AssertionError(
                f"number of nodes not the same {old_num_nodes - delta}, {new_num_nodes}\n {fx_g.graph} \n {new_graph}"
            )

    # a second pass should not reduce more nodes
    pass_2_graph = fx_graph_cse(new_graph)
    pass_2_num_nodes = len(pass_2_graph.nodes)
    if pass_2_num_nodes != new_num_nodes:
        raise AssertionError(
            f"second pass graph has less node {pass_2_num_nodes}, {new_num_nodes}\n {new_graph} \n {pass_2_graph}"
        )

    # check correctness
    if check_val:
        true_result = fx_g(t)
        our_result = new_g(t)
        if true_result is None:  # both return None
            if our_result is not None:
                raise AssertionError(f"true result is None, CSE result is {our_result}")
        else:  # results returned are the same
            if not torch.all(true_result == our_result):
                raise AssertionError(
                    f"results are different {true_result}, {our_result}"
                )


class NoChangeTestCase(TestCase):
    def test_nochange(self):
        def f(x):
            a = x + 1
            b = x + a
            a = x
            d = x + a
            return b + d

        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_empty(self):
        def f(x):
            pass

        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_rand_like(self):
        def f(x):
            a = torch.rand_like(x)
            b = torch.rand_like(x)
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_rand_n(self):
        def f(x):
            a = torch.randn(4)
            b = torch.randn(4)
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_hash_with_numbers(self):
        # Test to repro issue with fx_graph_cse when
        # hash((primals_2, 1.0)) == hash((primals_2, 1))

        if torch._dynamo.is_compiling():
            self.skipTest("Unsupported if test run is compiled")

        def f(inpt, osize):
            size = inpt.shape[-1]
            s1 = size - 1
            s2 = size - 1.0
            scale = s2 / (osize - 1.0)
            inpt = torch.clamp(inpt, 0, s1)
            return scale * inpt

        # Fetch dynamic graph
        gms = []

        def toy_backend(gm, _):
            gms.append(gm)
            return gm.forward

        torch._dynamo.reset()
        fn = torch.compile(backend=toy_backend, dynamic=True)(f)

        t = torch.rand(3, 100)
        _ = fn(t, 50)
        if len(gms) != 1:
            raise AssertionError(f"Expected 1 graph module, got {len(gms)}: {gms}")
        fx_g = gms[0]
        check(fx_g, None, 0, check_val=False, graph_input=True)

    def test_neg_nan_not_merged(self):
        def f(x):
            a = torch.full_like(x, float("nan"))
            b = torch.full_like(x, -float("nan"))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_neg_zero_not_merged(self):
        def f(x):
            a = torch.full_like(x, 0.0)
            b = torch.full_like(x, -0.0)
            return torch.stack([a.reciprocal(), b.reciprocal()])

        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_complex_neg_zero_not_merged(self):
        def f(x):
            y = x.to(torch.cfloat)
            a = torch.full_like(y, complex(0.0, 0.0))
            b = torch.full_like(y, complex(-0.0, 0.0))
            return torch.stack([a.real.reciprocal(), b.real.reciprocal()])

        t = torch.randn(2, 2)
        check(f, t, 0)


class ReduceTestCase(TestCase):
    def test_immutable_list_type(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1)
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_immutable_list_multiple_entries(self):
        def f(x):
            a = x.sum(dim=[0, 1])
            b = x.sum(dim=[0, 1])
            c = x.sum(dim=1)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple(self):
        def f(x):
            a = x.cos()
            b = x.cos()
            c = a + a
            d = b + b
            return c + d

        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple_2(self):
        def f(x):
            a = x.cos().sin()
            b = x.cos().sin()
            c = a + a
            d = b + b
            return c + d

        t = torch.randn(1)
        check(f, t, 3)

    def test_two_args_default(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=False)
            c = x.sum(dim=1, keepdim=False)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 3)

    def test_two_args(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=True)
            c = x.sum(dim=1, keepdim=True)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple_multiple_same_ops(self):
        def f(x):
            a = x.sum()
            b = x.sum()
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 3)

    def test_nested_immutable_list_type(self):
        def f(x):
            a = torch.cat((x, x))
            b = torch.cat((x, x))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1)

    def test_kwarg(self):
        def f(x):
            a = torch.ones_like(x)
            b = torch.ones_like(x)
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1)

    def test_nan_full_deduplication(self):
        def f(x):
            a = torch.full_like(x, float("nan"))
            b = torch.full_like(x, float("nan"))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1, check_val=False)

    def test_nan_dedup_non_factory_op(self):
        def f(x):
            a = x.clamp(min=float("nan"))
            b = x.clamp(min=float("nan"))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1, check_val=False)

    def test_nan_dedup_constant_pad(self):
        def f(x):
            a = torch.nn.functional.pad(x, (1, 1, 1, 1), value=float("nan"))
            b = torch.nn.functional.pad(x, (1, 1, 1, 1), value=float("nan"))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1, check_val=False)

    def test_complex_nan_full_deduplication(self):
        def f(x):
            y = x.to(torch.cfloat)
            a = torch.full_like(y, complex(float("nan"), 0.0))
            b = torch.full_like(y, complex(float("nan"), 0.0))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1, check_val=False)


class RandomOpTestCase(TestCase):
    def test_random(self):
        def f(x):
            vals = [x]
            ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
            for _ in range(100):
                new_val = random.choice(ops)(random.choice(vals))
                vals.append(new_val)
            return vals[-1]

        fx_g = fx.symbolic_trace(f)
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
        t = torch.randn(2, 2)

        for _ in range(30):
            check(fx_g, t, -1, graph_input=True)


if __name__ == "__main__":
    run_tests()
