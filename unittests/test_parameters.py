# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import inspect


import reframe as rfm
from reframe.core.exceptions import ReframeSyntaxError


class NoParams(rfm.RunOnlyRegressionTest):
    pass


class TwoParams(NoParams):
    P0 = parameter(['a'])
    P1 = parameter(['b'])


class Abstract(TwoParams):
    '''An abstract test is a test with undefined parameters.'''
    P0 = parameter()


class ExtendParams(TwoParams):
    P1 = parameter(['c', 'd', 'e'], inherit_params=True)
    P2 = parameter(['f', 'g'])


def test_param_space_is_empty():
    class MyTest(NoParams):
        pass

    assert MyTest.param_space.is_empty()


def test_params_are_present():
    class MyTest(TwoParams):
        pass

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('b',)


def test_abstract_param():
    class MyTest(Abstract):
        # Add another abstract parameter
        P2 = parameter()

    assert MyTest.param_space['P0'] == ()
    assert MyTest.param_space['P1'] == ('b',)
    assert MyTest.param_space['P2'] == ()
    assert MyTest.param_space.undefined_params() == ['P0', 'P2']


def test_param_override():
    class MyTest(TwoParams):
        P1 = parameter(['-'])

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('-',)


def test_param_inheritance():
    class MyTest(TwoParams):
        P1 = parameter(['c'], inherit_params=True)

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('b', 'c',)


def test_filter_params():
    class MyTest(ExtendParams):
        # We make the return type of the filtering function a list to ensure
        # that other iterables different to tuples are also valid.
        P1 = parameter(inherit_params=True,
                       filter_params=lambda x: list(x[2:]))

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('d', 'e',)
    assert MyTest.param_space['P2'] == ('f', 'g',)


def test_wrong_filter():
    with pytest.raises(TypeError):
        class MyTest(ExtendParams):
            '''Filter function is not a function'''
            P1 = parameter(inherit_params=True, filter_params='not callable')

    with pytest.raises(TypeError):
        class MyTest(ExtendParams):
            '''Filter function takes more than 1 argument'''
            P1 = parameter(inherit_params=True, filter_params=lambda x, y: [])

    def bad_filter(x):
        raise RuntimeError('bad filter')

    with pytest.raises(RuntimeError):
        class Foo(ExtendParams):
            '''Filter function raises'''
            P1 = parameter(inherit_params=True, filter_params=bad_filter)

    with pytest.raises(ReframeSyntaxError):
        class Foo(ExtendParams):
            '''Wrong filter return type'''
            P1 = parameter(inherit_params=True, filter_params=lambda x: 1)


def test_is_abstract_test():
    class MyTest(Abstract):
        pass

    assert MyTest.is_abstract()


def test_is_not_abstract_test():
    class MyTest(TwoParams):
        pass

    assert not MyTest.is_abstract()


def test_param_len_is_zero():
    class MyTest(Abstract):
        pass

    assert len(MyTest.param_space) == 0


def test_extended_param_len():
    class MyTest(ExtendParams):
        pass

    assert len(MyTest.param_space) == 8


def test_instantiate_abstract_test():
    class MyTest(Abstract):
        pass

    test = MyTest()
    assert test.P0 is None
    assert test.P1 is None


def test_param_values_are_not_set():
    class MyTest(TwoParams):
        pass

    test = MyTest()
    assert test.P0 is None
    assert test.P1 is None


def test_consume_param_space():
    class MyTest(ExtendParams):
        pass

    for i in range(MyTest.num_variants):
        test = MyTest(variant_num=i)
        assert test.P0 is not None
        assert test.P1 is not None
        assert test.P2 is not None

    test = MyTest()
    assert test.P0 is None
    assert test.P1 is None
    assert test.P2 is None

    with pytest.raises(ValueError):
        test = MyTest(variant_num=i+1)


def test_inject_params_wrong_index():
    class MyTest(ExtendParams):
        pass

    inst = MyTest()
    with pytest.raises(RuntimeError):
        MyTest.param_space.inject(inst, params_index=MyTest.num_variants+1)


def test_simple_test_decorator():
    @rfm.simple_test
    class MyTest(ExtendParams):
        pass

    mod = inspect.getmodule(MyTest)
    tests = mod._rfm_test_registry.instantiate_all()
    assert len(tests) == 8
    for test in tests:
        assert test.P0 is not None
        assert test.P1 is not None
        assert test.P2 is not None


def test_param_space_clash():
    class Spam(rfm.RegressionTestPlugin):
        P0 = parameter([1])

    class Ham(rfm.RegressionTestPlugin):
        P0 = parameter([2])

    with pytest.raises(ReframeSyntaxError):
        class Eggs(Spam, Ham):
            '''Trigger error from param name clashing.'''


def test_multiple_inheritance():
    class Spam(rfm.RegressionTestPlugin):
        P0 = parameter()

    class Ham(rfm.RegressionTestPlugin):
        P0 = parameter([2])

    class Eggs(Spam, Ham):
        '''Multiple inheritance is allowed if only one class defines P0.'''


def test_namespace_clash():
    class Spam(rfm.RegressionTest):
        foo = variable(int)

    with pytest.raises(ReframeSyntaxError):
        class Ham(Spam):
            foo = parameter([1])


def test_double_declare():
    with pytest.raises(ReframeSyntaxError):
        class MyTest(rfm.RegressionTest):
            P0 = parameter([1, 2, 3])
            P0 = parameter()


def test_overwrite_param():
    with pytest.raises(ReframeSyntaxError):
        class MyTest(TwoParams):
            P0 = [1, 2, 3]


def test_param_deepcopy():
    '''Test that there is no cross-class pollution.

    Each instance must deal with its own copies of the parameters.
    '''
    class MyParam:
        def __init__(self, val):
            self.val = val

    class Base(rfm.RegressionTest):
        p0 = parameter([MyParam(1), MyParam(2)])

    class Foo(Base):
        def __init__(self):
            self.p0.val = -20

    class Bar(Base):
        pass

    assert Foo(variant_num=0).p0.val == -20
    assert Foo(variant_num=1).p0.val == -20
    assert Bar(variant_num=0).p0.val == 1
    assert Bar(variant_num=1).p0.val == 2


def test_param_access():
    with pytest.raises(ReframeSyntaxError):
        class Foo(rfm.RegressionTest):
            p = parameter([1, 2, 3])
            x = f'accessing {p!r} in the class body is disallowed.'


def test_param_space_read_only():
    class Foo(rfm.RegressionTestPlugin):
        pass

    with pytest.raises(ValueError):
        Foo.param_space['a'] = (1, 2, 3)


def test_parameter_override():
    with pytest.raises(ReframeSyntaxError):
        # Trigger the check from MetaNamespace
        class MyTest(rfm.RegressionTest):
            p = parameter([1, 2])
            p = 0

    # Trigger the check in the extend method from ParameterSpace
    class Foo(rfm.RegressionTest):
        p = parameter([1, 2])

    with pytest.raises(ReframeSyntaxError):
        class Bar(Foo):
            p = 0


def test_override_regular_attribute():
    class Foo(rfm.RegressionTest):
        p = 4
        p = parameter([1, 2])

    assert Foo.p.values == (1, 2,)


def test_override_parameter():
    with pytest.raises(ReframeSyntaxError):
        class Foo(rfm.RegressionTest):
            p = parameter([1, 2])
            p = 1

    with pytest.raises(ReframeSyntaxError):
        class Foo(rfm.RegressionTest):
            p = parameter([1, 2])
            p += 1


def test_local_paramspace_is_empty():
    class MyTest(rfm.RegressionTest):
        p = parameter([1, 2, 3])

    assert len(MyTest._rfm_local_param_space) == 0


def test_class_attr_access():
    class MyTest(rfm.RegressionTest):
        p = parameter([1, 2, 3])

    assert MyTest.p.values == (1, 2, 3,)
    with pytest.raises(ReframeSyntaxError,
                       match='assignment to a test parameter is not allowed'):
        MyTest.p = (4, 5,)


def test_get_variant_nums():
    class MyTest(rfm.RegressionTest):
        p = parameter(range(10))

    with pytest.raises(NameError):
        MyTest.param_space.get_variant_nums(p0=lambda x: x == 2)

    with pytest.raises(ValueError):
        MyTest.param_space.get_variant_nums(p=lambda x, y: x == 2)


# Tests for bundled parameters


class BundledParams(NoParams):
    P0 = parameter(["a", "b"], bundle=True)
    P1 = parameter([1, 2, 3], bundle=True)


class MixedParams(NoParams):
    # Non-bundled parameter creates separate variants
    P0 = parameter(["x", "y"])
    # Bundled parameters run in a loop within each job
    P1 = parameter([1, 2], bundle=True)
    P2 = parameter(["a", "b"], bundle=True)


def test_bundled_param_is_bundled():
    class MyTest(BundledParams):
        pass

    assert MyTest.param_space.has_bundled_params()
    assert "P0" in MyTest.param_space.bundled_params
    assert "P1" in MyTest.param_space.bundled_params


def test_bundled_param_single_variant():
    """Tests with only bundled params should have only 1 variant."""

    class MyTest(BundledParams):
        pass

    # Only 1 variant since all params are bundled
    assert len(MyTest.param_space) == 1
    assert MyTest.num_variants == 1


def test_bundled_param_combinations():
    """Bundled combinations should contain all value combinations."""

    class MyTest(BundledParams):
        pass

    combos = MyTest.param_space.bundled_combinations
    # 2 values for P0 * 3 values for P1 = 6 combinations
    assert len(combos) == 6


def test_mixed_bundled_variants():
    """Tests with mixed params should have variants for non-bundled only."""

    class MyTest(MixedParams):
        pass

    # 2 variants from non-bundled P0 ('x' and 'y')
    assert len(MyTest.param_space) == 2
    assert MyTest.num_variants == 2

    # But bundled combinations have 2*2=4 combinations
    assert len(MyTest.param_space.bundled_combinations) == 4


def test_bundled_param_instantiate():
    """Bundled params should be initialized with first value."""

    class MyTest(BundledParams):
        pass

    # Instantiate the single variant
    test = MyTest(variant_num=0)

    # Bundled params should be set to their first value
    assert test.P0 == "a"
    assert test.P1 == 1


def test_mixed_bundled_instantiate():
    """Non-bundled params vary per variant, bundled start at first value."""

    class MyTest(MixedParams):
        pass

    # Variant 0: P0='x'
    test0 = MyTest(variant_num=0)
    assert test0.P0 == "x"
    assert test0.P1 == 1  # First bundled value
    assert test0.P2 == "a"  # First bundled value

    # Variant 1: P0='y'
    test1 = MyTest(variant_num=1)
    assert test1.P0 == "y"
    assert test1.P1 == 1  # First bundled value
    assert test1.P2 == "a"  # First bundled value


def test_bundled_parameters_property():
    """Test the bundled_parameters property on test instances."""

    class MyTest(BundledParams):
        pass

    test = MyTest(variant_num=0)
    bundle_info = test.bundled_parameters

    assert bundle_info["names"] == ["P0", "P1"]
    assert len(bundle_info["combinations"]) == 6

    # Check first combination
    first = bundle_info["combinations"][0]
    assert first["P0"] == "a"
    assert first["P1"] == 1


def test_has_bundled_params_method():
    """Test the has_bundled_params method."""

    class WithBundle(BundledParams):
        pass

    class WithoutBundle(TwoParams):
        pass

    test_with = WithBundle(variant_num=0)
    test_without = WithoutBundle(variant_num=0)

    assert test_with.has_bundled_params()
    assert not test_without.has_bundled_params()


def test_get_bundled_combination():
    """Test getting a specific bundled combination."""

    class MyTest(BundledParams):
        pass

    combo0 = MyTest.param_space.get_bundled_combination(0)
    assert combo0 == {"P0": "a", "P1": 1}

    combo5 = MyTest.param_space.get_bundled_combination(5)
    assert combo5 == {"P0": "b", "P1": 3}


def test_bundle_with_inheritance():
    """Test that bundled parameters work with inheritance."""

    class Base(NoParams):
        P0 = parameter([1, 2], bundle=True)

    class Derived(Base):
        P1 = parameter(["a", "b"], bundle=True)

    assert Derived.param_space.has_bundled_params()
    assert len(Derived.param_space) == 1
    assert len(Derived.param_space.bundled_combinations) == 4
