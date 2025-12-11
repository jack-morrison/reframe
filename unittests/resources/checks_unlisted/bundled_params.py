# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class BundledParamTest(rfm.RunOnlyRegressionTest):
    """Test with bundled parameters that run in a single job."""

    valid_systems = ["*"]
    valid_prog_environs = ["*"]
    executable = "echo"

    # Bundled parameters - these will run in a loop within a single job
    message_size = parameter([128, 256, 512], bundle=True)
    precision = parameter(["single", "double"], bundle=True)

    # Non-bundled parameter - creates separate test variants
    backend = parameter(["cpu", "gpu"])

    @run_before("run")
    def set_exec_opts(self):
        # Access the bundled parameter values via environment variables
        # Environment vars are named <TESTNAME>_<PARAM>, e.g.:
        # BUNDLEDPARAMTEST_MESSAGE_SIZE and BUNDLEDPARAMTEST_PRECISION
        self.executable_opts = [
            f"backend={self.backend}",
            "size=$BUNDLEDPARAMTEST_MESSAGE_SIZE",
            "precision=$BUNDLEDPARAMTEST_PRECISION",
        ]

    @sanity_function
    def validate_all_iterations(self):
        # Check that all bundle iterations produced output
        bundle_info = self.bundled_parameters
        num_iterations = len(bundle_info["combinations"])

        checks = []
        for i in range(num_iterations):
            marker = f"=== REFRAME BUNDLE ITERATION {i} ==="
            checks.append(sn.assert_found(marker, self.stdout))

        return sn.all(checks)

    @run_before("performance")
    def set_perf_vars(self):
        # Extract performance metrics for each bundle iteration
        self.perf_variables = {}
        for idx, params, output in self.iter_bundle_outputs():
            size = params["message_size"]
            prec = params["precision"]
            # In a real test, you'd extract actual metrics from the output
            # Here we just use the iteration index as a dummy value
            key = f"metric_{size}_{prec}"
            self.perf_variables[key] = sn.make_performance_function(
                sn.defer(float(idx)), "units"
            )


@rfm.simple_test
class SimpleBundledTest(rfm.RunOnlyRegressionTest):
    """Simple test with only bundled parameters."""

    valid_systems = ["*"]
    valid_prog_environs = ["*"]
    executable = "echo"

    # Only bundled parameters - single test variant, single job
    value = parameter([1, 2, 3, 4, 5], bundle=True)

    @run_before("run")
    def set_exec_opts(self):
        # Env var is SIMPLEBUNDLEDTEST_VALUE
        self.executable_opts = ["value=$SIMPLEBUNDLEDTEST_VALUE"]

    @sanity_function
    def validate(self):
        # Check all iterations ran
        return sn.assert_found(
            "=== REFRAME BUNDLED EXECUTION END ===", self.stdout
        )


@rfm.simple_test
class MixedParamTest(rfm.RunOnlyRegressionTest):
    """Test with both bundled and non-bundled parameters."""

    valid_systems = ["*"]
    valid_prog_environs = ["*"]
    executable = "echo"

    # Non-bundled: creates 2 test variants (2 jobs)
    mode = parameter(["fast", "slow"])

    # Bundled: runs 3 iterations within each job
    iteration = parameter([1, 2, 3], bundle=True)

    @run_before("run")
    def set_exec_opts(self):
        # Env var is MIXEDPARAMTEST_ITERATION
        self.executable_opts = [f"mode={self.mode}", "iter=$MIXEDPARAMTEST_ITERATION"]

    @sanity_function
    def validate(self):
        # mode is a regular parameter, directly accessible
        # iteration values are looped within the job
        checks = [
            sn.assert_found(f"mode={self.mode}", self.stdout),
            sn.assert_found("=== REFRAME BUNDLED EXECUTION END ===", self.stdout),
        ]
        return sn.all(checks)
