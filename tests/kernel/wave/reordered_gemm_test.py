import math
import pytest
import torch
from torch.nn import functional as F
from dataclasses import replace

import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_empty,
    device_arange,
    device_randint,
    device_full,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.reordered_matmul import (
    get_reordered_matmul
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
import os
from enum import Enum
from torch.testing import assert_close

from .common.utils import (
    require_e2e,
    require_cdna3,
    enable_scheduling_barriers,
    dump_generated_mlir,
    param_bool,
)
from .common.shapes import get_test_shapes, construct_test_name

@require_e2e
@pytest.mark.parametrize("shape", [(512, 512, 512), (1792, 1792, 1792)])
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE, SchedulingType.PREFETCH, SchedulingType.MODULO],
)
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testNonTransposeGemm(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    M = shape[0]
    N = shape[1]
    K = shape[2]
    # Workgroup tile sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    # Group size
    GROUP_SIZE_N = 4

    reordered_gemm, hyperparams = get_reordered_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_N, mfma_variant)

    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        #dynamic_symbols=dynamic_symbols,
        #dynamic_symbols_map=dynamic_symbols_map,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + perf_filename) if dump_perf else None
        ),
    )
    options = set_default_run_config(options)
    reordered_gemm = wave_compile(options, reordered_gemm)
    a = device_randn(shape[0], shape[2], dtype=torch.float16)
    b = device_randn(shape[2], shape[1], dtype=torch.float16)
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    asm = reordered_gemm(a, b, c)

    if dump_generated_mlir:
        filename = f"wave_gemm_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    if run_bench:
        if dump_perf is not None:
            options.benchmark_results_file = os.path.join(
                dump_perf, "iree_" + perf_filename
            )
    # TODO: switch to comparison against generated iree_ref
    print(f"M: {shape[0]}, N: {shape[1]}\n")
    torch_ref = torch.matmul(a, b)
    assert_close(
        c.to(torch.float16), torch_ref, atol=1e-2, rtol=1e-2, check_device=False
    )