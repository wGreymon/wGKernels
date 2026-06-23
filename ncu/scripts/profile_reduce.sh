#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <benchmark_executable> [benchmark_args...]"
    echo
    echo "environment overrides:"
    echo "  WGKERNEL_NCU_SET=basic|detailed|full|roofline"
    echo "  WGKERNEL_NCU_KERNEL='regex:reduce_sum_v2_kernel'"
    echo "  WGKERNEL_NCU_LAUNCH_SKIP=20"
    echo "  WGKERNEL_NCU_LAUNCH_COUNT=1"
    echo "  WGKERNEL_NCU_EXPORT=tests/cuda/test_reduce/profiling/reduce_sum_basic"
    exit 1
fi

benchmark_executable="$1"
shift

ncu_args=(
    --set "${WGKERNEL_NCU_SET:-basic}"
    --kernel-name-base demangled
    --target-processes all
    --kernel-name "${WGKERNEL_NCU_KERNEL:-regex:reduce_.*kernel}"
    --launch-skip "${WGKERNEL_NCU_LAUNCH_SKIP:-20}"
    --launch-count "${WGKERNEL_NCU_LAUNCH_COUNT:-1}"
)

if [[ -n "${WGKERNEL_NCU_EXPORT:-}" ]]; then
    ncu_args+=(--export "${WGKERNEL_NCU_EXPORT}" --force-overwrite)
fi

ncu "${ncu_args[@]}" "${benchmark_executable}" "$@"
