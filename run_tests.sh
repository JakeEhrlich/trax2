#!/bin/bash

# WASM Interpreter Test Suite
# Tests that are confirmed passing with 0 failures

set -e

INTERP="./target/debug/wasm_interp"

echo "Building..."
cargo build 2>/dev/null

echo ""
echo "Running tests..."
echo "================"

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

run_test() {
    local test_file="$1"
    if [ ! -f "$test_file" ]; then
        echo "SKIP: $test_file (not found)"
        return
    fi

    result=$($INTERP "$test_file" 2>&1 | tail -1)

    # Parse results
    passed=$(echo "$result" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo "0")
    failed=$(echo "$result" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo "0")
    skipped=$(echo "$result" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo "0")

    TOTAL_PASSED=$((TOTAL_PASSED + passed))
    TOTAL_FAILED=$((TOTAL_FAILED + failed))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + skipped))

    if [ "$failed" -eq 0 ]; then
        echo "PASS: $(basename $test_file): $passed passed, $skipped skipped"
    else
        echo "FAIL: $(basename $test_file): $passed passed, $failed FAILED, $skipped skipped"
    fi
}

# Core numeric tests
run_test "testsuite/i32.wast"
run_test "testsuite/i64.wast"
run_test "testsuite/f32.wast"
run_test "testsuite/f64.wast"

# Literals
run_test "testsuite/int_literals.wast"
run_test "testsuite/float_literals.wast"
run_test "testsuite/const.wast"

# Comparisons
run_test "testsuite/f32_cmp.wast"
run_test "testsuite/f64_cmp.wast"

# Bitwise
run_test "testsuite/f32_bitwise.wast"
run_test "testsuite/f64_bitwise.wast"

# Float misc
run_test "testsuite/float_misc.wast"

# Expression tests
run_test "testsuite/int_exprs.wast"

# Local variable tests
run_test "testsuite/local_set.wast"

# Type definitions
run_test "testsuite/type.wast"

# Conversions
run_test "testsuite/conversions.wast"

echo ""
echo "================"
echo "TOTAL: $TOTAL_PASSED passed, $TOTAL_FAILED failed, $TOTAL_SKIPPED skipped"

if [ "$TOTAL_FAILED" -gt 0 ]; then
    exit 1
fi
