"""
Test script for exposure table upload + VaR/CVaR calculation.

Validates that the system works with ANY price data file,
as long as the exposure table products match the loaded data.

Tests:
1. Oil products (Merged_Futures_Spot_Data.xlsx) - products with contract months
2. Chinese commodities (group VaR model(2).XLSX) - products without contract months
3. Edge cases: invalid products, empty positions, mixed valid/invalid
"""

import io
import sys
import os
import numpy as np
import pandas as pd
import openpyxl

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import from var_calculator (without Streamlit)
with open(os.path.join(os.path.dirname(__file__), "var_calculator.py"), encoding="utf-8") as f:
    code = f.read()
exec(compile(code.split("def main():")[0], "var_calculator.py", "exec"))


def create_test_excel(rows: list) -> io.BytesIO:
    """Create an in-memory Excel file from row data."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["品种", "合约月份", "现货持仓", "期货持仓", "单位"])
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def run_test(test_name, data_file, exposure_rows, expect_success=True):
    """Run a single end-to-end test."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"Data file: {data_file}")
    print(f"{'='*60}")

    # Step 1: Load price data
    try:
        di = DataIngestion(data_file)
        metadata = di.get_asset_metadata()
        returns = di.get_returns(250)
        prices = di.get_latest_prices()
        print(f"  [OK] Loaded {len(metadata)} assets, {returns.shape[0]} days of returns")
    except Exception as e:
        print(f"  [FAIL] Data loading error: {e}")
        return False

    # Step 2: Extract available products
    available_products = _extract_base_products(list(returns.columns))
    base_products = sorted({
        col.rsplit("_", 1)[0].split(" ")[0]
        for col in returns.columns
    })
    print(f"  [OK] Available base products ({len(base_products)}): {', '.join(base_products[:10])}...")

    # Step 3: Create and parse exposure table
    exposure_file = create_test_excel(exposure_rows)
    try:
        exposure_df, parse_warnings = parse_exposure_table(
            exposure_file, valid_products=available_products
        )
        print(f"  [OK] Parsed {len(exposure_df)} exposure rows")
        if parse_warnings:
            for w in parse_warnings:
                print(f"    WARNING: {w}")
    except ValueError as e:
        if not expect_success:
            print(f"  [OK] Expected failure: {e}")
            return True
        print(f"  [FAIL] Parse error: {e}")
        return False

    # Step 4: Build position vector
    pos_vec, active_pos, build_warnings = build_exposure_position_vector(
        exposure_df, returns, prices, metadata, fx_rate=7.0
    )
    print(f"  [OK] Built position vector: {len(active_pos)} active positions")
    if build_warnings:
        for w in build_warnings:
            print(f"    WARNING: {w}")

    if np.allclose(pos_vec, 0):
        if not expect_success:
            print(f"  [OK] Expected: no valid positions mapped")
            return True
        print(f"  [FAIL] All positions are zero - no mapping succeeded")
        return False

    # Step 5: Calculate VaR + CVaR
    try:
        engine = RiskEngine(returns, decay_factor=0.94)
        results = engine.get_var_cvar_results(pos_vec)
        print(f"  [OK] VaR/CVaR calculated successfully:")
        print(f"    1-Day VaR 95%:  {results['1-Day']['VaR_95%']:>15,.2f}")
        print(f"    1-Day CVaR 95%: {results['1-Day']['CVaR_95%']:>15,.2f}")
        print(f"    1-Day VaR 99%:  {results['1-Day']['VaR_99%']:>15,.2f}")
        print(f"    1-Day CVaR 99%: {results['1-Day']['CVaR_99%']:>15,.2f}")
        print(f"    10-Day VaR 95%: {results['10-Day']['VaR_95%']:>15,.2f}")
        print(f"    10-Day VaR 99%: {results['10-Day']['VaR_99%']:>15,.2f}")
    except Exception as e:
        print(f"  [FAIL] VaR calculation error: {e}")
        return False

    # Step 6: Validate results are reasonable
    for horizon in results:
        for metric, value in results[horizon].items():
            if value <= 0:
                print(f"  [FAIL] {horizon} {metric} = {value} (should be > 0)")
                return False
            if np.isnan(value) or np.isinf(value):
                print(f"  [FAIL] {horizon} {metric} = {value} (NaN/Inf)")
                return False

    # CVaR should always be >= VaR
    for horizon in results:
        if results[horizon]["CVaR_95%"] < results[horizon]["VaR_95%"] * 0.99:
            print(f"  [FAIL] {horizon} CVaR_95% < VaR_95% (CVaR must >= VaR)")
            return False
        if results[horizon]["CVaR_99%"] < results[horizon]["VaR_99%"] * 0.99:
            print(f"  [FAIL] {horizon} CVaR_99% < VaR_99% (CVaR must >= VaR)")
            return False

    # 10-Day should be ~sqrt(10) * 1-Day
    ratio = results["10-Day"]["VaR_95%"] / results["1-Day"]["VaR_95%"]
    expected_ratio = np.sqrt(10)
    if abs(ratio - expected_ratio) > 0.01:
        print(f"  [FAIL] 10-Day/1-Day ratio = {ratio:.4f}, expected {expected_ratio:.4f}")
        return False

    # Position details
    print(f"\n  Position breakdown:")
    for p in active_pos:
        print(f"    {p['ProductCode']:12s} {p['PositionType']:4s} "
              f"qty={p['Quantity']:>10,.0f} price={p['Price']:>10,.2f} "
              f"notional={p['Notional_CNY']:>15,.0f} {p['Direction']}")

    print(f"\n  [PASS] All checks passed!")
    return True


if __name__ == "__main__":
    results = []

    # =========================================================================
    # Test Group 1: Oil products (Merged_Futures_Spot_Data.xlsx)
    # =========================================================================
    data_file_1 = os.path.join(os.path.dirname(__file__), "Merged_Futures_Spot_Data.xlsx")

    if os.path.exists(data_file_1):
        # Test 1a: Oil products with contract months
        results.append(("Oil: spot + futures with months", run_test(
            "Oil products - spot + futures with contract months",
            data_file_1,
            [
                ["SG380", "Apr26", 3000, -2000, "MT"],
                ["SG180", None, 5000, None, "MT"],
                ["Brt Fut", "Jun26", None, 5000, "BBL"],
            ],
        )))

        # Test 1b: Single product, single position
        results.append(("Oil: single position", run_test(
            "Oil products - single product",
            data_file_1,
            [
                ["GO 10ppm", "Apr26", None, -3000, "MT"],
            ],
        )))

        # Test 1c: All 5 oil products
        results.append(("Oil: all 5 products", run_test(
            "Oil products - all 5 products",
            data_file_1,
            [
                ["SG180", "Apr26", 5000, -3000, "MT"],
                ["SG380", "May26", 2000, -1000, "MT"],
                ["MF 0.5", "Jun26", -1000, 500, "MT"],
                ["GO 10ppm", "Apr26", 3000, -2000, "MT"],
                ["Brt Fut", "Sep26", None, 10000, "BBL"],
            ],
        )))

        # Test 1d: Invalid product mixed with valid
        results.append(("Oil: mixed valid/invalid", run_test(
            "Oil products - invalid product should warn and skip",
            data_file_1,
            [
                ["SG380", "Apr26", 3000, None, "MT"],
                ["CUFI", None, 100, None, "TON"],  # Not in oil data!
                ["Brt Fut", "Jun26", None, 5000, "BBL"],
            ],
        )))
    else:
        print(f"\n[SKIP] {data_file_1} not found")

    # =========================================================================
    # Test Group 2: Chinese commodities (group VaR model(2).XLSX)
    # =========================================================================
    data_file_2 = os.path.join(
        os.path.dirname(__file__),
        "..", "ITG-X GROUP DATA", "group VaR model(2).XLSX"
    )
    data_file_2 = os.path.normpath(data_file_2)

    if os.path.exists(data_file_2):
        # Test 2a: Chinese commodities (no contract months, bare product IDs)
        results.append(("Commodity: basic", run_test(
            "Chinese commodities - basic (CUFI, AUFI, RBFI)",
            data_file_2,
            [
                ["CUFI", None, 100, -50, "TON"],
                ["AUFI", None, 10, None, "G"],
                ["RBFI", None, 500, -300, "TON"],
            ],
        )))

        # Test 2b: Multi-commodity portfolio
        results.append(("Commodity: multi", run_test(
            "Chinese commodities - large portfolio",
            data_file_2,
            [
                ["CUFI", None, 200, -100, "TON"],
                ["ALFI", None, 300, None, "TON"],
                ["ZNFI", None, None, -500, "TON"],
                ["SCFI", None, 100, -50, "BUK"],
                ["FUFI", None, 1000, None, "TON"],
                ["LMECUFI", None, 50, -30, "TON"],
            ],
        )))

        # Test 2c: Oil product in commodity data should fail
        results.append(("Commodity: oil product rejected", run_test(
            "Chinese commodities - oil product should be rejected",
            data_file_2,
            [
                ["SG380", "Apr26", 3000, None, "MT"],  # Not in commodity data!
            ],
            expect_success=False,
        )))

        # Test 2d: Mix of valid commodity + invalid oil product
        results.append(("Commodity: mixed", run_test(
            "Chinese commodities - mixed valid/invalid",
            data_file_2,
            [
                ["CUFI", None, 100, None, "TON"],
                ["SG380", None, 3000, None, "MT"],  # Invalid for this dataset
                ["RBFI", None, 500, None, "TON"],
            ],
        )))
    else:
        print(f"\n[SKIP] {data_file_2} not found")

    # =========================================================================
    # Test Group 3: Edge cases
    # =========================================================================
    if os.path.exists(data_file_1):
        # Test 3a: All zeros should fail
        results.append(("Edge: all zeros", run_test(
            "Edge case - all zero positions",
            data_file_1,
            [
                ["SG380", "Apr26", 0, 0, "MT"],
            ],
            expect_success=False,
        )))

        # Test 3b: Missing unit (should default)
        results.append(("Edge: missing unit", run_test(
            "Edge case - missing unit column value",
            data_file_1,
            [
                ["SG380", "Apr26", 3000, None, None],
            ],
        )))

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
    else:
        print(f"\n  {total - passed} TESTS FAILED!")
        sys.exit(1)
