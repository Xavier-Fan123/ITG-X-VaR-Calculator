"""
ITG-X系统 - 集团风险价值计算器
ITG-X System - Group Value at Risk (VaR) Calculator

A production-grade, data-driven VaR calculator using:
- Parametric VaR (Variance-Covariance Method)
- EWMA (Exponentially Weighted Moving Average) for volatility modeling
- Full correlation matrix for diversification/netting effects
- Basis risk capture between futures (Settlement) and spot prices

Author: Xavier Fan
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

warnings.filterwarnings("ignore")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AssetMetadata:
    """Metadata for a single asset (product + price type combination)."""
    asset_id: str           # e.g., "CUFI_Settlement"
    product_id: str         # e.g., "CUFI"
    product_name: str       # e.g., "沪铜加权-东方财富"
    price_type: str         # "Settlement" or "Spot"
    unit: str               # e.g., "TON", "KG", "G", "BUK"
    currency: str           # "CNY" or "USD"

    @property
    def display_name(self) -> str:
        """Human-readable display name for UI."""
        price_label = "期货" if self.price_type == "Settlement" else "现货"
        return f"{self.product_name} ({price_label})"

    @property
    def short_name(self) -> str:
        """Short name for tables."""
        return f"{self.product_id}_{self.price_type}"


# =============================================================================
# FX Service
# =============================================================================

class FXService:
    """
    Service for fetching live exchange rates.
    Uses yfinance with fallback to default values.
    """

    DEFAULT_USDCNY = 7

    @staticmethod
    def get_usdcny_rate() -> Tuple[float, bool]:
        """
        Fetch live USD/CNY exchange rate.

        Returns:
            Tuple of (rate, is_live) where is_live indicates if rate was fetched successfully.
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker("CNY=X")
            # Get the most recent price
            hist = ticker.history(period="1d")

            if not hist.empty and "Close" in hist.columns:
                rate = float(hist["Close"].iloc[-1])
                if 5.0 < rate < 10.0:  # Sanity check for reasonable FX rate
                    return rate, True

            # Try alternative method - fast_info
            if hasattr(ticker, 'fast_info'):
                rate = ticker.fast_info.get('lastPrice', None)
                if rate and 5.0 < rate < 10.0:
                    return rate, True

        except Exception as e:
            st.warning(f"获取实时汇率失败: {str(e)[:50]}... 使用默认值")

        return FXService.DEFAULT_USDCNY, False


# =============================================================================
# Data Ingestion Class
# =============================================================================

class DataIngestion:
    """
    Responsible for loading, transforming, and cleaning price data.

    Handles:
    - Reading Excel/CSV files
    - Pivoting from long to wide format
    - Forward-filling missing data
    - Extracting asset metadata
    """

    # Column name mapping (Chinese -> English)
    COLUMN_MAP = {
        "合约细则ID": "ProductID",
        "合约细则描述": "ProductName",
        "报价日期": "Date",
        "结算价": "Settlement",
        "现货价格": "Spot",
        "报价单位": "Unit",
        "报价货币": "Currency"
    }

    def __init__(self, file_path: str):
        """
        Initialize the DataIngestion class.

        Args:
            file_path: Path to the Excel/CSV file containing price data.
        """
        self.file_path = file_path
        self._raw_data: Optional[pd.DataFrame] = None
        self._price_matrix: Optional[pd.DataFrame] = None
        self._asset_metadata: Optional[List[AssetMetadata]] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the raw data file.

        Returns:
            DataFrame with standardized column names.

        Raises:
            ValueError: If file format is not supported or required columns are missing.
        """
        if self._raw_data is not None:
            return self._raw_data

        try:
            if self.file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(self.file_path)
            elif self.file_path.lower().endswith('.csv'):
                df = pd.read_csv(self.file_path)
            else:
                raise ValueError(f"Unsupported file format: {self.file_path}")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

        # Rename columns to English
        df.columns = [self.COLUMN_MAP.get(col, col) for col in df.columns]

        # Validate required columns
        required_cols = ["ProductID", "Date", "Settlement", "Spot", "Unit", "Currency"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure Date is datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Sort by date
        df = df.sort_values("Date")

        self._raw_data = df
        return df

    def get_asset_metadata(self) -> List[AssetMetadata]:
        """
        Extract metadata for all available assets.

        新逻辑（方案2）：
        - 不再假设每个 ProductID 都有 Settlement 和 Spot
        - 根据实际数据判断：期货ID (如 "SG180 Apr26") 只有 Settlement
        - 现货ID (如 "SG180") 只有 Spot

        Returns:
            List of AssetMetadata objects for each product/price-type combination.
        """
        if self._asset_metadata is not None:
            return self._asset_metadata

        df = self.load_data()

        # Get unique products with their metadata
        products = df.groupby("ProductID").agg({
            "ProductName": "first",
            "Unit": "first",
            "Currency": "first"
        }).reset_index()

        metadata_list = []

        for _, row in products.iterrows():
            product_id = row['ProductID']

            # 获取该 ProductID 的所有数据
            product_data = df[df["ProductID"] == product_id]

            # 检查是否有有效的 Settlement 数据（非全 NaN）
            has_settlement = product_data["Settlement"].notna().any()

            # 检查是否有有效的 Spot 数据（非全 NaN）
            has_spot = product_data["Spot"].notna().any()

            if has_settlement:
                # Create Settlement asset
                metadata_list.append(AssetMetadata(
                    asset_id=f"{product_id}_Settlement",
                    product_id=product_id,
                    product_name=row["ProductName"],
                    price_type="Settlement",
                    unit=row["Unit"],
                    currency=row["Currency"]
                ))

            if has_spot:
                # Create Spot asset
                metadata_list.append(AssetMetadata(
                    asset_id=f"{product_id}_Spot",
                    product_id=product_id,
                    product_name=row["ProductName"],
                    price_type="Spot",
                    unit=row["Unit"],
                    currency=row["Currency"]
                ))

        # Sort by product ID for consistent ordering
        metadata_list.sort(key=lambda x: (x.product_id, x.price_type))

        self._asset_metadata = metadata_list
        return metadata_list

    def get_price_matrix(self, lookback_days: int = 250) -> pd.DataFrame:
        """
        Generate a wide-format price matrix with forward-filled missing values.

        新逻辑（方案2）：
        - 期货ID (如 "SG180 Apr26") 只有 Settlement 列
        - 现货ID (如 "SG180") 只有 Spot 列
        - 删除全是 NaN 的列

        Args:
            lookback_days: Number of most recent trading days to include.

        Returns:
            DataFrame with dates as index and asset prices as columns.
            Column names follow pattern: "{ProductID}_{Settlement|Spot}"
        """
        if self._price_matrix is not None:
            # Check if we have enough rows for the lookback
            if len(self._price_matrix) >= lookback_days:
                return self._price_matrix.tail(lookback_days)
            return self._price_matrix

        df = self.load_data()

        # Pivot Settlement prices
        settlement_pivot = df.pivot_table(
            index="Date",
            columns="ProductID",
            values="Settlement",
            aggfunc="first"
        )
        settlement_pivot.columns = [f"{col}_Settlement" for col in settlement_pivot.columns]

        # Pivot Spot prices
        spot_pivot = df.pivot_table(
            index="Date",
            columns="ProductID",
            values="Spot",
            aggfunc="first"
        )
        spot_pivot.columns = [f"{col}_Spot" for col in spot_pivot.columns]

        # Combine both price types
        price_matrix = pd.concat([settlement_pivot, spot_pivot], axis=1)

        # Handle missing data:
        # 1. Replace 0 values with NaN (0 is likely missing data, not actual zero price)
        price_matrix = price_matrix.replace(0, np.nan)

        # 2. 删除全是 NaN 的列（新数据结构下，期货ID没有Spot，现货ID没有Settlement）
        price_matrix = price_matrix.dropna(axis=1, how='all')

        # 3. Forward fill (ffill) to handle non-trading days
        price_matrix = price_matrix.ffill()

        # 4. Backward fill for any remaining NaN at the start
        price_matrix = price_matrix.bfill()

        # Sort columns for consistent ordering
        price_matrix = price_matrix.reindex(sorted(price_matrix.columns), axis=1)

        self._price_matrix = price_matrix

        # Return latest N days
        return price_matrix.tail(lookback_days)

    def get_returns(self, lookback_days: int = 250) -> pd.DataFrame:
        """
        Calculate log returns from the price matrix.

        IMPORTANT: To preserve correlation structure, we calculate returns
        BEFORE forward-filling. Days with missing data (zeros) result in
        NaN returns which are excluded from covariance calculation.

        This prevents artificial zero returns from destroying correlations
        between related assets (e.g., Settlement vs Spot for same product).

        新逻辑（方案2）：
        - 删除全是 NaN 的列（期货ID没有Spot，现货ID没有Settlement）

        Args:
            lookback_days: Number of trading days for the return calculation window.

        Returns:
            DataFrame of log returns with NaN for missing data days.
        """
        df = self.load_data()

        # Pivot Settlement prices (keep 0 as NaN, do NOT ffill yet)
        settlement_pivot = df.pivot_table(
            index="Date",
            columns="ProductID",
            values="Settlement",
            aggfunc="first"
        )
        settlement_pivot.columns = [f"{col}_Settlement" for col in settlement_pivot.columns]

        # Pivot Spot prices
        spot_pivot = df.pivot_table(
            index="Date",
            columns="ProductID",
            values="Spot",
            aggfunc="first"
        )
        spot_pivot.columns = [f"{col}_Spot" for col in spot_pivot.columns]

        # Combine both price types
        price_matrix_raw = pd.concat([settlement_pivot, spot_pivot], axis=1)

        # Replace 0 with NaN (0 means missing data, not actual zero price)
        price_matrix_raw = price_matrix_raw.replace(0, np.nan)

        # 删除全是 NaN 的列（新数据结构下，期货ID没有Spot，现货ID没有Settlement）
        price_matrix_raw = price_matrix_raw.dropna(axis=1, how='all')

        # Sort columns for consistent ordering
        price_matrix_raw = price_matrix_raw.reindex(sorted(price_matrix_raw.columns), axis=1)

        # Calculate log returns BEFORE ffill
        # This way, if either P_t or P_{t-1} is NaN, the return is NaN
        returns = np.log(price_matrix_raw / price_matrix_raw.shift(1))

        # Get the latest N days
        returns = returns.tail(lookback_days + 1)

        # Drop the first row (NaN from shift)
        returns = returns.iloc[1:]

        return returns

    def get_latest_prices(self) -> pd.Series:
        """
        Get the most recent price for each asset.

        Returns:
            Series with asset_id as index and latest price as value.
        """
        price_matrix = self.get_price_matrix()
        # Get the last row (most recent date)
        latest_prices = price_matrix.iloc[-1]
        return latest_prices

    def get_latest_price_date(self) -> pd.Timestamp:
        """
        Get the date of the most recent price data.

        Returns:
            Timestamp of the latest price date.
        """
        price_matrix = self.get_price_matrix()
        return price_matrix.index[-1]


# =============================================================================
# Risk Engine Class
# =============================================================================

class RiskEngine:
    """
    Risk calculation engine implementing Parametric VaR with EWMA volatility.

    Key features:
    - EWMA covariance matrix with configurable decay factor
    - Multi-asset portfolio VaR calculation
    - Support for multiple confidence levels
    - Square-root-of-time scaling for multi-day VaR
    """

    # Z-scores for standard confidence levels
    Z_SCORES = {
        0.95: 1.6449,   # 95% confidence (one-tailed)
        0.99: 2.3263    # 99% confidence (one-tailed)
    }

    def __init__(self, returns: pd.DataFrame, decay_factor: float = 0.94):
        """
        Initialize the RiskEngine.

        Args:
            returns: DataFrame of asset returns (dates as index, assets as columns).
            decay_factor: Lambda for EWMA calculation (default 0.94, industry standard).
        """
        if not 0 < decay_factor < 1:
            raise ValueError("Decay factor must be between 0 and 1")

        self.returns = returns
        self.decay_factor = decay_factor
        self._ewma_cov: Optional[np.ndarray] = None
        self.asset_names = list(returns.columns)

    def calculate_ewma_covariance(self) -> np.ndarray:
        """
        Calculate the EWMA covariance matrix with proper handling of missing data.

        Uses PAIRWISE DELETION: For each pair of assets (i, j), only uses days
        where BOTH assets have valid (non-NaN) returns. This preserves the
        correlation structure even when some assets have missing data.

        The EWMA covariance gives more weight to recent observations:
        σ_ij(t) = λ * σ_ij(t-1) + (1-λ) * r_i(t) * r_j(t)

        Returns:
            Covariance matrix as numpy array.
        """
        if self._ewma_cov is not None:
            return self._ewma_cov

        returns_df = self.returns
        T, n_assets = returns_df.shape

        if T < 2:
            raise ValueError("Need at least 2 observations for covariance calculation")

        lambda_ = self.decay_factor

        # Generate exponential weights (most recent = highest weight)
        base_weights = np.array([(1 - lambda_) * (lambda_ ** i) for i in range(T - 1, -1, -1)])

        # Initialize covariance matrix
        cov_matrix = np.zeros((n_assets, n_assets))

        # Calculate EWMA covariance for each pair using pairwise deletion
        for i in range(n_assets):
            for j in range(i, n_assets):  # Only upper triangle (symmetric)
                # Get returns for assets i and j
                r_i = returns_df.iloc[:, i].values
                r_j = returns_df.iloc[:, j].values

                # Find days where BOTH have valid returns
                valid_mask = ~(np.isnan(r_i) | np.isnan(r_j))
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) < 10:
                    # Not enough data - use a high variance as fallback
                    cov_matrix[i, j] = 0.001 if i == j else 0
                    cov_matrix[j, i] = cov_matrix[i, j]
                    continue

                # Extract valid returns
                r_i_valid = r_i[valid_mask]
                r_j_valid = r_j[valid_mask]

                # Get weights for valid days and renormalize
                weights_valid = base_weights[valid_mask]
                weights_valid = weights_valid / weights_valid.sum()

                # Demean returns (using weighted mean)
                mean_i = np.sum(weights_valid * r_i_valid)
                mean_j = np.sum(weights_valid * r_j_valid)
                r_i_centered = r_i_valid - mean_i
                r_j_centered = r_j_valid - mean_j

                # Calculate weighted covariance
                cov_ij = np.sum(weights_valid * r_i_centered * r_j_centered)

                cov_matrix[i, j] = cov_ij
                cov_matrix[j, i] = cov_ij  # Symmetric

        self._ewma_cov = cov_matrix
        return cov_matrix

    def calculate_portfolio_var(
        self,
        positions: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate portfolio VaR for a given position vector.

        VaR = sqrt(V^T · Σ · V) · Z_α

        Args:
            positions: Position vector (signed notional amounts, positive=long, negative=short).
            confidence: Confidence level (0.95 or 0.99).

        Returns:
            VaR value in the same currency as positions.
        """
        if len(positions) != len(self.asset_names):
            raise ValueError(
                f"Position vector length ({len(positions)}) must match "
                f"number of assets ({len(self.asset_names)})"
            )

        cov_matrix = self.calculate_ewma_covariance()

        # Portfolio variance: V^T · Σ · V
        positions = np.array(positions).flatten()  # Ensure 1D array
        portfolio_variance = positions @ cov_matrix @ positions  # Result is scalar

        # Portfolio standard deviation
        portfolio_std = np.sqrt(portfolio_variance)

        # Get Z-score for confidence level
        z_score = self.Z_SCORES.get(confidence)
        if z_score is None:
            z_score = norm.ppf(confidence)

        # VaR = σ_portfolio * Z_α
        var_1day = portfolio_std * z_score

        return var_1day

    def calculate_portfolio_cvar(
        self,
        positions: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate portfolio CVaR (Conditional VaR / Expected Shortfall).

        Uses parametric method: CVaR = σ_p × φ(z_α) / (1 - α)
        where φ is the standard normal PDF and z_α is the quantile.

        Args:
            positions: Position vector (signed notional amounts).
            confidence: Confidence level (0.95 or 0.99).

        Returns:
            CVaR value in the same currency as positions.
        """
        cov_matrix = self.calculate_ewma_covariance()

        positions = np.array(positions).flatten()
        portfolio_variance = positions @ cov_matrix @ positions
        portfolio_std = np.sqrt(portfolio_variance)

        z_score = self.Z_SCORES.get(confidence)
        if z_score is None:
            z_score = norm.ppf(confidence)

        # CVaR = σ × φ(z_α) / (1 - α)
        cvar_1day = portfolio_std * norm.pdf(z_score) / (1 - confidence)

        return cvar_1day

    def get_var_results(
        self,
        positions: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive VaR results for multiple confidence levels and horizons.

        Args:
            positions: Position vector (signed notional amounts).

        Returns:
            Dictionary with VaR values:
            {
                "1-Day": {"95%": value, "99%": value},
                "10-Day": {"95%": value, "99%": value}
            }
        """
        results = {}

        # Calculate 1-Day VaR
        var_1d_95 = self.calculate_portfolio_var(positions, 0.95)
        var_1d_99 = self.calculate_portfolio_var(positions, 0.99)

        results["1-Day"] = {
            "95%": var_1d_95,
            "99%": var_1d_99
        }

        # Calculate 10-Day VaR using square root of time rule
        sqrt_10 = np.sqrt(10)
        results["10-Day"] = {
            "95%": var_1d_95 * sqrt_10,
            "99%": var_1d_99 * sqrt_10
        }

        return results

    def get_var_cvar_results(
        self,
        positions: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive VaR and CVaR results.

        Returns:
            Dictionary with VaR and CVaR values for 1-Day and 10-Day horizons.
        """
        var_1d_95 = self.calculate_portfolio_var(positions, 0.95)
        var_1d_99 = self.calculate_portfolio_var(positions, 0.99)
        cvar_1d_95 = self.calculate_portfolio_cvar(positions, 0.95)
        cvar_1d_99 = self.calculate_portfolio_cvar(positions, 0.99)

        sqrt_10 = np.sqrt(10)

        return {
            "1-Day": {
                "VaR_95%": var_1d_95,
                "VaR_99%": var_1d_99,
                "CVaR_95%": cvar_1d_95,
                "CVaR_99%": cvar_1d_99,
            },
            "10-Day": {
                "VaR_95%": var_1d_95 * sqrt_10,
                "VaR_99%": var_1d_99 * sqrt_10,
                "CVaR_95%": cvar_1d_95 * sqrt_10,
                "CVaR_99%": cvar_1d_99 * sqrt_10,
            },
        }

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get the correlation matrix derived from EWMA covariance.

        Returns:
            DataFrame with asset correlations.
        """
        cov_matrix = self.calculate_ewma_covariance()

        # Convert covariance to correlation
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        return pd.DataFrame(
            corr_matrix,
            index=self.asset_names,
            columns=self.asset_names
        )

    def get_individual_volatilities(self) -> pd.Series:
        """
        Get annualized volatilities for each asset.

        Returns:
            Series of annualized volatilities.
        """
        cov_matrix = self.calculate_ewma_covariance()
        daily_vol = np.sqrt(np.diag(cov_matrix))
        annual_vol = daily_vol * np.sqrt(252)  # Annualize

        return pd.Series(annual_vol, index=self.asset_names)


# =============================================================================
# Streamlit UI
# =============================================================================

def format_currency(value: float, currency: str = "CNY") -> str:
    """Format a number as currency string."""
    if abs(value) >= 1_000_000:
        return f"{value:,.0f} {currency}"
    elif abs(value) >= 1_000:
        return f"{value:,.2f} {currency}"
    else:
        return f"{value:.4f} {currency}"


# =============================================================================
# Exposure Table Parsing
# =============================================================================

# Column header mapping (Chinese -> internal)
EXPOSURE_HEADER_MAP = {
    "品种": "ProductCode",
    "合约月份": "ContractMonth",
    "现货持仓": "SpotPosition",
    "期货持仓": "FuturesPosition",
    "单位": "Unit",
}


def _extract_base_products(asset_columns: List[str]) -> set:
    """
    Extract base product codes from asset column names.

    Handles both formats:
    - "CUFI_Settlement" -> "CUFI"
    - "SG380 Apr26_Spot" -> "SG380"
    - "Brt Fut Jun26_Settlement" -> "Brt Fut"
    - "GO 10ppm Apr26_Spot" -> "GO 10ppm"
    """
    import re
    month_pattern = re.compile(
        r"\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2}$"
    )
    products = set()
    for col in asset_columns:
        # Remove suffix (_Settlement or _Spot)
        base = col.rsplit("_", 1)[0]
        # Also add the full base (with month) for exact matching
        products.add(base)
        # Strip trailing contract month (e.g., "Brt Fut Jun26" -> "Brt Fut")
        product_name = month_pattern.sub("", base).strip()
        products.add(product_name)
    return products


def parse_exposure_table(
    uploaded_file,
    valid_products: Optional[set] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse an exposure table Excel file (敞口表).

    Expected columns: 品种 | 合约月份 | 现货持仓 | 期货持仓 | 单位

    Args:
        uploaded_file: Streamlit UploadedFile object or file-like.
        valid_products: Optional set of valid product codes from loaded price data.
                        If None, all products are accepted (validation deferred to mapping).

    Returns:
        Tuple of (parsed DataFrame, list of warning messages).
    """
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        raise ValueError(f"无法读取Excel文件: {str(e)}")

    # Rename columns using header map
    df.columns = [EXPOSURE_HEADER_MAP.get(col.strip(), col.strip()) for col in df.columns]

    # Validate required column exists
    if "ProductCode" not in df.columns:
        raise ValueError(
            "未找到必需的列 '品种'。\n"
            "期望的列名: 品种, 合约月份, 现货持仓, 期货持仓, 单位"
        )

    warnings_list: List[str] = []
    rows: List[Dict] = []

    for idx, row in df.iterrows():
        row_num = idx + 2  # Excel row number (1-indexed header + data)

        # Get product code
        product = str(row.get("ProductCode", "")).strip()
        if not product or product in ("nan", "None", ""):
            continue

        # Validate product against loaded data (if provided)
        if valid_products is not None and product not in valid_products:
            warnings_list.append(
                f"第{row_num}行: 品种 '{product}' 在当前价格数据中不存在，已跳过。"
            )
            continue

        # Parse contract month
        contract_month = str(row.get("ContractMonth", "")).strip()
        if contract_month in ("", "nan", "None", "NaT"):
            contract_month = None

        # Parse positions (handle NaN, empty, string values)
        def safe_float(val) -> float:
            if pd.isna(val):
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        spot = safe_float(row.get("SpotPosition", 0))
        futures = safe_float(row.get("FuturesPosition", 0))

        # Skip rows with both positions zero
        if spot == 0 and futures == 0:
            warnings_list.append(
                f"第{row_num}行: '{product}' 现货和期货持仓均为零，已跳过。"
            )
            continue

        # Parse unit (accept any unit — different datasets use different units)
        unit = str(row.get("Unit", "")).strip().upper()
        if unit in ("NAN", "NONE", ""):
            unit = "-"
        elif unit == "BBLS":
            unit = "BBL"

        # Warn if futures position has no contract month
        if futures != 0 and contract_month is None:
            warnings_list.append(
                f"第{row_num}行: '{product}' 期货持仓无合约月份，将使用通用期货价格。"
            )

        rows.append({
            "ProductCode": product,
            "ContractMonth": contract_month,
            "SpotPosition": spot,
            "FuturesPosition": futures,
            "Unit": unit,
        })

    if not rows:
        raise ValueError("未找到有效的敞口数据。请检查文件格式和品种名称。")

    return pd.DataFrame(rows), warnings_list


def _find_asset_id(
    product: str,
    contract_month: Optional[str],
    price_type: str,
    asset_columns: List[str],
) -> Optional[str]:
    """
    Find the matching asset_id in the price matrix columns.

    Data has product IDs like "SG380 Apr26" with both Settlement and Spot.
    Mapping rules:
    - With contract month: "{product} {month}_{price_type}"
    - Without contract month (spot): find the front month for that product
    """
    if contract_month:
        # Direct match with contract month
        asset_id = f"{product} {contract_month}_{price_type}"
        if asset_id in asset_columns:
            return asset_id

    # Also try without contract month (in case data has bare product IDs)
    bare_id = f"{product}_{price_type}"
    if bare_id in asset_columns:
        return bare_id

    # If no contract month, find the front (first available) month for this product
    if not contract_month:
        candidates = [
            col for col in asset_columns
            if col.startswith(f"{product} ") and col.endswith(f"_{price_type}")
        ]
        if candidates:
            # Sort to get the front month (alphabetical gives chronological for MonYY format)
            candidates.sort()
            return candidates[0]

    return None


def build_exposure_position_vector(
    exposure_df: pd.DataFrame,
    returns: pd.DataFrame,
    latest_prices: pd.Series,
    asset_metadata: List[AssetMetadata],
    fx_rate: float,
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """
    Build the position vector from parsed exposure data.

    Data structure: ProductIDs are like "SG380 Apr26" with both Settlement and Spot.
    Mapping:
    - Spot position for "SG380" (no month) -> find front month's Spot price
    - Spot position for "SG380" month "Apr26" -> "SG380 Apr26_Spot"
    - Futures position for "SG380" month "Apr26" -> "SG380 Apr26_Settlement"

    Args:
        exposure_df: Parsed exposure DataFrame.
        returns: Returns DataFrame (columns = asset IDs).
        latest_prices: Latest prices Series.
        asset_metadata: List of AssetMetadata objects.
        fx_rate: USD/CNY exchange rate.

    Returns:
        Tuple of (position_vector, active_positions_list, warnings).
    """
    asset_columns = list(returns.columns)
    position_vector = np.zeros(len(asset_columns))
    active_positions = []
    warnings_list: List[str] = []

    asset_index_map = {col: i for i, col in enumerate(asset_columns)}
    asset_meta_map = {a.asset_id: a for a in asset_metadata}

    for _, row in exposure_df.iterrows():
        product = row["ProductCode"]
        contract_month = row["ContractMonth"]
        spot_qty = row["SpotPosition"]
        futures_qty = row["FuturesPosition"]
        unit = row["Unit"]

        # --- Handle Spot Position ---
        if spot_qty != 0:
            spot_asset_id = _find_asset_id(product, contract_month, "Spot", asset_columns)

            if spot_asset_id and spot_asset_id in asset_index_map:
                idx = asset_index_map[spot_asset_id]
                price = latest_prices.get(spot_asset_id, 0)

                if price > 0:
                    notional = spot_qty * price

                    asset_meta = asset_meta_map.get(spot_asset_id)
                    currency = asset_meta.currency if asset_meta else "USD"
                    if currency == "USD":
                        notional *= fx_rate

                    position_vector[idx] += notional

                    active_positions.append({
                        "ProductCode": product,
                        "PositionType": "现货",
                        "ContractMonth": contract_month or "(前月)",
                        "Quantity": spot_qty,
                        "Unit": unit,
                        "Price": price,
                        "Currency": currency,
                        "Notional_CNY": abs(notional),
                        "Direction": "多头" if spot_qty > 0 else "空头",
                        "AssetID": spot_asset_id,
                    })
                else:
                    warnings_list.append(f"'{product}' 现货价格为零或不可用，已跳过现货持仓。")
            else:
                warnings_list.append(
                    f"'{product}' 无现货价格数据，已跳过现货持仓。"
                    f"（可用资产: 需要 '{product} <月份>_Spot'）"
                )

        # --- Handle Futures Position ---
        if futures_qty != 0:
            futures_asset_id = _find_asset_id(product, contract_month, "Settlement", asset_columns)

            if futures_asset_id and futures_asset_id in asset_index_map:
                idx = asset_index_map[futures_asset_id]
                price = latest_prices.get(futures_asset_id, 0)

                if price > 0:
                    notional = futures_qty * price

                    asset_meta = asset_meta_map.get(futures_asset_id)
                    currency = asset_meta.currency if asset_meta else "USD"
                    if currency == "USD":
                        notional *= fx_rate

                    position_vector[idx] += notional

                    active_positions.append({
                        "ProductCode": product,
                        "PositionType": "期货",
                        "ContractMonth": contract_month or "(前月)",
                        "Quantity": futures_qty,
                        "Unit": unit,
                        "Price": price,
                        "Currency": currency,
                        "Notional_CNY": abs(notional),
                        "Direction": "多头" if futures_qty > 0 else "空头",
                        "AssetID": futures_asset_id,
                    })
                else:
                    warnings_list.append(
                        f"'{product}' 期货 ({contract_month or '通用'}) 价格为零或不可用，已跳过。"
                    )
            else:
                warnings_list.append(
                    f"'{product}' 无期货价格数据 (合约月份: {contract_month or '未指定'})，已跳过期货持仓。"
                )

    return position_vector, active_positions, warnings_list


def create_position_input(
    asset: AssetMetadata,
    key_prefix: str
) -> Tuple[float, str]:
    """
    Create position input widgets for a single asset.

    Returns:
        Tuple of (position_size, direction)
    """
    price_type_label = "期货头寸" if asset.price_type == "Settlement" else "现货头寸"
    currency_flag = "🇺🇸" if asset.currency == "USD" else "🇨🇳"

    col1, col2 = st.columns([3, 1])

    with col1:
        position = st.number_input(
            f"{price_type_label} (单位: {asset.unit}) {currency_flag}",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
            key=f"{key_prefix}_{asset.asset_id}_pos",
            help=f"输入头寸数量，单位: {asset.unit}，货币: {asset.currency}"
        )

    with col2:
        direction = st.selectbox(
            "方向",
            options=["Long", "Short"],
            format_func=lambda x: "多头" if x == "Long" else "空头",
            key=f"{key_prefix}_{asset.asset_id}_dir",
            label_visibility="collapsed"
        )

    return position, direction


def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="ITG-X Risk Dashboard",
        page_icon="🛡️",
        layout="wide"
    )

    # Custom CSS for better styling (支持深色/浅色模式)
    st.markdown("""
    <style>
    /* 浅色模式 */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
    }

    /* 深色模式适配 */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #93C5FD !important;  /* 浅蓝色 */
        }
        .sub-header {
            color: #9CA3AF !important;  /* 浅灰色 */
        }
        .stMetric {
            background-color: #1F2937 !important;
            border: 1px solid #374151 !important;
        }
    }

    /* Streamlit 深色主题适配 (data-theme 属性) */
    [data-theme="dark"] .main-header,
    [data-testid="stAppViewContainer"][data-theme="dark"] .main-header {
        color: #93C5FD !important;
    }
    [data-theme="dark"] .sub-header,
    [data-testid="stAppViewContainer"][data-theme="dark"] .sub-header {
        color: #9CA3AF !important;
    }
    [data-theme="dark"] .stMetric,
    [data-testid="stAppViewContainer"][data-theme="dark"] .stMetric {
        background-color: #1F2937 !important;
        border: 1px solid #374151 !important;
    }

    /* 基于 Streamlit 的 CSS 变量 (最可靠的方法) */
    :root {
        --header-color-light: #1E3A5F;
        --header-color-dark: #93C5FD;
        --subheader-color-light: #6B7280;
        --subheader-color-dark: #9CA3AF;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">🛡️ ITG-X Risk Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">集团VaR计算器 | Group Value at Risk Calculator</p>', unsafe_allow_html=True)
    st.markdown("""
    **参数法VaR** · EWMA协方差矩阵 (λ = 0.94) · 支持期货与现货基差风险计算
    """)

    st.divider()

    # =========================================================================
    # Settings Sidebar
    # =========================================================================

    with st.sidebar:
        st.header("⚙️ 参数设置")

        # FX Rate Section
        st.subheader("汇率 (USD → CNY)")

        # Fetch live rate
        if "fx_rate" not in st.session_state:
            rate, is_live = FXService.get_usdcny_rate()
            st.session_state.fx_rate = rate
            st.session_state.fx_is_live = is_live

        fx_status = "🟢 实时" if st.session_state.fx_is_live else "🟡 默认"
        st.caption(f"状态: {fx_status}")

        fx_rate = st.number_input(
            "USD/CNY 汇率",
            min_value=1.0,
            max_value=20.0,
            value=st.session_state.fx_rate,
            step=0.01,
            format="%.4f",
            help="实时汇率来自Yahoo Finance，可手动修改"
        )

        if st.button("🔄 刷新汇率"):
            rate, is_live = FXService.get_usdcny_rate()
            st.session_state.fx_rate = rate
            st.session_state.fx_is_live = is_live
            st.rerun()

        st.divider()

        # Model Parameters
        st.subheader("模型参数")

        lookback = st.slider(
            "回看周期 (交易日)",
            min_value=60,
            max_value=500,
            value=250,
            step=10,
            help="用于VaR计算的历史交易日天数"
        )

        decay_factor = st.slider(
            "EWMA衰减因子 (λ)",
            min_value=0.90,
            max_value=0.99,
            value=0.94,
            step=0.01,
            help="λ越大，历史数据权重越高。行业标准: 0.94"
        )

        st.divider()

        st.subheader("📁 数据文件")
        uploaded_file = st.file_uploader(
            "上传价格数据 (可选)",
            type=["xlsx", "xls", "csv"],
            help="留空则使用默认文件: group VaR model.XLSX"
        )

    # =========================================================================
    # Load Data
    # =========================================================================

    # Determine file path - with session state caching
    import tempfile
    import os

    if uploaded_file is not None:
        # Cache uploaded file in session state to persist across interactions
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        # Check if this is a new file or same as cached
        if ("uploaded_file_bytes" not in st.session_state or
            st.session_state.uploaded_file_bytes != file_bytes):
            st.session_state.uploaded_file_bytes = file_bytes
            st.session_state.uploaded_file_name = file_name

            # Save to temp file
            suffix = os.path.splitext(file_name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                st.session_state.uploaded_file_path = tmp.name

        file_path = st.session_state.uploaded_file_path
        st.sidebar.success(f"✅ 已加载: {st.session_state.uploaded_file_name}")

    elif "uploaded_file_path" in st.session_state:
        # Use previously uploaded file from session state
        file_path = st.session_state.uploaded_file_path
        st.sidebar.success(f"✅ 已加载: {st.session_state.uploaded_file_name}")

    else:
        # Use default file
        file_path = os.path.join(os.path.dirname(__file__), "group VaR model.XLSX")

    # Initialize data ingestion
    try:
        data_ingestion = DataIngestion(file_path)
        asset_metadata = data_ingestion.get_asset_metadata()
        returns = data_ingestion.get_returns(lookback)
        latest_prices = data_ingestion.get_latest_prices()
        latest_price_date = data_ingestion.get_latest_price_date()
    except Exception as e:
        st.error(f"❌ 数据加载错误: {str(e)}")
        st.stop()

    st.info(f"📅 最新价格日期: **{latest_price_date.strftime('%Y-%m-%d')}**")

    # Get unique products for grouping
    products = {}
    for asset in asset_metadata:
        if asset.product_id not in products:
            products[asset.product_id] = {
                "name": asset.product_name,
                "unit": asset.unit,
                "currency": asset.currency,
                "assets": []
            }
        products[asset.product_id]["assets"].append(asset)

    # =========================================================================
    # Main Mode Selection: Manual Input vs Exposure Upload
    # =========================================================================

    main_tab_manual, main_tab_exposure = st.tabs([
        "📝 手动输入头寸",
        "📤 上传敞口表 (Exposure VaR)",
    ])

    # =========================================================================
    # TAB 1: Manual Position Input (original functionality)
    # =========================================================================
    with main_tab_manual:

        st.header("📝 输入头寸")

        # Create tabs for CNY and USD products
        cny_products = {k: v for k, v in products.items() if v["currency"] == "CNY"}
        usd_products = {k: v for k, v in products.items() if v["currency"] == "USD"}

        tab_cny, tab_usd = st.tabs([
            f"🇨🇳 人民币产品 ({len(cny_products)})",
            f"🇺🇸 美元产品 ({len(usd_products)})"
        ])

        positions_input = {}

        with tab_cny:
            st.info("💡 请输入头寸数量，单位如括号所示。多头(Long) = 做多敞口，空头(Short) = 做空敞口")

            col1, col2 = st.columns(2)
            product_list = list(cny_products.items())
            mid = len(product_list) // 2

            for idx, (product_id, product_info) in enumerate(product_list):
                target_col = col1 if idx < mid else col2
                with target_col:
                    with st.expander(f"**{product_id}** - {product_info['name']} [{product_info['unit']}]"):
                        for asset in product_info["assets"]:
                            pos, direction = create_position_input(asset, "cny")
                            positions_input[asset.asset_id] = {
                                "position": pos,
                                "direction": direction,
                                "asset": asset
                            }

        with tab_usd:
            st.warning(f"⚠️ 美元头寸将按汇率 **{fx_rate:.4f}** 转换为人民币")
            for product_id, product_info in usd_products.items():
                with st.expander(f"**{product_id}** - {product_info['name']} [{product_info['unit']}]"):
                    for asset in product_info["assets"]:
                        pos, direction = create_position_input(asset, "usd")
                        positions_input[asset.asset_id] = {
                            "position": pos,
                            "direction": direction,
                            "asset": asset
                        }

        st.divider()

        # Calculate VaR
        col_calc, col_clear = st.columns([1, 5])
        with col_calc:
            calculate_button = st.button("🧮 计算VaR", type="primary", width="stretch")
        with col_clear:
            if st.button("🗑️ 清空"):
                st.rerun()

        if calculate_button:
            position_vector = []
            active_positions = []

            for asset_id in returns.columns:
                if asset_id in positions_input:
                    info = positions_input[asset_id]
                    quantity = info["position"]
                    direction = info["direction"]
                    asset = info["asset"]

                    current_price = latest_prices.get(asset_id, 0)
                    notional = quantity * current_price
                    signed_notional = notional if direction == "Long" else -notional

                    if asset.currency == "USD":
                        signed_notional *= fx_rate

                    position_vector.append(signed_notional)

                    if quantity != 0:
                        active_positions.append({
                            "Asset": asset.display_name,
                            "ID": asset.asset_id,
                            "Quantity": quantity,
                            "Unit": asset.unit,
                            "Price": current_price,
                            "Direction": direction,
                            "Currency": asset.currency,
                            "Notional (CNY)": abs(signed_notional)
                        })
                else:
                    position_vector.append(0.0)

            position_vector = np.array(position_vector)

            if np.allclose(position_vector, 0):
                st.warning("⚠️ 请至少输入一个非零头寸")
            else:
                try:
                    engine = RiskEngine(returns, decay_factor=decay_factor)
                    results = engine.get_var_results(position_vector)
                except Exception as e:
                    st.error(f"❌ VaR计算错误: {str(e)}")
                    st.stop()

                st.header("📊 VaR计算结果")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("1日VaR (95%)", format_currency(results["1-Day"]["95%"]))
                with col2:
                    st.metric("1日VaR (99%)", format_currency(results["1-Day"]["99%"]))
                with col3:
                    st.metric("10日VaR (95%)", format_currency(results["10-Day"]["95%"]))
                with col4:
                    st.metric("10日VaR (99%)", format_currency(results["10-Day"]["99%"]))

                st.divider()

                st.subheader("📋 持仓汇总")
                if active_positions:
                    positions_df = pd.DataFrame(active_positions)
                    display_df = positions_df.copy()
                    display_df["Price"] = display_df["Price"].apply(lambda x: f"{x:,.2f}")
                    display_df["Notional (CNY)"] = display_df["Notional (CNY)"].apply(lambda x: f"{x:,.0f}")
                    display_df["Direction"] = display_df["Direction"].apply(lambda x: "多头" if x == "Long" else "空头")
                    display_df = display_df.rename(columns={
                        "Asset": "资产", "ID": "代码", "Quantity": "数量",
                        "Unit": "单位", "Price": "价格", "Direction": "方向",
                        "Currency": "货币", "Notional (CNY)": "名义金额(CNY)"
                    })
                    st.dataframe(display_df, width="stretch", hide_index=True)

                    total_long = sum(p["Notional (CNY)"] for p in active_positions if p["Direction"] == "Long")
                    total_short = sum(p["Notional (CNY)"] for p in active_positions if p["Direction"] == "Short")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("多头敞口", format_currency(total_long))
                    with col2:
                        st.metric("空头敞口", format_currency(total_short))
                    with col3:
                        st.metric("净敞口", format_currency(total_long - total_short))

                st.divider()

                with st.expander("📈 波动率与相关性分析"):
                    st.subheader("年化波动率 (活跃资产)")
                    vol_series = engine.get_individual_volatilities()
                    active_asset_ids = [p["ID"] for p in active_positions]
                    active_vols = vol_series[vol_series.index.isin(active_asset_ids)]

                    if not active_vols.empty:
                        vol_df = pd.DataFrame({
                            "资产": active_vols.index,
                            "年化波动率": [f"{v*100:.2f}%" for v in active_vols.values]
                        })
                        st.dataframe(vol_df, width="stretch", hide_index=True)

                    st.subheader("相关系数矩阵 (活跃资产)")
                    corr_matrix = engine.get_correlation_matrix()
                    if len(active_asset_ids) > 1:
                        active_corr = corr_matrix.loc[
                            corr_matrix.index.isin(active_asset_ids),
                            corr_matrix.columns.isin(active_asset_ids)
                        ]
                        st.dataframe(
                            active_corr.style.format("{:.2%}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                            width="stretch"
                        )
                    else:
                        st.info("输入2个以上资产头寸可查看相关系数矩阵")

    # =========================================================================
    # TAB 2: Exposure Table Upload (敞口表上传)
    # =========================================================================
    with main_tab_exposure:

        st.header("📤 敞口表上传 (Exposure VaR)")
        st.markdown(
            "上传包含现货/期货持仓的Excel敞口表，系统将自动解析并计算 **VaR + CVaR**。"
        )

        # Show available products from loaded data
        available_products_set = _extract_base_products(list(returns.columns))
        # Filter to base product codes only (not "SG380 Apr26", just "SG380")
        base_products = sorted({
            col.rsplit("_", 1)[0].split(" ")[0]
            for col in returns.columns
        })

        # Format info
        with st.expander("📋 敞口表格式说明", expanded=False):
            st.markdown(f"""
            **Excel列名（中文表头）：**

            | 品种 | 合约月份 | 现货持仓 | 期货持仓 | 单位 |
            |------|----------|----------|----------|------|
            | {base_products[0] if base_products else 'XXX'} | | 5000 | | - |
            | {base_products[1] if len(base_products) > 1 else 'YYY'} | Apr26 | 3000 | -2000 | - |

            **规则：**
            - **品种**: 必须在已加载的价格数据中存在。当前可用品种 ({len(base_products)}个):
              `{', '.join(base_products[:15])}{'...' if len(base_products) > 15 else ''}`
            - **合约月份**: 期货持仓时填写（如 Apr26, May26），现货可留空
            - **正数 = 多头，负数 = 空头**，空白 = 0
            - **单位**: 自由填写（如 MT, BBL, TON, KG 等），留空则显示 "-"
            - 每行至少一个持仓列非零
            """)

        # File uploader
        exposure_file = st.file_uploader(
            "上传敞口表 (Excel)",
            type=["xlsx", "xls"],
            key="exposure_uploader",
            help="拖拽或点击上传 .xlsx / .xls 文件，最大 10MB",
        )

        if exposure_file is not None:
            # Validate file size
            if exposure_file.size > 10 * 1024 * 1024:
                st.error("❌ 文件大小超过10MB限制")
            else:
                st.info(f"📄 已选择: **{exposure_file.name}** ({exposure_file.size / 1024:.1f} KB)")

                # Parse the exposure table
                try:
                    # Extract valid product codes dynamically from loaded price data
                    available_products = _extract_base_products(list(returns.columns))
                    exposure_df, parse_warnings = parse_exposure_table(
                        exposure_file, valid_products=available_products
                    )
                except ValueError as e:
                    st.error(f"❌ 解析错误: {str(e)}")
                    exposure_df = None
                    parse_warnings = []

                if exposure_df is not None and len(exposure_df) > 0:
                    # Show parsed data
                    st.subheader("📄 解析结果 (敞口明细)")

                    display_exposure = exposure_df.copy()
                    display_exposure = display_exposure.rename(columns={
                        "ProductCode": "品种",
                        "ContractMonth": "合约月份",
                        "SpotPosition": "现货持仓",
                        "FuturesPosition": "期货持仓",
                        "Unit": "单位",
                    })
                    # Replace None with "-"
                    display_exposure["合约月份"] = display_exposure["合约月份"].fillna("-")

                    st.dataframe(display_exposure, width="stretch", hide_index=True)

                    # Show parse warnings
                    if parse_warnings:
                        with st.expander(f"⚠️ 解析警告 ({len(parse_warnings)})", expanded=True):
                            for w in parse_warnings:
                                st.warning(w)

                    st.divider()

                    # Build position vector and calculate VaR/CVaR
                    position_vector, active_positions, build_warnings = build_exposure_position_vector(
                        exposure_df, returns, latest_prices, asset_metadata, fx_rate
                    )

                    # Show build warnings
                    all_warnings = parse_warnings + build_warnings
                    if build_warnings:
                        with st.expander(f"⚠️ 持仓映射警告 ({len(build_warnings)})", expanded=True):
                            for w in build_warnings:
                                st.warning(w)

                    if np.allclose(position_vector, 0):
                        st.error("❌ 所有敞口均无法匹配到价格数据，无法计算VaR。请检查品种名称和合约月份。")
                    else:
                        # Calculate VaR + CVaR
                        try:
                            engine = RiskEngine(returns, decay_factor=decay_factor)
                            var_cvar_results = engine.get_var_cvar_results(position_vector)
                        except Exception as e:
                            st.error(f"❌ VaR/CVaR计算错误: {str(e)}")
                            st.stop()

                        # =============================================
                        # Display VaR/CVaR Results
                        # =============================================

                        st.header("📊 组合风险汇总 (Portfolio Risk Summary)")

                        # Summary chips
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("解析行数", f"{len(exposure_df)} 行")
                        with col_s2:
                            st.metric("有效持仓", f"{len(active_positions)} 个")
                        with col_s3:
                            total_value = sum(p["Notional_CNY"] for p in active_positions)
                            st.metric("总持仓价值", format_currency(total_value))

                        st.divider()

                        # VaR / CVaR metrics
                        st.subheader("VaR & CVaR")

                        res_1d = var_cvar_results["1-Day"]
                        res_10d = var_cvar_results["10-Day"]

                        # Row 1: 1-Day metrics
                        st.markdown("**1日 (1-Day)**")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("VaR 95%", format_currency(res_1d["VaR_95%"]),
                                      help="95%置信度下1日最大预期损失")
                        with c2:
                            st.metric("VaR 99%", format_currency(res_1d["VaR_99%"]),
                                      help="99%置信度下1日最大预期损失")
                        with c3:
                            st.metric("CVaR 95%", format_currency(res_1d["CVaR_95%"]),
                                      help="95%置信度下1日条件期望损失")
                        with c4:
                            st.metric("CVaR 99%", format_currency(res_1d["CVaR_99%"]),
                                      help="99%置信度下1日条件期望损失")

                        # Row 2: 10-Day metrics
                        st.markdown("**10日 (10-Day, √10缩放)**")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("VaR 95%", format_currency(res_10d["VaR_95%"]))
                        with c2:
                            st.metric("VaR 99%", format_currency(res_10d["VaR_99%"]))
                        with c3:
                            st.metric("CVaR 95%", format_currency(res_10d["CVaR_95%"]))
                        with c4:
                            st.metric("CVaR 99%", format_currency(res_10d["CVaR_99%"]))

                        # Summary table
                        st.divider()
                        st.subheader("📋 VaR/CVaR 汇总表")
                        summary_table = pd.DataFrame({
                            "指标": ["VaR (1日)", "CVaR (1日)", "VaR (10日)", "CVaR (10日)"],
                            "95% 置信度": [
                                format_currency(res_1d["VaR_95%"]),
                                format_currency(res_1d["CVaR_95%"]),
                                format_currency(res_10d["VaR_95%"]),
                                format_currency(res_10d["CVaR_95%"]),
                            ],
                            "99% 置信度": [
                                format_currency(res_1d["VaR_99%"]),
                                format_currency(res_1d["CVaR_99%"]),
                                format_currency(res_10d["VaR_99%"]),
                                format_currency(res_10d["CVaR_99%"]),
                            ],
                        })
                        st.dataframe(summary_table, width="stretch", hide_index=True)

                        # =============================================
                        # Position Detail Table
                        # =============================================
                        st.divider()
                        st.subheader("📋 持仓明细 (Position Breakdown)")

                        if active_positions:
                            pos_df = pd.DataFrame(active_positions)

                            # Per-position VaR (individual, undiversified)
                            asset_col_map = {col: i for i, col in enumerate(returns.columns)}
                            individual_vars = []
                            for p in active_positions:
                                asset_id = p["AssetID"]
                                if asset_id in asset_col_map:
                                    asset_idx = asset_col_map[asset_id]
                                    # Single-asset position vector
                                    single_pos = np.zeros(len(returns.columns))
                                    single_pos[asset_idx] = position_vector[asset_idx]

                                    cov_matrix = engine.calculate_ewma_covariance()
                                    single_var = np.sqrt(single_pos @ cov_matrix @ single_pos)
                                    daily_vol = np.sqrt(cov_matrix[asset_idx, asset_idx])

                                    individual_vars.append({
                                        "DailyVol": f"{daily_vol * 100:.2f}%",
                                        "VaR_95": format_currency(single_var * 1.6449),
                                        "CVaR_95": format_currency(single_var * norm.pdf(1.6449) / 0.05),
                                        "VaR_99": format_currency(single_var * 2.3263),
                                        "CVaR_99": format_currency(single_var * norm.pdf(2.3263) / 0.01),
                                    })
                                else:
                                    individual_vars.append({
                                        "DailyVol": "-",
                                        "VaR_95": "-", "CVaR_95": "-",
                                        "VaR_99": "-", "CVaR_99": "-",
                                    })

                            # Build display table
                            detail_data = []
                            for p, v in zip(active_positions, individual_vars):
                                detail_data.append({
                                    "品种": p["ProductCode"],
                                    "类型": p["PositionType"],
                                    "合约月份": p["ContractMonth"],
                                    "数量": f"{p['Quantity']:,.0f} {p['Unit']}",
                                    "价格": f"{p['Price']:,.2f}",
                                    "名义金额(CNY)": f"{p['Notional_CNY']:,.0f}",
                                    "方向": p["Direction"],
                                    "日波动率": v["DailyVol"],
                                    "VaR 95%": v["VaR_95"],
                                    "CVaR 95%": v["CVaR_95"],
                                    "VaR 99%": v["VaR_99"],
                                    "CVaR 99%": v["CVaR_99"],
                                })

                            st.dataframe(
                                pd.DataFrame(detail_data),
                                width="stretch",
                                hide_index=True,
                            )

                            # Long/Short summary
                            total_long = sum(p["Notional_CNY"] for p in active_positions if p["Direction"] == "多头")
                            total_short = sum(p["Notional_CNY"] for p in active_positions if p["Direction"] == "空头")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("多头敞口", format_currency(total_long))
                            with col2:
                                st.metric("空头敞口", format_currency(total_short))
                            with col3:
                                st.metric("净敞口", format_currency(total_long - total_short))

                        # Volatility & Correlation
                        st.divider()
                        with st.expander("📈 波动率与相关性分析"):
                            vol_series = engine.get_individual_volatilities()
                            active_asset_ids = [p["AssetID"] for p in active_positions]
                            active_vols = vol_series[vol_series.index.isin(active_asset_ids)]

                            if not active_vols.empty:
                                st.subheader("年化波动率")
                                vol_df = pd.DataFrame({
                                    "资产": active_vols.index,
                                    "年化波动率": [f"{v*100:.2f}%" for v in active_vols.values]
                                })
                                st.dataframe(vol_df, width="stretch", hide_index=True)

                            corr_matrix = engine.get_correlation_matrix()
                            if len(active_asset_ids) > 1:
                                st.subheader("相关系数矩阵")
                                active_corr = corr_matrix.loc[
                                    corr_matrix.index.isin(active_asset_ids),
                                    corr_matrix.columns.isin(active_asset_ids)
                                ]
                                st.dataframe(
                                    active_corr.style.format("{:.2%}").background_gradient(
                                        cmap="RdYlGn", vmin=-1, vmax=1
                                    ),
                                    width="stretch"
                                )

    # =========================================================================
    # Footer
    # =========================================================================

    st.divider()
    st.caption(f"""
    **模型说明:**
    - 方法: 参数法VaR (方差-协方差法)
    - 波动率: EWMA衰减因子 λ = {decay_factor}
    - 回看周期: {lookback} 交易日
    - 10日VaR: 时间平方根缩放 (√10)
    - 数据: {len(products)} 个产品 × 2 种价格类型 = {len(asset_metadata)} 个资产
    """)
    st.caption("© 2026 ITG-X系统 | Powered by EWMA Risk Engine")


if __name__ == "__main__":
    main()
