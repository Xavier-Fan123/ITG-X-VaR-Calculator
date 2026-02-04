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

    # Custom CSS for better styling
    st.markdown("""
    <style>
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
    # Position Input Form
    # =========================================================================

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

        # Create columns for better layout
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

    # =========================================================================
    # Calculate VaR
    # =========================================================================

    col_calc, col_clear = st.columns([1, 5])

    with col_calc:
        calculate_button = st.button("🧮 计算VaR", type="primary", width="stretch")

    with col_clear:
        if st.button("🗑️ 清空"):
            st.rerun()

    if calculate_button:
        # Build position vector
        position_vector = []
        active_positions = []

        # Get the asset order from returns columns
        for asset_id in returns.columns:
            if asset_id in positions_input:
                info = positions_input[asset_id]
                quantity = info["position"]  # User input is QUANTITY (e.g., tons)
                direction = info["direction"]
                asset = info["asset"]

                # Get current price for this asset
                current_price = latest_prices.get(asset_id, 0)

                # Calculate NOTIONAL VALUE = Quantity × Price
                notional = quantity * current_price

                # Apply direction (Long = positive, Short = negative)
                signed_notional = notional if direction == "Long" else -notional

                # Convert USD to CNY for USD-denominated assets
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

        # Check if any positions were entered
        if np.allclose(position_vector, 0):
            st.warning("⚠️ 请至少输入一个非零头寸")
        else:
            # Initialize Risk Engine
            try:
                engine = RiskEngine(returns, decay_factor=decay_factor)
                results = engine.get_var_results(position_vector)
            except Exception as e:
                st.error(f"❌ VaR计算错误: {str(e)}")
                st.stop()

            # =========================================================================
            # Display Results
            # =========================================================================

            st.header("📊 VaR计算结果")

            # Results table
            results_df = pd.DataFrame({
                "置信水平": ["95%", "99%"],
                "1日VaR (CNY)": [
                    format_currency(results["1-Day"]["95%"]),
                    format_currency(results["1-Day"]["99%"])
                ],
                "10日VaR (CNY)": [
                    format_currency(results["10-Day"]["95%"]),
                    format_currency(results["10-Day"]["99%"])
                ]
            })

            # Display as metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "1日VaR (95%)",
                    format_currency(results["1-Day"]["95%"]),
                    help="95%置信度下1日最大预期损失"
                )

            with col2:
                st.metric(
                    "1日VaR (99%)",
                    format_currency(results["1-Day"]["99%"]),
                    help="99%置信度下1日最大预期损失"
                )

            with col3:
                st.metric(
                    "10日VaR (95%)",
                    format_currency(results["10-Day"]["95%"]),
                    help="95%置信度下10日最大预期损失 (√10缩放)"
                )

            with col4:
                st.metric(
                    "10日VaR (99%)",
                    format_currency(results["10-Day"]["99%"]),
                    help="99%置信度下10日最大预期损失 (√10缩放)"
                )

            st.divider()

            # Active Positions Summary
            st.subheader("📋 持仓汇总")

            if active_positions:
                positions_df = pd.DataFrame(active_positions)

                # Rename columns to Chinese
                column_rename = {
                    "Asset": "资产",
                    "ID": "代码",
                    "Quantity": "数量",
                    "Unit": "单位",
                    "Price": "价格",
                    "Direction": "方向",
                    "Currency": "货币",
                    "Notional (CNY)": "名义金额(CNY)"
                }

                # Format the dataframe for display
                display_df = positions_df.copy()
                display_df["Price"] = display_df["Price"].apply(lambda x: f"{x:,.2f}")
                display_df["Notional (CNY)"] = display_df["Notional (CNY)"].apply(lambda x: f"{x:,.0f}")
                display_df["Direction"] = display_df["Direction"].apply(lambda x: "多头" if x == "Long" else "空头")
                display_df = display_df.rename(columns=column_rename)

                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True
                )

                # Summary stats
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

            # Volatility Analysis (Expandable)
            with st.expander("📈 波动率与相关性分析"):
                st.subheader("年化波动率 (活跃资产)")

                vol_series = engine.get_individual_volatilities()

                # Filter to show only active assets
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
                    # Filter correlation matrix to active assets
                    active_corr = corr_matrix.loc[
                        corr_matrix.index.isin(active_asset_ids),
                        corr_matrix.columns.isin(active_asset_ids)
                    ]

                    # Format as percentage
                    st.dataframe(
                        active_corr.style.format("{:.2%}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                        width="stretch"
                    )
                else:
                    st.info("输入2个以上资产头寸可查看相关系数矩阵")

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
