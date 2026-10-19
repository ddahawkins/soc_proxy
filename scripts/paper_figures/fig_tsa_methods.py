"""Plot pooled LDES capacity errors across TSA methods (10-year cases).

This version pools the selected representative-period counts k (default: 20, 40,
60) into a single distribution within each TSA method / W_CDS column.

Visual encoding:
    - Main x groups: TSA methods
    - Within each method: W_CDS = 0, 0.5, 1, left to right
    - Colour: W_CDS using the paper-wide mapping
        * W_CDS = 0   -> grey
        * positive W_CDS values mapped onto plasma from 0 to 0.9, calibrated over
          the full positive range 0.25 to 1.0 so that colours match other figures
    - Circles: individual country / weather-horizon / case observations
    - Short black horizontal bars: medians per method / W_CDS group
    - No horizontal jitter or IQR whiskers

Input files:
    results/results_2_5_10_year/parameters.parquet
    results/results_2_5_10_year/investment_metrics.parquet

Outputs (PDF and summary CSV only):
    results/figures/fig_tsa_methods_points_pooled/
        fig_tsa_methods_points_pooled.pdf
        fig_tsa_methods_points_pooled_summary.csv

Use --ymax 100 if an upper y-axis cap of 100% is wanted. Data outside a
requested limit remain in all median/summary calculations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

DEFAULT_SOURCE_DIR = Path("results/results_2_5_10_year")
DEFAULT_OUTPUT_DIR = Path("results/figures/fig_tsa_methods_points_pooled")
OUTPUT_STEM = "fig_tsa_methods_points_pooled"

DEFAULT_WIDTH_PX = 1800
DEFAULT_HEIGHT_PX = 1200
DEFAULT_DPI = 300

DEFAULT_K_VALUES = (20, 40, 60)
WP_ORDER = (0.0, 0.5, 1.0)

TARGET_HORIZON_YEARS = 10.0
HORIZON_TOLERANCE_YEARS = 0.10

PLASMA_MIN = 0.0
PLASMA_MAX = 0.90
WP_POSITIVE_MIN = 0.25
WP_POSITIVE_MAX = 1.0

POINT_SIZE = 24
POINT_ALPHA = 0.65
MEDIAN_HALF_WIDTH = 0.06
MEDIAN_LINEWIDTH = 2.5
MEDIAN_UNDERLAY_LINEWIDTH = 4.0

WP_GROUP_WIDTH = 0.72
BETWEEN_METHOD_GAP = 0.40

LDES_TICK_INTERVAL = 10
LDES_METRIC = "ldes_capacity_error_signed"

METHOD_ORDER = [
    "hierarchical_medoid",
    "kmeans_medoid",
    "hierarchical_distribution_local",
    "hierarchical_distribution_minmax_local",
]

METHOD_LABELS = {
    "hierarchical_medoid": "Hierarchical\n+ medoid",
    "kmeans_medoid": "k-means\n+ medoid",
    "hierarchical_distribution_local": "Hierarchical\n+ distribution",
    "hierarchical_distribution_minmax_local": "Hierarchical\n+ distribution\nminmax",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot TSA-method LDES capacity errors as pooled, unjittered observations "
            "+ median bars."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--width-px", type=int, default=DEFAULT_WIDTH_PX)
    parser.add_argument("--height-px", type=int, default=DEFAULT_HEIGHT_PX)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional y-axis upper limit in percent, e.g. --ymax 100.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Optional y-axis lower limit in percent.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], *, table_name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}\n"
            f"Available columns:\n{sorted(frame.columns)}"
        )


def filter_ten_year_runs(parameters: pd.DataFrame) -> pd.DataFrame:
    parameters = parameters.copy()
    parameters["start_date"] = pd.to_datetime(parameters["start_date"])
    parameters["end_date"] = pd.to_datetime(parameters["end_date"])
    parameters["horizon_years"] = (
        parameters["end_date"] - parameters["start_date"]
    ).dt.total_seconds() / (365.2425 * 24 * 60 * 60)

    parameters = parameters.loc[
        np.isclose(
            parameters["horizon_years"],
            TARGET_HORIZON_YEARS,
            atol=HORIZON_TOLERANCE_YEARS,
            rtol=0.0,
        )
    ].copy()
    if parameters.empty:
        raise ValueError("No approximately 10-year cases were found.")
    return parameters


def infer_representation_scope(parameters: pd.DataFrame) -> pd.Series:
    for column in ("representation_scope", "scope"):
        if column in parameters.columns:
            return parameters[column].astype("string").str.lower().str.strip().fillna("")

    names = parameters["experiment_name"].astype("string").str.lower().fillna("")
    scope = pd.Series("", index=parameters.index, dtype="string")
    scope.loc[names.str.contains("local", regex=False)] = "local"
    scope.loc[names.str.contains("global", regex=False)] = "global"
    return scope


def assign_method_key(parameters: pd.DataFrame) -> pd.Series:
    cluster = parameters["cluster_method"].astype(str).str.lower()
    representation = parameters["representation_method"].astype(str).str.lower()
    scope = infer_representation_scope(parameters)
    method_key = pd.Series(pd.NA, index=parameters.index, dtype="string")

    method_key.loc[cluster.eq("hierarchical") & representation.eq("medoid")] = (
        "hierarchical_medoid"
    )
    method_key.loc[cluster.eq("kmeans") & representation.eq("medoid")] = "kmeans_medoid"
    method_key.loc[
        cluster.eq("hierarchical")
        & representation.eq("distribution")
        & scope.eq("local")
    ] = "hierarchical_distribution_local"
    method_key.loc[
        cluster.eq("hierarchical")
        & representation.eq("distribution_minmax")
        & scope.eq("local")
    ] = "hierarchical_distribution_minmax_local"
    return method_key


def load_results(source_dir: Path, *, k_values: list[int]) -> pd.DataFrame:
    """Load LDES metric data and validate pooled method / W_CDS channels."""
    k_values = sorted(set(int(k) for k in k_values))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("Supply one or more positive representative-period counts.")

    parameters = pd.read_parquet(source_dir / "parameters.parquet")
    require_columns(
        parameters,
        {
            "case_id",
            "country",
            "start_date",
            "end_date",
            "k_periods",
            "lambda_soc",
            "cluster_method",
            "representation_method",
            "experiment_name",
        },
        table_name="parameters.parquet",
    )

    parameters = filter_ten_year_runs(parameters)
    parameters["case_id"] = parameters["case_id"].astype(str)
    parameters["k_periods"] = parameters["k_periods"].astype(int)
    parameters["lambda_soc"] = parameters["lambda_soc"].astype(float)
    parameters["method_key"] = assign_method_key(parameters)

    weight_mask = np.zeros(len(parameters), dtype=bool)
    for wp in WP_ORDER:
        weight_mask |= np.isclose(parameters["lambda_soc"], wp)

    selected = parameters.loc[
        parameters["method_key"].isin(METHOD_ORDER)
        & parameters["k_periods"].isin(k_values)
        & weight_mask
    ].copy()

    if selected.empty:
        raise ValueError("No cases match the requested methods, k values, and W_CDS weights.")

    for wp in WP_ORDER:
        selected.loc[np.isclose(selected["lambda_soc"], wp), "lambda_soc"] = wp

    duplicate_cases = selected["case_id"].duplicated(keep=False)
    if duplicate_cases.any():
        raise ValueError(
            "Duplicate case_id values in selected parameters:\n"
            + selected.loc[duplicate_cases, "case_id"].head(20).to_string(index=False)
        )

    expected_channels = pd.MultiIndex.from_product(
        [METHOD_ORDER, WP_ORDER],
        names=["method_key", "lambda_soc"],
    )
    present_channels = pd.MultiIndex.from_frame(
        selected[["method_key", "lambda_soc"]].drop_duplicates()
    )
    missing_channels = expected_channels.difference(present_channels)
    if len(missing_channels):
        examples = "\n".join(f"  {m}, W_CDS={wp:g}" for m, wp in missing_channels)
        raise ValueError(
            "Missing TSA method / W_CDS channels. Check available data:\n" + examples
        )

    requested_k = set(k_values)
    missing_k_messages: list[str] = []
    for method in METHOD_ORDER:
        for wp in WP_ORDER:
            subset = selected.loc[
                selected["method_key"].eq(method)
                & np.isclose(selected["lambda_soc"], wp)
            ]
            available_k = set(int(x) for x in subset["k_periods"].unique())
            missing_k = sorted(requested_k - available_k)
            if missing_k:
                missing_k_messages.append(
                    f"{method}, W_CDS={wp:g}: missing k={missing_k}"
                )
    if missing_k_messages:
        raise ValueError(
            "The pooled channels do not all contain the same requested k subset:\n"
            + "\n".join(missing_k_messages)
        )

    investment = pd.read_parquet(source_dir / "investment_metrics.parquet")
    require_columns(
        investment,
        {"case_id", "metric", "value"},
        table_name="investment_metrics.parquet",
    )
    investment["case_id"] = investment["case_id"].astype(str)
    investment = investment.loc[
        investment["case_id"].isin(selected["case_id"])
        & investment["metric"].eq(LDES_METRIC),
        ["case_id", "value"],
    ].copy()

    duplicate_metrics = investment.duplicated(subset=["case_id"], keep=False)
    if duplicate_metrics.any():
        raise ValueError(
            "Multiple LDES metrics for the same case_id:\n"
            + investment.loc[duplicate_metrics, "case_id"].head(20).to_string(index=False)
        )

    investment = investment.rename(columns={"value": "ldes_capacity_error_signed"})
    data = selected.merge(investment, on="case_id", how="left", validate="one_to_one")
    data["ldes_capacity_error_pct"] = 100.0 * pd.to_numeric(
        data["ldes_capacity_error_signed"], errors="raise"
    )

    invalid = ~np.isfinite(data["ldes_capacity_error_pct"].to_numpy(dtype=float))
    if invalid.any():
        raise ValueError(
            f"Missing/nonfinite LDES capacity errors for {invalid.sum()} cases:\n"
            + data.loc[invalid, "case_id"].head(20).to_string(index=False)
        )

    return data


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = data.groupby(["method_key", "lambda_soc"], as_index=False).agg(
        n=("ldes_capacity_error_pct", "size"),
        k_min=("k_periods", "min"),
        k_max=("k_periods", "max"),
        k_unique=("k_periods", lambda x: int(pd.Series(x).nunique())),
        min=("ldes_capacity_error_pct", "min"),
        q25=("ldes_capacity_error_pct", lambda x: x.quantile(0.25)),
        median=("ldes_capacity_error_pct", "median"),
        q75=("ldes_capacity_error_pct", lambda x: x.quantile(0.75)),
        max=("ldes_capacity_error_pct", "max"),
    )
    return summary


def build_wp_colours() -> dict[float, object]:
    """Use paper-wide W_P colours, with W_P=0 shown in grey."""
    cmap = plt.get_cmap("plasma")

    wp_positive_min = 0.25
    wp_positive_max = 1.0

    colours = {}

    for wp in WP_ORDER:
        if np.isclose(wp, 0.0):
            colours[wp] = "0.45"
        else:
            position = (
                (wp - wp_positive_min)
                / (wp_positive_max - wp_positive_min)
            )
            colours[wp] = cmap(
                PLASMA_MIN
                + position * (PLASMA_MAX - PLASMA_MIN)
            )

    return colours


def build_positions() -> tuple[dict[str, float], dict[float, float]]:
    method_step = WP_GROUP_WIDTH + BETWEEN_METHOD_GAP
    method_positions = {m: index * method_step for index, m in enumerate(METHOD_ORDER)}
    wp_offsets_array = np.linspace(-WP_GROUP_WIDTH / 2, WP_GROUP_WIDTH / 2, len(WP_ORDER))
    wp_offsets = dict(zip(WP_ORDER, wp_offsets_array, strict=True))
    return method_positions, wp_offsets


def set_ldes_axis(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    values = data["ldes_capacity_error_pct"].to_numpy(dtype=float)
    lo = min(0.0, float(values.min()))
    hi = max(0.0, float(values.max()))
    rounded_min = LDES_TICK_INTERVAL * np.floor(lo / LDES_TICK_INTERVAL)
    rounded_max = LDES_TICK_INTERVAL * np.ceil(hi / LDES_TICK_INTERVAL)
    if np.isclose(rounded_min, rounded_max):
        rounded_min -= LDES_TICK_INTERVAL
        rounded_max += LDES_TICK_INTERVAL
    pad = max(0.6, 0.025 * (rounded_max - rounded_min))
    ax.set_ylim(rounded_min - pad, rounded_max + pad)
    ax.yaxis.set_major_locator(MultipleLocator(LDES_TICK_INTERVAL))
    ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--", zorder=1)

    if ymin is not None or ymax is not None:
        low, high = ax.get_ylim()
        low = float(ymin) if ymin is not None else low
        high = float(ymax) if ymax is not None else high
        if low >= high:
            raise ValueError(f"Invalid requested y-axis limits: {low} >= {high}")
        ax.set_ylim(low, high)
        clipped = int(((values < low) | (values > high)).sum())
        if clipped:
            print(
                f"WARNING: {clipped}/{len(values)} observations lie outside "
                f"the displayed y-range [{low:g}, {high:g}]%. "
                "These cases are retained in median calculations."
            )


def legend_label(weight: float) -> str:
    if np.isclose(weight, 0.0):
        return r"$W_{CDS}=0$ (Original TSA)"
    return rf"$W_{{CDS}}={weight:g}$"


def make_figure(
    data: pd.DataFrame,
    *,
    width_px: int,
    height_px: int,
    dpi: int,
    ymin: float | None = None,
    ymax: float | None = None,
) -> plt.Figure:
    if min(width_px, height_px, dpi) <= 0:
        raise ValueError("--width-px, --height-px and --dpi must be positive.")

    figsize = (width_px / dpi, height_px / dpi)
    wp_colours = build_wp_colours()
    method_positions, wp_offsets = build_positions()

    rc = {
        "font.size": 10,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for method in METHOD_ORDER:
            method_data = data.loc[data["method_key"].eq(method)]
            for wp in WP_ORDER:
                category_data = method_data.loc[np.isclose(method_data["lambda_soc"], wp)]
                if category_data.empty:
                    raise ValueError(f"Unexpected empty category: {method}, W_CDS={wp}")

                ys = category_data["ldes_capacity_error_pct"].to_numpy(dtype=float)
                centre = method_positions[method] + wp_offsets[wp]

                ax.scatter(
                    np.full(ys.size, centre),
                    ys,
                    s=POINT_SIZE,
                    color=wp_colours[wp],
                    alpha=POINT_ALPHA,
                    edgecolors="none",
                    zorder=2,
                )

                median = float(np.median(ys))
                ax.hlines(
                    median,
                    centre - MEDIAN_HALF_WIDTH,
                    centre + MEDIAN_HALF_WIDTH,
                    color="white",
                    linewidth=MEDIAN_UNDERLAY_LINEWIDTH,
                    zorder=3,
                )
                ax.hlines(
                    median,
                    centre - MEDIAN_HALF_WIDTH,
                    centre + MEDIAN_HALF_WIDTH,
                    color="black",
                    linewidth=MEDIAN_LINEWIDTH,
                    zorder=4,
                )

        method_centres = [method_positions[m] for m in METHOD_ORDER]
        for left, right in zip(method_centres[:-1], method_centres[1:], strict=True):
            ax.axvline((left + right) / 2, color="0.90", linewidth=0.7, zorder=0)

        full_span = max(abs(x) for x in wp_offsets.values())
        horizontal_pad = BETWEEN_METHOD_GAP / 2 + MEDIAN_HALF_WIDTH
        ax.set_xlim(
            method_centres[0] - full_span - horizontal_pad,
            method_centres[-1] + full_span + horizontal_pad,
        )
        ax.set_xticks(method_centres)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
        ax.set_ylabel("LDES capacity error (%)")
        ax.grid(axis="y", linewidth=0.6, alpha=0.28, zorder=0)
        set_ldes_axis(ax, data, ymin=ymin, ymax=ymax)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=wp_colours[wp],
                markeredgecolor="none",
                markersize=6.2,
                label=legend_label(wp),
            )
            for wp in WP_ORDER
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.985),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=1.3,
            handletextpad=0.55,
        )
        fig.text(
            0.54,
            0.9,
            (
                r"Within each method: $W_{CDS}=0,\ 0.5,\ 1$ from left to right."
            ),
            ha="center",
            va="top",
            fontsize=10,
            color="0.4",
        )
        fig.subplots_adjust(left=0.12, right=0.985, top=0.85, bottom=0.18)
        return fig


def save_outputs(fig: plt.Figure, summary: pd.DataFrame, *, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    summary_path = output_dir / f"{OUTPUT_STEM}_summary.csv"
    fig.savefig(pdf_path, facecolor="white")
    summary.to_csv(summary_path, index=False)
    return pdf_path, summary_path


def main() -> None:
    args = parse_args()
    data = load_results(args.source_dir, k_values=args.k_values)
    summary = build_summary(data)

    print("TSA method sensitivity — pooled observations + medians")
    print("=" * 56)
    print(f"Cases:          {len(data)}")
    print(f"Countries:      {sorted(data['country'].astype(str).unique())}")
    print(f"Horizon:        ~{TARGET_HORIZON_YEARS:g} years")
    print(f"k pooled:       {sorted(data['k_periods'].unique())}")
    print(f"W_CDS:          {sorted(data['lambda_soc'].unique())}")
    print(f"n / channel:    {summary['n'].min()}–{summary['n'].max()}")
    print(f"unique k / ch.: {summary['k_unique'].min()}–{summary['k_unique'].max()}")
    if summary["n"].nunique() > 1:
        print("NOTE: Channel sample counts differ; check the summary CSV.")
    if summary["k_unique"].nunique() > 1 or int(summary["k_unique"].min()) != len(set(args.k_values)):
        print("NOTE: Not all pooled channels contain the same number of unique k values.")

    fig = make_figure(
        data,
        width_px=args.width_px,
        height_px=args.height_px,
        dpi=args.dpi,
        ymin=args.ymin,
        ymax=args.ymax,
    )
    outputs = save_outputs(fig, summary, output_dir=args.output_dir)

    print("\nSaved:")
    for output in outputs:
        print(f"  {output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
