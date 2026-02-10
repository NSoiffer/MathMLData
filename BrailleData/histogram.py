from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
from pandas import DataFrame, ExcelWriter


# ---------- Axis / size helpers ----------

def _apply_x_axis(chart: Any) -> None:
    chart.set_x_axis({  # type: ignore[attr-defined]
        "name": "Character Count",
        "type": "value",
        "min": 0,
        "max": 45,
        "major_unit": 5,
        "minor_unit": None,
        "major_gridlines": {"visible": True},
        "minor_gridlines": {"visible": False},
        "minor_tick_mark": "none",
        "num_format": "0",
    })


# ---------- Chart helpers ----------
def _apply_legend_visibility(chart: Any, series_count: int) -> None:
    """Hide legend if only one series; show it otherwise."""
    if series_count <= 1:
        chart.set_legend({"none": True})  # type: ignore[attr-defined]
    else:
        chart.set_legend({"none": False})  # type: ignore[attr-defined]


def _create_scatter_chart(
    workbook: Any,
    sheet_name: str,
    start_row: int,
    end_row: int,
    series: list[dict[str, object]],
    title: str,
    height: int = 580,
) -> Any:
    chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})  # type: ignore[attr-defined]

    # Add all series
    for s in series:
        chart.add_series(  # type: ignore[attr-defined]
            {
                "name": s["name"],
                "categories": [sheet_name, start_row, 0, end_row, 0],
                "values": [sheet_name, start_row, s["col_idx"], end_row, s["col_idx"]],
                "line": {"color": s["color"]},
                "marker": {"type": "none"},
            }
        )

    # Title + axes
    chart.set_title({"name": title})  # type: ignore[attr-defined]
    _apply_x_axis(chart)
    chart.set_size({"height": height})  # type: ignore[attr-defined]

    # Hide legend if only one series
    if len(series) <= 1:
        chart.set_legend({"none": True})  # type: ignore[attr-defined]
    else:
        chart.set_legend({"none": False})  # type: ignore[attr-defined]

    return chart


# ---------- Histogram helpers ----------

def _compute_histogram_for_code(
    base_dir: str,
    code: str,
    levels: list[str],
) -> DataFrame:
    lengths: list[int] = []

    for level in levels:
        directory = Path(base_dir) / code / level
        files = sorted(directory.glob("*.brls"))
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    lengths.append(len(line.rstrip("\n")))

    if not lengths:
        return pd.DataFrame()

    df = pd.DataFrame({"Character Count": lengths})
    hist = (
        df["Character Count"]
        .value_counts()
        .sort_index()
        .reindex(range(1, 81), fill_value=0)
        .reset_index()
    )
    hist.columns = ["Character Count", "Frequency"]

    total: int = int(hist["Frequency"].sum())
    hist["Frequency %"] = (hist["Frequency"] / total * 100).round(2)
    hist["Cumulative %"] = hist["Frequency %"].cumsum().round(2)

    return hist


# --------- Summary helpers ----------
def _build_combined_dataframe(
    hist_by_code: dict[str, DataFrame],
) -> DataFrame:
    combined_df: DataFrame = pd.DataFrame({"Character Count": range(1, 81)})

    for code, hist in hist_by_code.items():
        if hist.empty:
            continue

        subset: DataFrame = hist[["Character Count", "Frequency %", "Cumulative %"]].copy()  # type: ignore[arg-type]
        subset = subset.rename(
            columns={
                "Frequency %": f"Frequency %_{code}",
                "Cumulative %": f"Cumulative %_{code}",
            }
        )
        combined_df = combined_df.merge(subset, on="Character Count", how="left")

    return combined_df.fillna(0)


def _color_for_code(code: str) -> str:
    """
    Return a distinct, color-blind-safe color for each code.
    Paired colors are used for 4-dot and 6-dot variants.
    """
    color_map: dict[str, str] = {
        "Nemeth":     "#1F77B4",  # blue
        "UEB":        "#D62728",  # red

        "LaTeX":      "#2CA02C",  # green
        "LaTeX6":     "#98DF8A",  # light green

        "ASCIIMath":  "#9467BD",  # purple
        "ASCIIMath6": "#C5B0D5",  # light purple
    }

    return color_map.get(code, "#7F7F7F")  # fallback gray


def _build_all_dataset_series(
    combined_df: DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    freq_cols = [c for c in combined_df.columns if c.startswith("Frequency %_")]
    cum_cols = [c for c in combined_df.columns if c.startswith("Cumulative %_")]

    freq_series = [
        {
            "name": col.replace("Frequency %_", ""),
            "col_idx": combined_df.columns.get_loc(col),  # type: ignore[arg-type]
            "color": _color_for_code(col.replace("Frequency %_", "")),
        }
        for col in freq_cols
    ]

    cum_series = [
        {
            "name": col.replace("Cumulative %_", ""),
            "col_idx": combined_df.columns.get_loc(col),  # type: ignore[arg-type]
            "color": _color_for_code(col.replace("Cumulative %_", "")),
        }
        for col in cum_cols
    ]

    return freq_series, cum_series


def _build_per_code_series(
    combined_df: DataFrame,
    code: str,
) -> tuple[list[dict[str, object]] | None, list[dict[str, object]] | None]:
    fcol = f"Frequency %_{code}"
    ccol = f"Cumulative %_{code}"

    if fcol not in combined_df.columns or ccol not in combined_df.columns:
        return None, None

    freq_series = [
        {
            "name": code,
            "col_idx": combined_df.columns.get_loc(fcol),  # type: ignore[arg-type]
            "color": _color_for_code(code),
        }
    ]
    cum_series = [
        {
            "name": code,
            "col_idx": combined_df.columns.get_loc(ccol),  # type: ignore[arg-type]
            "color": _color_for_code(code),
        }
    ]
    return freq_series, cum_series


def _insert_width_fit_table_generic(
    worksheet: Any,
    workbook: Any,
    combined_df: DataFrame,
    codes: list[str],
    label: str,
    start_row: int,
    start_col: int,
) -> None:
    widths = [14, 18, 20, 32, 40]

    label_fmt = workbook.add_format({
        "bold": True,
        "align": "left",
        "valign": "vcenter",
    })
    header_left_fmt = workbook.add_format({
        "bold": True,
        "align": "left",
        "valign": "vcenter",
        "border": 1,
    })
    header_right_fmt = workbook.add_format({
        "bold": True,
        "align": "right",
        "valign": "vcenter",
        "border": 1,
    })
    number_fmt = workbook.add_format({
        "align": "right",
        "border": 1,
    })
    text_fmt = workbook.add_format({
        "align": "left",
        "border": 1,
    })

    worksheet.set_column(start_col, start_col, 18)
    worksheet.set_column(start_col + 1, start_col + len(widths), 6)

    worksheet.write(start_row, start_col, label, label_fmt)

    header_row = start_row + 1
    worksheet.write(header_row, start_col, "Code / # Cells", header_left_fmt)

    for j, w in enumerate(widths):
        worksheet.write(header_row, start_col + 1 + j, w, header_right_fmt)

    for i, code in enumerate(codes):
        excel_row = header_row + 1 + i
        worksheet.write(excel_row, start_col, code, text_fmt)

        colname = f"Cumulative %_{code}"
        if colname not in combined_df.columns:
            continue

        for j, w in enumerate(widths):
            rows = combined_df.loc[
                combined_df["Character Count"] == w,
                colname,
            ]
            if rows.empty:
                continue

            value = float(rows.iloc[0])
            worksheet.write(
                excel_row,
                start_col + 1 + j,
                int(round(value)),
                number_fmt,
            )


def _append_summary_statistics(
    worksheet: Any,
    combined_df: DataFrame,
    start_row: int,
    data_rows: int,
) -> None:
    """
    Append mean and median rows for frequency columns only,
    placed two rows after the last data row.
    """
    first_data_excel_row: int = start_row + 1
    last_data_excel_row: int = start_row + data_rows
    stats_excel_row: int = last_data_excel_row + 2
    stats_row: int = stats_excel_row - 1

    freq_cols: list[str] = [
        col for col in combined_df.columns
        if col.startswith("Frequency %_")
    ]

    worksheet.write(stats_row, 0, "Mean")        # type: ignore[attr-defined]
    worksheet.write(stats_row + 1, 0, "Median")  # type: ignore[attr-defined] # noqa: E501
    for col in freq_cols:
        col_idx: int = combined_df.columns.get_loc(col)  # type: ignore[arg-type]
        excel_col: str = chr(ord("A") + col_idx)  # type: ignore[operator]

        data_start: int = first_data_excel_row
        data_end: int = last_data_excel_row

        worksheet.write_formula(  # type: ignore[attr-defined]
            stats_row,
            col_idx,
            f"=ROUND(AVERAGE({excel_col}{data_start}:{excel_col}{data_end}), 2)",
        )
        worksheet.write_formula(  # type: ignore[attr-defined]
            stats_row + 1,
            col_idx,
            f"=ROUND(MEDIAN({excel_col}{data_start}:{excel_col}{data_end}), 2)",
        )


# ---------- Summary sheet orchestrator ----------

def generate_summary_sheet(
    writer: ExcelWriter,
    hist_by_code: dict[str, DataFrame],
    codes: list[str],
) -> None:
    sheet_name = "Summary"
    workbook = writer.book  # type: ignore[attr-defined]

    combined_df = _build_combined_dataframe(hist_by_code)
    combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]  # type: ignore[attr-defined]

    start_row = 1
    data_rows = len(combined_df)
    x_end_row = min(start_row + 44, start_row + data_rows - 1)

    # --- Table ---
    table_start_row = data_rows + 5
    table_start_col = 21
    _insert_width_fit_table_generic(
        worksheet,
        workbook,
        combined_df,
        codes,
        label="Combined",
        start_row=table_start_row,
        start_col=table_start_col,
    )

    # --- All-codes charts ---
    freq_series, cum_series = _build_all_dataset_series(combined_df)

    dist_chart = _create_scatter_chart(
        workbook, sheet_name, start_row, x_end_row, freq_series, "Distribution"
    )
    dist_chart.set_y_axis({"min": 0, "max": 30, "major_unit": 10})
    worksheet.insert_chart("D90", dist_chart)

    cum_chart = _create_scatter_chart(
        workbook, sheet_name, start_row, x_end_row, cum_series, "Cumulative"
    )
    cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})
    worksheet.insert_chart("L90", cum_chart)

    # --- Per-code charts ---
    row = 130
    for code in codes:
        freq_s, cum_s = _build_per_code_series(combined_df, code)
        if freq_s is None:
            continue

        dist_chart = _create_scatter_chart(
            workbook, sheet_name, start_row, x_end_row, freq_s, f"{code} Distribution"
        )
        dist_chart.set_y_axis({"min": 0, "max": 30, "major_unit": 10})
        worksheet.insert_chart(f"D{row}", dist_chart)

        cum_chart = _create_scatter_chart(
            workbook, sheet_name, start_row, x_end_row, cum_s, f"{code} Cumulative"  # pyright: ignore[reportArgumentType]
        )
        cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})
        worksheet.insert_chart(f"L{row}", cum_chart)

        row += 40

    print(f"Successfully added summary sheet '{sheet_name}' to workbook.")


def main() -> None:
    codes: list[str] = [
        "Nemeth",
        "UEB",
        "LaTeX",
        "LaTeX6",
        "ASCIIMath",
        "ASCIIMath6",
    ]
    levels = ["highschool", "college"]
    base_dir = "Braille"

    writer: ExcelWriter | None = None
    hist_by_code: dict[str, DataFrame] = {}

    try:
        try:
            writer = ExcelWriter("braille-lengths.xlsx", engine="xlsxwriter")
        except PermissionError:
            print(
                "ERROR: Could not open 'braille-lengths.xlsx' for writing.\n"
                "The file is probably open in Excel. "
                "Please close it and run the program again."
            )
            sys.exit(1)

        for code in codes:
            hist = _compute_histogram_for_code(base_dir, code, levels)
            if not hist.empty:
                hist_by_code[code] = hist

        if hist_by_code:
            generate_summary_sheet(writer, hist_by_code, codes)

    finally:
        if writer is not None:
            try:
                writer.close()
                print("Summary sheet written to 'braille-lengths.xlsx'.")
            except PermissionError:
                print(
                    "ERROR: Could not write 'braille-lengths.xlsx'. "
                    "The file is probably open in Excel. "
                    "Please close it and try again."
                )
            except OSError as e:
                print(
                    f"ERROR: Could not write 'braille-lengths.xlsx' ({e}). "
                    "Please close the file if it is open and try again."
                )


if __name__ == "__main__":
    main()
