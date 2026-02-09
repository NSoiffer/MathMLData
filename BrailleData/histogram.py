from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
from pandas import DataFrame, ExcelWriter


# ---------- Axis / size helpers ----------

def _apply_x_axis(chart: Any) -> None:
    """
    Apply consistent x-axis settings:
      - Show only 0–45
      - Vertical gridlines every 5
      - No minor gridlines
      - Label: Character Count
    """
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


# ---------- Chart / combine helpers ----------

def _create_scatter_chart(
    workbook: Any,
    sheet_name: str,
    start_row: int,
    end_row: int,
    series: list[dict[str, object]],
    title: str,
    height: int = 580,
) -> Any:
    """
    Create a scatter chart with one or more series.

    Each series dict must have:
      - "name": str
      - "col_idx": int
      - "color": str
    """
    chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})  # type: ignore[attr-defined]
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

    chart.set_title({"name": title})  # type: ignore[attr-defined]
    _apply_x_axis(chart)
    chart.set_size({"height": height})  # type: ignore[attr-defined]
    return chart


def _combine_datasets(datasets: list[DataFrame]) -> DataFrame | None:
    """
    Combine multiple histograms (with 'Character Count' and 'Frequency')
    by summing frequencies and recomputing percentages.
    """
    if not datasets:
        return None

    combined = pd.concat(datasets, ignore_index=True)
    if "Frequency" not in combined.columns:
        return None

    grouped = combined.groupby("Character Count", as_index=False)["Frequency"].sum()
    total = grouped["Frequency"].sum()
    grouped["Frequency %"] = (grouped["Frequency"] / total * 100).round(2)  # type: ignore[attr-defined]
    grouped["Cumulative %"] = grouped["Frequency %"].cumsum().round(2)  # type: ignore[attr-defined]
    return grouped  # type: ignore[return-value]


# ---------- Per-sheet histogram ----------

def generate_line_histogram(
    directory: str,
    file_pattern: str,
    writer: ExcelWriter | None = None,
) -> tuple[ExcelWriter | None, DataFrame, str]:
    files = sorted(Path(directory).glob(file_pattern))
    if not files:
        return writer, pd.DataFrame(), ""

    lengths: list[int] = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                lengths.append(len(line.rstrip("\n")))

    if not lengths:
        return writer, pd.DataFrame(), ""

    df = pd.DataFrame({"Character Count": lengths})
    hist = (
        df["Character Count"]
        .value_counts()
        .sort_index()
        .reindex(range(1, 81), fill_value=0)
        .reset_index()
    )
    hist.columns = ["Character Count", "Frequency"]

    total = hist["Frequency"].sum()
    hist["Frequency %"] = (hist["Frequency"] / total * 100).round(2)
    hist["Cumulative %"] = hist["Frequency %"].cumsum().round(2)

    if writer is None:
        try:
            writer = ExcelWriter("braille-lengths.xlsx", engine="xlsxwriter")
        except PermissionError:
            print(
                "ERROR: Could not open 'braille-lengths.xlsx' for writing.\n"
                "The file is probably open in Excel. "
                "Please close it and run the program again."
            )
            sys.exit(1)

    sheet_name = directory.replace("/", "_")
    hist.to_excel(writer, sheet_name=sheet_name, index=False)

    workbook = writer.book  # type: ignore[attr-defined]
    worksheet = writer.sheets[sheet_name]  # type: ignore[attr-defined]

    start_row = 1
    end_row = start_row + len(hist) - 1
    x_end_row = min(end_row, start_row + 44)   # CC = 45

    # Frequency chart
    dist_chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})  # type: ignore[attr-defined]
    dist_chart.set_legend({"none": True})  # type: ignore[attr-defined]
    dist_chart.add_series(  # type: ignore[attr-defined]
        {
            "name": "Frequency %",
            "categories": [sheet_name, start_row, 0, x_end_row, 0],
            "values": [sheet_name, start_row, 2, x_end_row, 2],
            "marker": {"type": "none"},
        }
    )
    dist_chart.set_y_axis({"min": 0, "max": 30, "major_unit": 10})  # type: ignore[attr-defined]
    _apply_x_axis(dist_chart)
    worksheet.insert_chart("G2", dist_chart)  # type: ignore[attr-defined]

    # Cumulative chart
    cum_chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})  # type: ignore[attr-defined]
    cum_chart.set_legend({"none": True})  # type: ignore[attr-defined]
    cum_chart.add_series(  # type: ignore[attr-defined]
        {
            "name": "Cumulative %",
            "categories": [sheet_name, start_row, 0, x_end_row, 0],
            "values": [sheet_name, start_row, 3, x_end_row, 3],
            "marker": {"type": "none"},
        }
    )
    cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})  # type: ignore[attr-defined]
    _apply_x_axis(cum_chart)
    worksheet.insert_chart("G20", cum_chart)  # type: ignore[attr-defined]

    return writer, hist, sheet_name


# ---------- Summary helpers ----------

def _build_combined_dataframe(
    all_data: list[tuple[DataFrame, str]],
) -> DataFrame:
    combined_df = pd.DataFrame({"Character Count": range(1, 81)})

    for hist_data, name in all_data:
        subset = hist_data[["Character Count", "Frequency %", "Cumulative %"]].copy()
        subset = subset.rename(
            columns={
                "Frequency %": f"Frequency %_{name}",
                "Cumulative %": f"Cumulative %_{name}",
            }
        )  # type: ignore[call-overload]
        combined_df = combined_df.merge(subset, on="Character Count", how="left")

    return combined_df.fillna(0)


def _add_per_code_combined_columns(
    combined_df: DataFrame,
    all_data: list[tuple[DataFrame, str]],
    codes: list[str],
) -> DataFrame:
    for code in codes:
        code_datasets = [
            hist for hist, name in all_data
            if f"_{code}_" in name
        ]
        if not code_datasets:
            continue

        combined = _combine_datasets(code_datasets)
        if combined is None:
            continue

        combined_df = combined_df.merge(
            combined[["Character Count", "Frequency %", "Cumulative %"]].rename(
                columns={
                    "Frequency %": f"Frequency %_{code}_Combined",
                    "Cumulative %": f"Cumulative %_{code}_Combined",
                }
            ),  # type: ignore[call-overload]
            on="Character Count",
            how="left",
        )

    return combined_df.fillna(0)


def _build_all_dataset_series(
    combined_df: DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    freq_cols = [c for c in combined_df.columns if c.startswith("Frequency %_")]
    cum_cols = [c for c in combined_df.columns if c.startswith("Cumulative %_")]

    colors = [
        "#4F81BD", "#C0504D", "#9BBB59", "#8064A2",
        "#F79646", "#1F497D", "#2F5597", "#953735",
    ]

    freq_series: list[dict[str, object]] = [
        {
            "name": col.replace("Frequency %_", ""),
            "col_idx": combined_df.columns.get_loc(col),
            "color": colors[i % len(colors)],
        }
        for i, col in enumerate(freq_cols)
    ]

    cum_series: list[dict[str, object]] = [
        {
            "name": col.replace("Cumulative %_", ""),
            "col_idx": combined_df.columns.get_loc(col),
            "color": colors[i % len(colors)],
        }
        for i, col in enumerate(cum_cols)
    ]

    return freq_series, cum_series


def _build_per_code_series(
    combined_df: DataFrame,
    code: str,
) -> tuple[list[dict[str, object]] | None, list[dict[str, object]] | None]:
    fcol = f"Frequency %_{code}_Combined"
    ccol = f"Cumulative %_{code}_Combined"

    if fcol not in combined_df.columns or ccol not in combined_df.columns:
        return None, None

    freq_series: list[dict[str, object]] = [
        {
            "name": f"{code} Combined",
            "col_idx": combined_df.columns.get_loc(fcol),
            "color": "#4F81BD",
        }
    ]
    cum_series: list[dict[str, object]] = [
        {
            "name": f"{code} Combined",
            "col_idx": combined_df.columns.get_loc(ccol),
            "color": "#4F81BD",
        }
    ]
    return freq_series, cum_series


def _build_four_line_series(
    combined_df: DataFrame,
    codes: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    colors = [
        "#4F81BD", "#C0504D", "#9BBB59", "#8064A2",
    ]

    freq_series: list[dict[str, object]] = []
    cum_series: list[dict[str, object]] = []

    for i, code in enumerate(codes):
        fcol = f"Frequency %_{code}_Combined"
        ccol = f"Cumulative %_{code}_Combined"
        if fcol in combined_df.columns and ccol in combined_df.columns:
            freq_series.append(
                {
                    "name": code,
                    "col_idx": combined_df.columns.get_loc(fcol),
                    "color": colors[i % len(colors)],
                }
            )
            cum_series.append(
                {
                    "name": code,
                    "col_idx": combined_df.columns.get_loc(ccol),
                    "color": colors[i % len(colors)],
                }
            )

    return freq_series, cum_series


def _write_summary_dataframe(
    writer: ExcelWriter,
    combined_df: DataFrame,
    sheet_name: str = "Summary",
) -> Any:
    combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    return writer.sheets[sheet_name]  # type: ignore[attr-defined]


def _insert_all_charts(
    worksheet: Any,
    workbook: Any,
    sheet_name: str,
    start_row: int,
    x_end_row: int,
    freq_series: list[dict[str, object]],
    cum_series: list[dict[str, object]],
) -> None:
    dist_chart = _create_scatter_chart(
        workbook,
        sheet_name,
        start_row,
        x_end_row,
        freq_series,
        "Distribution of Line Lengths - All Datasets",
    )
    dist_chart.set_y_axis({"min": 0, "max": 30, "major_unit": 10})  # type: ignore[attr-defined]

    cum_chart = _create_scatter_chart(
        workbook,
        sheet_name,
        start_row,
        x_end_row,
        cum_series,
        "Cumulative Distribution - All Datasets",
    )
    cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})  # type: ignore[attr-defined]

    worksheet.insert_chart("D90", dist_chart)  # type: ignore[attr-defined]
    worksheet.insert_chart("L90", cum_chart)  # type: ignore[attr-defined]


def _insert_per_code_charts(
    worksheet: Any,
    workbook: Any,
    sheet_name: str,
    start_row: int,
    x_end_row: int,
    codes: list[str],
    combined_df: DataFrame,
) -> None:
    row = 130
    for code in codes:
        freq_series, cum_series = _build_per_code_series(combined_df, code)
        if freq_series is None or cum_series is None:
            continue

        dist_chart = _create_scatter_chart(
            workbook,
            sheet_name,
            start_row,
            x_end_row,
            freq_series,
            f"Distribution - {code} Combined",
        )
        dist_chart.set_y_axis({"min": 0, "max": 30, "major_unit": 10})  # type: ignore[attr-defined]

        cum_chart = _create_scatter_chart(
            workbook,
            sheet_name,
            start_row,
            x_end_row,
            cum_series,
            f"Cumulative Distribution - {code} Combined",
        )
        cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})  # type: ignore[attr-defined]

        worksheet.insert_chart(f"D{row}", dist_chart)  # type: ignore[attr-defined]
        worksheet.insert_chart(f"L{row}", cum_chart)  # type: ignore[attr-defined]
        row += 40


def _insert_four_line_charts(
    worksheet: Any,
    workbook: Any,
    sheet_name: str,
    start_row: int,
    x_end_row: int,
    freq_series: list[dict[str, object]],
    cum_series: list[dict[str, object]],
) -> None:
    if not freq_series:
        return

    freq_chart = _create_scatter_chart(
        workbook,
        sheet_name,
        start_row,
        x_end_row,
        freq_series,
        "Distribution - Combined Highschool+College",
    )
    freq_chart.set_y_axis({"min": 0, "max": 30, "major_unit": 10})  # type: ignore[attr-defined]

    cum_chart = _create_scatter_chart(
        workbook,
        sheet_name,
        start_row,
        x_end_row,
        cum_series,
        "Cumulative - Combined Highschool+College",
    )
    cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})  # type: ignore[attr-defined]

    worksheet.insert_chart("T90", freq_chart)  # type: ignore[attr-defined]
    worksheet.insert_chart("AB90", cum_chart)  # type: ignore[attr-defined]


def _append_summary_statistics(
    worksheet: Any,
    combined_df: DataFrame,
    start_row: int,
    max_chart_row: int,
) -> None:
    """
    Append mean and median rows for frequency columns only,
    placed two rows after the last data row.
    """

    # Excel row numbers (1-based)
    first_data_excel_row = start_row + 1          # e.g. 2
    last_data_excel_row = start_row + max_chart_row  # e.g. 81

    # Stats should start two rows after last data row (e.g. 83)
    stats_excel_row = last_data_excel_row + 2     # e.g. 83

    # XlsxWriter uses 0-based row indices
    stats_row = stats_excel_row - 1               # e.g. 82

    freq_cols = [
        col for col in combined_df.columns
        if col.startswith("Frequency %_")
    ]

    worksheet.write(stats_row, 0, "Mean")         # type: ignore[attr-defined]
    worksheet.write(stats_row + 1, 0, "Median")   # type: ignore[attr-defined]

    for col in freq_cols:
        col_idx = combined_df.columns.get_loc(col)
        excel_col = chr(ord("A") + col_idx)  # type: ignore[operator]

        data_start = first_data_excel_row         # e.g. 2
        data_end = last_data_excel_row            # e.g. 81

        worksheet.write_formula(                  # type: ignore[attr-defined]
            stats_row,
            col_idx,
            f"=ROUND(AVERAGE({excel_col}{data_start}:{excel_col}{data_end}), 2)",
        )
        worksheet.write_formula(                  # type: ignore[attr-defined]
            stats_row + 1,
            col_idx,
            f"=ROUND(MEDIAN({excel_col}{data_start}:{excel_col}{data_end}), 2)",
        )


def _insert_width_fit_table_generic(
    worksheet: Any,
    workbook: Any,
    combined_df: DataFrame,
    codes: list[str],
    label: str,
    mode: str,  # "highschool" or "combined"
    start_row: int,
    start_col: int,
) -> None:
    """
    Generic table builder for cumulative % fit tables.

    mode = "highschool":
        use the original highschool dataset column for each code
    mode = "combined":
        use the per-code Combined column (Cumulative %_{code}_Combined)
    """

    widths = [14, 18, 20, 32, 40]

    # --- Formats ---
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

    # --- Column widths ---
    worksheet.set_column(start_col, start_col, 18)
    worksheet.set_column(start_col + 1, start_col + len(widths), 6)

    # --- Label row ---
    worksheet.write(start_row, start_col, label, label_fmt)

    # --- Header row ---
    header_row = start_row + 1
    worksheet.write(header_row, start_col, "Code / # Cells", header_left_fmt)

    for j, w in enumerate(widths):
        worksheet.write(header_row, start_col + 1 + j, w, header_right_fmt)

    # --- Data rows ---
    for i, code in enumerate(codes):
        excel_row = header_row + 1 + i

        worksheet.write(excel_row, start_col, code, text_fmt)

        # Decide which column to use
        if mode == "combined":
            colname = f"Cumulative %_{code}_Combined"
            if colname not in combined_df.columns:
                continue
        elif mode == "highschool":
            # Find the original highschool column for this code
            candidates = [
                c for c in combined_df.columns
                if c.startswith("Cumulative %_")
                and f"_{code}_" in c
                and "highschool" in c.lower()
            ]
            if not candidates:
                continue
            # If there are multiple, just take the first
            colname = candidates[0]
        else:
            continue

        for j, w in enumerate(widths):
            rows = combined_df.loc[
                combined_df["Character Count"] == w,
                colname,
            ]
            if rows.empty:
                continue

            value = rows.iloc[0]
            rounded = int(round(float(value)))

            worksheet.write(
                excel_row,
                start_col + 1 + j,
                rounded,
                number_fmt,
            )


# ---------- Summary sheet orchestrator ----------
def generate_summary_sheet(
    writer: ExcelWriter,
    all_data: list[tuple[DataFrame, str]],
    codes: list[str],
) -> None:
    sheet_name = "Summary"
    workbook = writer.book  # type: ignore[attr-defined]

    start_row = 1
    max_chart_row = 80
    end_row = start_row + max_chart_row - 1
    x_end_row = min(end_row, start_row + 44)   # CC = 45

    combined_df = _build_combined_dataframe(all_data)
    combined_df = _add_per_code_combined_columns(combined_df, all_data, codes)

    worksheet = _write_summary_dataframe(writer, combined_df, sheet_name)
    _insert_width_fit_table_generic(
        worksheet,
        workbook,
        combined_df,
        codes,
        label="Highschool",
        mode="highschool",
        start_row=129,  # Excel row 130
        start_col=21,   # Excel column V
    )
    _insert_width_fit_table_generic(
        worksheet,
        workbook,
        combined_df,
        codes,
        label="Combined",
        mode="combined",
        start_row=129 + 11,  # Excel row 140
        start_col=21,
    )

    freq_series, cum_series = _build_all_dataset_series(combined_df)
    _insert_all_charts(
        worksheet,
        workbook,
        sheet_name,
        start_row,
        x_end_row,
        freq_series,
        cum_series,
    )

    _insert_per_code_charts(
        worksheet,
        workbook,
        sheet_name,
        start_row,
        x_end_row,
        codes,
        combined_df,
    )

    four_freq, four_cum = _build_four_line_series(combined_df, codes)
    _insert_four_line_charts(
        worksheet,
        workbook,
        sheet_name,
        start_row,
        x_end_row,
        four_freq,
        four_cum,
    )

    _append_summary_statistics(worksheet, combined_df, start_row, max_chart_row)

    workbook.worksheets_objs.insert(0, workbook.worksheets_objs.pop())  # type: ignore[attr-defined]
    print(f"Successfully added summary sheet '{sheet_name}' to workbook.")


def main() -> None:
    codes = ["Nemeth", "UEB", "LaTeX", "LaTeX6", "ASCIIMath", "ASCIIMath6"]
    levels = ["highschool", "college"]

    writer: ExcelWriter | None = None
    all_data: list[tuple[DataFrame, str]] = []

    try:
        for code in codes:
            for level in levels:
                directory = f"Braille/{code}/{level}"
                file_pattern = "*.brls"

                writer, hist_data, sheet_name = generate_line_histogram(
                    directory=directory,
                    file_pattern=file_pattern,
                    writer=writer,
                )

                if not hist_data.empty:
                    all_data.append((hist_data, sheet_name))

        if all_data and writer is not None:
            generate_summary_sheet(writer, all_data, codes)

    finally:
        if writer is not None:
            try:
                writer.close()
                print("All sheets added to 'braille-lengths.xlsx'.")
            except PermissionError:
                print(
                    "ERROR: Could not write 'braille-lengths.xlsx'. "
                    "The file is probably open in Excel. "
                    "Please close it and run the program again."
                )
            except OSError as e:
                print(
                    f"ERROR: Could not write 'braille-lengths.xlsx' ({e}). "
                    "Please close the file if it is open and try again."
                )


if __name__ == "__main__":
    main()
