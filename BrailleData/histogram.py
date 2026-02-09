from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
from pandas import DataFrame, ExcelWriter


# ---------- Axis / size helpers ----------
def _apply_x_axis(chart) -> None:
    chart.set_x_axis({
        "name": "Character Count",
        "type": "value",            # <-- CRITICAL FIX
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
def _create_chart_with_series(
    workbook,
    sheet_name: str,
    start_row: int,
    end_row: int,
    series: list[dict[str, object]],
    title: str,
):
    """
    Create a line chart with one or more series.

    Each series dict must have:
      - "name": str
      - "col_idx": int
      - "color": str
    """
    chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    for s in series:
        chart.add_series(
            {
                "name": s["name"],
                "categories": [sheet_name, start_row, 0, end_row, 0],
                "values": [sheet_name, start_row, s["col_idx"], end_row, s["col_idx"]],
                "line": {"color": s["color"]},
            }
        )

    chart.set_title({"name": title})

    # Unified x‑axis: gridlines + labels at 5,10,15,...
    _apply_x_axis(chart)

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

    grouped = (
        combined.groupby("Character Count", as_index=False)["Frequency"].sum()
    )
    total = grouped["Frequency"].sum()
    grouped["Frequency %"] = (grouped["Frequency"] / total * 100).round(2)  # type: ignore[attr-defined]
    grouped["Cumulative %"] = grouped["Frequency %"].cumsum().round(2)  # type: ignore[attr-defined]
    return grouped  # type: ignore[return-value]  # type: ignore[return-value]


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

    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    start_row = 1
    end_row = start_row + len(hist) - 1
    x_end_row = min(end_row, start_row + 44)   # CC = 45

    # Frequency chart
    dist_chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    dist_chart.set_legend({"none": True})
    dist_chart.add_series(
        {
            "name": "Frequency %",
            "categories": [sheet_name, start_row, 0, x_end_row, 0],
            "values": [sheet_name, start_row, 2, x_end_row, 2],
        }
    )
    dist_chart.set_y_axis({"min": 0, "max": 30, "major_unit": 10})
    _apply_x_axis(dist_chart)
    worksheet.insert_chart("G2", dist_chart)

    # Cumulative chart
    cum_chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    cum_chart.set_legend({"none": True})
    cum_chart.add_series(
        {
            "name": "Cumulative %",
            "categories": [sheet_name, start_row, 0, x_end_row, 0],
            "values": [sheet_name, start_row, 3, x_end_row, 3],
        }
    )
    cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})
    _apply_x_axis(cum_chart)
    worksheet.insert_chart("G20", cum_chart)

    return writer, hist, sheet_name


# ---------- Summary sheet ----------
def generate_summary_sheet(
    writer: ExcelWriter,
    all_data: list[tuple[DataFrame, str]],
) -> None:
    sheet_name = "Summary"
    workbook = writer.book

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

    combined_df = combined_df.fillna(0)

    start_row = 1
    max_chart_row = 80
    end_row = start_row + max_chart_row - 1
    x_end_row = min(end_row, start_row + 44)   # CC = 45

    freq_cols = [c for c in combined_df.columns if c.startswith("Frequency %_")]
    cum_cols = [c for c in combined_df.columns if c.startswith("Cumulative %_")]

    freq_indices = [combined_df.columns.get_loc(c) for c in freq_cols]
    cum_indices = [combined_df.columns.get_loc(c) for c in cum_cols]

    colors = [
        "#4F81BD", "#C0504D", "#9BBB59", "#8064A2",
        "#F79646", "#1F497D", "#2F5597", "#953735",
    ]

    # ----- Per-code combined datasets -----
    codes = ["Nemeth", "UEB", "LaTeX", "ASCIIMath"]
    combined_by_code: dict[str, DataFrame] = {}

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

        combined_by_code[code] = combined

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

    combined_df = combined_df.fillna(0)

    # ----- Write sheet ONCE -----
    combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]

    # ----- Add mean and median rows -----
    last_data_row = start_row + max_chart_row  # row after the last data row

    numeric_cols = [
        col for col in combined_df.columns
        if col != "Character Count"
    ]

    # Write labels
    worksheet.write(last_data_row, 0, "Mean")
    worksheet.write(last_data_row + 1, 0, "Median")

    # Write formulas for each numeric column
    for col in numeric_cols:
        col_idx = combined_df.columns.get_loc(col)
        excel_col_letter = chr(ord('A') + col_idx)  # type: ignore[operator]

        # Excel rows are 1‑based; data starts at row 2
        data_start = 2
        data_end = data_start + max_chart_row - 1

        worksheet.write_formula(
            last_data_row,
            col_idx,
            f"=AVERAGE({excel_col_letter}{data_start}:{excel_col_letter}{data_end})"
        )
        worksheet.write_formula(
            last_data_row + 1,
            col_idx,
            f"=MEDIAN({excel_col_letter}{data_start}:{excel_col_letter}{data_end})"
        )
    # ----- All-datasets frequency chart -----
    dist_series = [
        {
            "name": freq_cols[i].replace("Frequency %_", ""),
            "col_idx": freq_indices[i],
            "color": colors[i % len(colors)],
        }
        for i in range(len(freq_cols))
    ]

    dist_chart = _create_chart_with_series(
        workbook, sheet_name, start_row, x_end_row,
        dist_series, "Distribution of Line Lengths - All Datasets"
    )
    dist_chart.set_y_axis({"min": 0, "max": 30, "major_unit": 10})
    _apply_x_axis(dist_chart)
    dist_chart.set_size({"height": 580})

    # ----- All-datasets cumulative chart -----
    cum_series = [
        {
            "name": cum_cols[i].replace("Cumulative %_", ""),
            "col_idx": cum_indices[i],
            "color": colors[i % len(colors)],
        }
        for i in range(len(cum_cols))
    ]

    cum_chart = _create_chart_with_series(
        workbook, sheet_name, start_row, x_end_row,
        cum_series, "Cumulative Distribution - All Datasets"
    )
    cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})
    _apply_x_axis(cum_chart)
    cum_chart.set_size({"height": 580})

    chart_row_start = 90
    worksheet.insert_chart(f"D{chart_row_start}", dist_chart)
    worksheet.insert_chart(f"L{chart_row_start}", cum_chart)

    # ----- Per-code combined charts -----
    combined_positions = chart_row_start + 40

    for code in codes:
        fcol = f"Frequency %_{code}_Combined"
        ccol = f"Cumulative %_{code}_Combined"
        if fcol not in combined_df.columns or ccol not in combined_df.columns:
            continue

        freq_col = combined_df.columns.get_loc(fcol)
        cum_col = combined_df.columns.get_loc(ccol)

        dist_chart_code = _create_chart_with_series(
            workbook, sheet_name, start_row, x_end_row,
            [{"name": f"{code} Combined", "col_idx": freq_col, "color": "#4F81BD"}],
            f"Distribution - {code} Combined"
        )
        dist_chart_code.set_y_axis({"min": 0, "max": 30, "major_unit": 10})
        _apply_x_axis(dist_chart_code)
        dist_chart_code.set_size({"height": 580})

        cum_chart_code = _create_chart_with_series(
            workbook, sheet_name, start_row, x_end_row,
            [{"name": f"{code} Combined", "col_idx": cum_col, "color": "#4F81BD"}],
            f"Cumulative Distribution - {code} Combined"
        )
        cum_chart_code.set_y_axis({"min": 0, "max": 100, "major_unit": 10})
        _apply_x_axis(cum_chart_code)
        cum_chart_code.set_size({"height": 580})

        worksheet.insert_chart(f"D{combined_positions}", dist_chart_code)
        worksheet.insert_chart(f"L{combined_positions}", cum_chart_code)
        combined_positions += 40

    # ----- Four-line charts -----
    four_freq_series = []
    four_cum_series = []

    for idx, code in enumerate(codes):
        fcol = f"Frequency %_{code}_Combined"
        ccol = f"Cumulative %_{code}_Combined"
        if fcol in combined_df.columns and ccol in combined_df.columns:
            four_freq_series.append(
                {"name": code, "col_idx": combined_df.columns.get_loc(fcol),
                 "color": colors[idx % len(colors)]}
            )
            four_cum_series.append(
                {"name": code, "col_idx": combined_df.columns.get_loc(ccol),
                 "color": colors[idx % len(colors)]}
            )

    if four_freq_series:
        four_freq_chart = _create_chart_with_series(
            workbook, sheet_name, start_row, x_end_row,
            four_freq_series, "Distribution - Combined Highschool+College"
        )
        four_freq_chart.set_y_axis({"min": 0, "max": 30, "major_unit": 10})
        _apply_x_axis(four_freq_chart)
        four_freq_chart.set_size({"height": 580})

        four_cum_chart = _create_chart_with_series(
            workbook, sheet_name, start_row, x_end_row,
            four_cum_series, "Cumulative - Combined Highschool+College"
        )
        four_cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})
        _apply_x_axis(four_cum_chart)
        four_cum_chart.set_size({"height": 580})

        worksheet.insert_chart(f"T{chart_row_start}", four_freq_chart)
        worksheet.insert_chart(f"AB{chart_row_start}", four_cum_chart)

    # ----- Make summary sheet first -----
    workbook.worksheets_objs.insert(0, workbook.worksheets_objs.pop())

    print(f"Successfully added summary sheet '{sheet_name}' to workbook.")


def main():
    # Codes and subdirectories to process
    codes = ["Nemeth", "UEB", "LaTeX", "ASCIIMath"]
    levels = ["highschool", "college"]

    writer: pd.ExcelWriter | None = None
    all_data = []   # (hist_data, sheet_name) tuples

    try:
        for code in codes:
            for level in levels:
                directory = f"Braille/{code}/{level}"
                file_pattern = "*.brls"

                writer, hist_data, sheet_name = generate_line_histogram(
                    directory=directory,
                    file_pattern=file_pattern,
                    writer=writer
                )

                # Only add non-empty sheets
                if not hist_data.empty:
                    all_data.append((hist_data, sheet_name))

        # Create summary sheet with overlay charts
        if all_data and writer is not None:
            generate_summary_sheet(writer, all_data)

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
                return
            except OSError as e:
                print(
                    f"ERROR: Could not write 'braille-lengths.xlsx' ({e}). "
                    "Please close the file if it is open and try again."
                )
                return


# --- Example Usage ---
if __name__ == "__main__":
    main()
