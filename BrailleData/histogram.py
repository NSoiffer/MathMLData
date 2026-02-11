# flake8: noqa=E221,E222,E241,E242
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import re
import math
import pandas as pd
from pandas import DataFrame, ExcelWriter
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element



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
) -> DataFrame:
    """
    Load a single merged <code>.brls file and compute the histogram.
    """
    file_path = Path(base_dir) / f"{code}.brls"

    if not file_path.is_file():
        return pd.DataFrame()

    lengths: list[int] = []
    with open(file_path, "r", encoding="utf-8") as f:
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


def _write_label(worksheet, row, col, label, fmt):
    worksheet.write(row, col, label, fmt)


def _write_headers(worksheet, row, col, columns, fmt):
    for j, colname in enumerate(columns):
        worksheet.write(row, col + j, colname, fmt)


def _write_dataframe_rows(worksheet, df, start_row, start_col, number_fmt, text_fmt):
    for i, (_, row) in enumerate(df.iterrows()):
        excel_row = start_row + i
        for j, col in enumerate(df.columns):
            val = row[col]
            if isinstance(val, float) and math.isnan(val):
                worksheet.write(excel_row, start_col + j, "", number_fmt)
            elif isinstance(val, (int, float)):
                worksheet.write(excel_row, start_col + j, val, number_fmt)
            else:
                worksheet.write(excel_row, start_col + j, val, text_fmt)


def _add_excel_table(worksheet, first_row, first_col, last_row, last_col, label, df):
    worksheet.add_table(
        first_row,
        first_col,
        last_row,
        last_col,
        {
            "name": f"T_{label.replace(' ', '_')}_table",
            "style": "Table Style Medium 2",
            "columns": [{"header": col} for col in df.columns],
        },
    )


def _insert_dataframe_table(
    worksheet,
    workbook,
    df,
    label,
    start_row,
    start_col,
):
    label_fmt = workbook.add_format({"bold": True, "align": "left", "valign": "vcenter"})
    header_fmt = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
    number_fmt = workbook.add_format({"align": "right", "border": 1})
    text_fmt   = workbook.add_format({"align": "left",  "border": 1})

    # Label
    _write_label(worksheet, start_row, start_col, label, label_fmt)

    # Headers
    header_row = start_row + 1
    _write_headers(worksheet, header_row, start_col, df.columns, header_fmt)

    # Data rows
    data_row_start = header_row + 1
    _write_dataframe_rows(worksheet, df, data_row_start, start_col, number_fmt, text_fmt)

    # Table object
    last_row = data_row_start + len(df) - 1
    last_col = start_col + len(df.columns) - 1
    _add_excel_table(worksheet, header_row, start_col, last_row, last_col, label, df)


def _insert_width_fit_table_generic(
    worksheet,
    workbook,
    combined_df,
    codes,
    label,
    start_row,
    start_col,
):
    widths = [14, 18, 20, 32, 40]

    label_fmt = workbook.add_format({"bold": True, "align": "left", "valign": "vcenter"})
    header_left_fmt  = workbook.add_format({"bold": True, "align": "left",  "valign": "vcenter", "border": 1})
    header_right_fmt = workbook.add_format({"bold": True, "align": "right", "valign": "vcenter", "border": 1})
    number_fmt = workbook.add_format({"align": "right", "border": 1})
    text_fmt   = workbook.add_format({"align": "left",  "border": 1})

    worksheet.set_column(start_col, start_col, 18)
    worksheet.set_column(start_col + 1, start_col + len(widths), 6)

    # Label
    _write_label(worksheet, start_row, start_col, label, label_fmt)

    # Headers
    header_row = start_row + 1
    worksheet.write(header_row, start_col, "Code / # Cells", header_left_fmt)
    for j, w in enumerate(widths):
        worksheet.write(header_row, start_col + 1 + j, w, header_right_fmt)

    # Data rows (matrix logic stays here)
    for i, code in enumerate(codes):
        excel_row = header_row + 1 + i
        worksheet.write(excel_row, start_col, code, text_fmt)

        colname = f"Cumulative %_{code}"
        if colname not in combined_df.columns:
            continue

        for j, w in enumerate(widths):
            rows = combined_df.loc[combined_df["Character Count"] == w, colname]
            if rows.empty:
                continue
            worksheet.write(excel_row, start_col + 1 + j, int(round(float(rows.iloc[0]))), number_fmt)


# ---------- Summary sheet orchestrator ----------
def generate_summary_sheet(
    writer: ExcelWriter,
    hist_by_code: dict[str, DataFrame],
    codes: list[str],
    sheet_name: str = "Total",
    total_unfiltered_count: int | None = None,
) -> None:
    workbook = writer.book  # type: ignore[attr-defined]

    combined_df = _build_combined_dataframe(hist_by_code)
    combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]  # type: ignore[attr-defined]

    start_row = 1
    data_rows = len(combined_df)
    x_end_row = min(start_row + 44, start_row + data_rows - 1)

    # ------------------------------------------------------------
    #  Table position (must be computed BEFORE writing percent label)
    # ------------------------------------------------------------
    table_start_row = data_rows + 5
    table_start_col = 21

    # ------------------------------------------------------------
    #  Percentage of data used (correct placement)
    # ------------------------------------------------------------
    if total_unfiltered_count is None:
        percent_used = 100.0
    else:
        used = sum(int(df["Frequency"].sum()) for df in hist_by_code.values())
        percent_used = round(used / total_unfiltered_count * 100, 2)

    # Place label to the right of "Combined"
    percent_label_row = table_start_row
    percent_label_col = table_start_col + 1

    worksheet.write(
        percent_label_row,
        percent_label_col,
        f"{percent_used}%",
    )

    # ------------------------------------------------------------
    #  Insert the table
    # ------------------------------------------------------------
    _insert_width_fit_table_generic(
        worksheet,
        workbook,
        combined_df,
        codes,
        label="Combined",
        start_row=table_start_row,
        start_col=table_start_col,
    )

    # ------------------------------------------------------------
    #  All-codes charts
    # ------------------------------------------------------------
    freq_series, cum_series = _build_all_dataset_series(combined_df)

    dist_chart = _create_scatter_chart(
        workbook, sheet_name, start_row, x_end_row, freq_series, "Distribution"
    )
    dist_chart.set_y_axis({"min": 0, "max": 20, "major_unit": 10})
    worksheet.insert_chart("D90", dist_chart)

    cum_chart = _create_scatter_chart(
        workbook, sheet_name, start_row, x_end_row, cum_series, "Cumulative"
    )
    cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})
    worksheet.insert_chart("L90", cum_chart)

    # ------------------------------------------------------------
    #  Per-code charts
    # ------------------------------------------------------------
    row = 130
    for code in codes:
        freq_s, cum_s = _build_per_code_series(combined_df, code)
        if freq_s is None:
            continue

        dist_chart = _create_scatter_chart(
            workbook, sheet_name, start_row, x_end_row, freq_s, f"{code} Distribution"
        )
        dist_chart.set_y_axis({"min": 0, "max": 20, "major_unit": 10})
        worksheet.insert_chart(f"D{row}", dist_chart)

        cum_chart = _create_scatter_chart(
            workbook, sheet_name, start_row, x_end_row, cum_s,
            f"{code} Cumulative"  # pyright: ignore[reportArgumentType]
        )
        cum_chart.set_y_axis({"min": 0, "max": 100, "major_unit": 10})
        worksheet.insert_chart(f"L{row}", cum_chart)

        row += 40

    print(f"Successfully added summary sheet '{sheet_name}' to workbook.")


def _load_textbook_mmls() -> list[str]:
    """
    Load ../SimpleSpeakData.mmls relative to the program's execution directory.
    Each line is a MathML expression. These lines will later be used to filter
    which lines from the six code files are included in graphs and tables.
    """
    file_path = Path("..") / "SimpleSpeakData-cnclz.mmls"

    if not file_path.is_file():
        print(f"WARNING: SimpleSpeakData-cnclz.mmls not found at {file_path}")
        return []

    lines: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.rstrip("\n"))

    print(f"Loaded {len(lines)} textbook MathML lines from {file_path}")
    return lines


# Allow whitespace between tags and around content
_SIMPLE_MI_MN_PATTERN = re.compile(
    r"""^\s*
        <math\b[^>]*>\s*            # <math ...>
        <(mi|mn)\b[^>]*>\s*         # <mi ...> or <mn ...>
        .*?                         # content
        \s*</\1>\s*                 # </mi> or </mn>
        </math>\s*$                 # </math>
    """,
    re.VERBOSE,
)


def filter_simple_mi_mn(textbook_lines: list[str]) -> list[int]:
    """
    Return line numbers (0-based) of textbook_lines that do NOT match the simple
    MathML patterns <math><mi>xxx</mi></math> or <math><mn>xxx</mn></math>,
    allowing attributes and whitespace.
    """
    selected: list[int] = []

    for idx, line in enumerate(textbook_lines):
        if not _SIMPLE_MI_MN_PATTERN.match(line):
            selected.append(idx)

    return selected


_TWO_D_TAGS = [
    "mfrac", "msqrt", "mroot",
    "msub", "msup", "msubsup",
    "munder", "mover", "munderover",
    "menclose", "mmultiscripts",
]

_TWO_D_PATTERN = re.compile(
    r"<(" + "|".join(_TWO_D_TAGS) + r")\b[^>]*>"
)


def filter_contains_2d(textbook_lines: list[str]) -> list[int]:
    """
    Return line numbers (0-based) of textbook_lines that contain
    any 2-D MathML tag (excluding tables and fenced expressions),
    allowing attributes inside the tag.
    """
    selected: list[int] = []

    for idx, line in enumerate(textbook_lines):
        if _TWO_D_PATTERN.search(line):
            selected.append(idx)

    return selected


def _load_filtered_code_lengths(
    base_dir: str,
    code: str,
    keep_lines: list[int],
) -> list[int]:
    """
    Load <code>.brls and return the lengths of only the lines whose
    line numbers appear in keep_lines.

    For LaTeX, LaTeX-6, ASCIIMath, ASCIIMath-6:
        remove whitespace before computing length.
    """

    file_path = Path(base_dir) / f"{code}.brls"
    if not file_path.is_file():
        return []

    keep_set = set(keep_lines)
    lengths: list[int] = []

    # Codes that require whitespace removal
    ws_codes = {"LaTeX", "LaTeX-6", "ASCIIMath", "ASCIIMath-6"}

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in keep_set:
                continue

            line = line.rstrip("\n")

            if code in ws_codes:
                # Remove whitespace before non-alphanumeric
                cleaned, _ = remove_ws_before_non_alnum(line)
                lengths.append(len(cleaned))
            else:
                lengths.append(len(line))

    return lengths


def _parse_mathml_lines(lines: list[str]) -> list[ET.Element]:
    parsed = []
    for line in lines:
        try:
            parsed.append(ET.fromstring(line))
        except ET.ParseError:
            # Skip malformed MathML
            continue
    return parsed


def analyze_2d_structure(mathml_lines: list[Element]) -> dict[str, Any]:
    two_d_tags = set(_TWO_D_TAGS)

    # Per-line counts: does this line contain at least one mfrac, msqrt, etc.
    line_has_tag = {tag: 0 for tag in two_d_tags}

    # Per-element counts: total number of each tag across all lines
    element_count = {tag: 0 for tag in two_d_tags}

    # Child simplicity counts (per element)
    simple_counts = {tag: [0, 0, 0] for tag in two_d_tags}
    total_counts  = {tag: [0, 0, 0] for tag in two_d_tags}

    # Extra stats for mfrac, msub/msubsup, msup/msubsup
    mfrac_num_int = 0
    mfrac_den_int = 0
    mfrac_both_int = 0

    sub_int = 0
    sup_2 = 0
    sup_3 = 0
    sup_mo = 0
    sup_mo_ops = {}

    for root in mathml_lines:
        # Track which tags appear in this line
        seen_this_line = set()

        for elem in root.iter():
            tag = elem.tag.split("}")[-1]
            if tag not in two_d_tags:
                continue

            # Per-element count
            element_count[tag] += 1
            seen_this_line.add(tag)

            # Child simplicity (direct children only)
            children = list(elem)
            for i, child in enumerate(children[:3]):
                ctag = child.tag.split("}")[-1]
                if ctag in {"mn", "mi", "mo", "mtext"}:
                    simple_counts[tag][i] += 1
                total_counts[tag][i] += 1

            # Extra stats for mfrac
            if tag == "mfrac":
                num = children[0] if len(children) > 0 else None
                den = children[1] if len(children) > 1 else None

                num_is_int = num is not None and num.tag.split("}")[-1] == "mn"
                den_is_int = den is not None and den.tag.split("}")[-1] == "mn"

                if num_is_int:
                    mfrac_num_int += 1
                if den_is_int:
                    mfrac_den_int += 1
                if num_is_int and den_is_int:
                    mfrac_both_int += 1

            # Extra stats for msub/msubsup (subscript integer)
            if tag in {"msub", "msubsup"}:
                if len(children) > 1:
                    sub = children[1]
                    if sub.tag.split("}")[-1] == "mn":
                        sub_int += 1

            # Extra stats for msup/msubsup (superscript)
            if tag in {"msup", "msubsup"}:
                if len(children) > 1:
                    sup = children[-1]
                    stag = sup.tag.split("}")[-1]
                    if stag == "mn":
                        if sup.text == "2":
                            sup_2 += 1
                        elif sup.text == "3":
                            sup_3 += 1
                    elif stag == "mo":
                        sup_mo += 1
                        op = sup.text or ""
                        sup_mo_ops[op] = sup_mo_ops.get(op, 0) + 1

        # Per-line counts
        for tag in seen_this_line:
            line_has_tag[tag] += 1

    total_lines = len(mathml_lines)
    total_2d_elements = sum(element_count.values())

    return {
        "line_has_tag": line_has_tag,
        "element_count": element_count,
        "simple_counts": simple_counts,
        "total_counts": total_counts,
        "total_lines": total_lines,
        "total_2d_elements": total_2d_elements,
        "mfrac_num_int": mfrac_num_int,
        "mfrac_den_int": mfrac_den_int,
        "mfrac_both_int": mfrac_both_int,
        "sub_int": sub_int,
        "sup_2": sup_2,
        "sup_3": sup_3,
        "sup_mo": sup_mo,
        "sup_mo_ops": sup_mo_ops,
    }


def _build_2d_simplicity_dataframe(
    line_has_tag: dict[str, int],
    element_count: dict[str, int],
    simple_counts: dict[str, list[int]],
    total_counts: dict[str, list[int]],
    total_lines: int,
    total_2d_elements: int,
) -> DataFrame:

    rows = []

    for tag in _TWO_D_TAGS:
        lines_with_tag = line_has_tag[tag]
        elems_of_tag   = element_count[tag]

        pct_lines = (
            None if total_lines == 0
            else round(100.0 * lines_with_tag / total_lines, 1)
        )

        pct_of_2d = (
            None if total_2d_elements == 0
            else round(100.0 * elems_of_tag / total_2d_elements, 1)
        )

        # Child simplicity percentages
        child_pcts = []
        for i in range(3):
            denom = total_counts[tag][i]
            num   = simple_counts[tag][i]

            if denom == 0:
                child_pcts.append("")   # <-- leave blank
            else:
                child_pcts.append(round(100.0 * num / denom, 1))

        rows.append({
            "2D Tag": tag,
            "% Lines With Tag": pct_lines,
            "% Of All 2D Elements": pct_of_2d,
            "Child1 Simple %": child_pcts[0],
            "Child2 Simple %": child_pcts[1],
            "Child3 Simple %": child_pcts[2],
        })

    return DataFrame(rows)


def _histogram_from_lengths(lengths: list[int]) -> DataFrame:
    """
    Build a histogram DataFrame from a list of line lengths.
    """
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


def _apply_filter_and_write_sheet(
    writer: ExcelWriter,
    base_dir: str,
    codes: list[str],
    keep_lines: list[int],
    sheet_name: str,
    total_unfiltered_count: int,
) -> None:
    filtered_hist_by_code: dict[str, DataFrame] = {}

    for code in codes:
        lengths = _load_filtered_code_lengths(base_dir, code, keep_lines)
        hist = _histogram_from_lengths(lengths)
        if not hist.empty:
            filtered_hist_by_code[code] = hist

    if filtered_hist_by_code:
        generate_summary_sheet(
            writer,
            filtered_hist_by_code,
            codes,
            sheet_name=sheet_name,
            total_unfiltered_count=total_unfiltered_count,
        )


def _open_excel_writer(path: str) -> ExcelWriter:
    try:
        return ExcelWriter(path, engine="xlsxwriter")
    except PermissionError:
        print(
            f"ERROR: Could not open '{path}' for writing.\n"
            "The file is probably open in Excel. "
            "Please close it and run the program again."
        )
        sys.exit(1)


def _close_excel_writer(writer: ExcelWriter) -> None:
    try:
        writer.close()
        print("Summary sheet written to 'braille-lengths.xlsx'.")
    except PermissionError:
        print(
            "ERROR: Could not write 'braille-lengths.xlsx'. "
            "The file is probably open in Excel. "
            "Please close the file and try again."
        )
    except OSError as e:
        print(
            f"ERROR: Could not write 'braille-lengths.xlsx' ({e}). "
            "Please close the file if it is open and try again."
        )


def _get_codes() -> list[str]:
    return [
        "Nemeth",
        "UEB",
        "LaTeX",
        "LaTeX-6",
        "ASCIIMath",
        "ASCIIMath-6",
    ]


def _load_unfiltered_histograms(base_dir: str, codes: list[str]) -> tuple[dict[str, DataFrame], int]:
    hist_by_code: dict[str, DataFrame] = {}
    for code in codes:
        hist = _compute_histogram_for_code(base_dir, code)
        if not hist.empty:
            hist_by_code[code] = hist

    total_unfiltered_count = sum(
        int(df["Frequency"].sum()) for df in hist_by_code.values()
    )
    return hist_by_code, total_unfiltered_count


def _write_total_sheet(
    writer: ExcelWriter,
    hist_by_code: dict[str, DataFrame],
    codes: list[str],
    total_unfiltered_count: int,
) -> None:
    generate_summary_sheet(
        writer,
        hist_by_code,
        codes,
        sheet_name="Total",
        total_unfiltered_count=total_unfiltered_count,
    )


def _write_2d_structure_analysis(
    writer: ExcelWriter,
    mathml_lines: list[Element],
) -> None:

    worksheet = writer.sheets["Total"]
    workbook  = writer.book

    result = analyze_2d_structure(mathml_lines)

    df_2d = _build_2d_simplicity_dataframe(
        result["line_has_tag"],
        result["element_count"],
        result["simple_counts"],
        result["total_counts"],
        result["total_lines"],
        result["total_2d_elements"],
    )

    start_row = 2
    start_col = 16
    worksheet.set_column(start_col, start_col, 42)

    _insert_dataframe_table(
        worksheet,
        workbook,
        df_2d,
        label="2D Structure Summary",
        start_row=start_row,
        start_col=start_col,
    )

    # ------------------------------------------------------------
    # Extra stats below the table
    # ------------------------------------------------------------
    summary_row = start_row + len(df_2d) + 4
    row = summary_row

    # -------------------------
    # mfrac stats
    # -------------------------
    mfrac_total = result["element_count"]["mfrac"]

    if mfrac_total > 0:
        worksheet.write(row, start_col, "mfrac numerator integer %:")
        worksheet.write(row, start_col + 1,
                        round(100.0 * result["mfrac_num_int"] / mfrac_total, 1))
        row += 1

        worksheet.write(row, start_col, "mfrac denominator integer %:")
        worksheet.write(row, start_col + 1,
                        round(100.0 * result["mfrac_den_int"] / mfrac_total, 1))
        row += 1

        worksheet.write(row, start_col, "mfrac both integer %:")
        worksheet.write(row, start_col + 1,
                        round(100.0 * result["mfrac_both_int"] / mfrac_total, 1))
        row += 2  # blank line

    # -------------------------
    # msub / msubsup stats
    # -------------------------
    sub_total = (
        result["element_count"]["msub"] +
        result["element_count"]["msubsup"]
    )

    if sub_total > 0:
        worksheet.write(row, start_col, "msub/msubsup subscript integer %:")
        worksheet.write(row, start_col + 1,
                        round(100.0 * result["sub_int"] / sub_total, 1))
        row += 2  # blank line

    # -------------------------
    # msup / msubsup stats
    # -------------------------
    sup_total = (
        result["element_count"]["msup"] +
        result["element_count"]["msubsup"]
    )

    if sup_total > 0:
        worksheet.write(row, start_col, "msup/msubsup superscript 2 %:")
        worksheet.write(row, start_col + 1,
                        round(100.0 * result["sup_2"] / sup_total, 1))
        row += 1

        worksheet.write(row, start_col, "msup/msubsup superscript 3 %:")
        worksheet.write(row, start_col + 1,
                        round(100.0 * result["sup_3"] / sup_total, 1))
        row += 1

        worksheet.write(row, start_col, "msup/msubsup superscript <mo> %:")
        worksheet.write(row, start_col + 1,
                        round(100.0 * result["sup_mo"] / sup_total, 1))
        row += 2  # blank line

        # Operator frequency table
        worksheet.write(row, start_col, "Superscript <mo> operators:")
        row += 1

        sup_mo_ops = result["sup_mo_ops"]
        total_ops = sum(sup_mo_ops.values())

        for op, count in sorted(sup_mo_ops.items(), key=lambda x: -x[1]):
            pct = round(100.0 * count / total_ops, 1)
            worksheet.write(row, start_col, op)
            worksheet.write(row, start_col + 1, f"{pct}%")
            row += 1


def _write_filtered_sheet_simple(
    writer: ExcelWriter,
    base_dir: str,
    codes: list[str],
    textbook_lines: list[str],
    total_unfiltered_count: int,
    braille_by_code: dict[str, list[str]],
) -> tuple[int, set[int]]:

    simple_indices_list = filter_simple_mi_mn(textbook_lines)
    simple_indices_set = set(simple_indices_list)

    def simple_length_fn(code: str, line: str) -> int:
        return len(line)

    hist_by_code = _build_hist_by_code(
        braille_by_code,
        codes,
        simple_indices_list,
        simple_length_fn,
    )

    generate_summary_sheet(
        writer,
        hist_by_code,
        codes,
        sheet_name="simple",
        total_unfiltered_count=total_unfiltered_count,
    )

    return len(simple_indices_list), simple_indices_set


def _write_filtered_sheet_2d(
    writer: ExcelWriter,
    base_dir: str,
    codes: list[str],
    textbook_lines: list[str],
    total_unfiltered_count: int,
) -> None:
    two_d_indices = filter_contains_2d(textbook_lines)
    _apply_filter_and_write_sheet(
        writer,
        base_dir,
        codes,
        two_d_indices,
        sheet_name="2D exprs",
        total_unfiltered_count=total_unfiltered_count,
    )


NON_ALNUM = r"[^A-Za-z0-9]"

def remove_ws_before_non_alnum(expr: str) -> tuple[str, int]:
    """
    Remove whitespace that appears immediately before a non-alphanumeric character.
    Returns (cleaned_expr, num_deletions).
    """
    # Pattern: whitespace followed by a non-alphanumeric
    pattern = re.compile(r"\s+(?=" + NON_ALNUM + ")")

    # Count how many matches
    deletions = len(pattern.findall(expr))

    # Remove them
    cleaned = pattern.sub("", expr)

    return cleaned, deletions


def _build_hist_by_code(
    braille_by_code: dict[str, list[str]],
    codes: list[str],
    keep_lines: list[int],
    length_fn,
) -> dict[str, DataFrame]:

    keep_set = set(keep_lines)
    hist_by_code = {}

    for code in codes:
        lines = braille_by_code.get(code, [])
        lengths = []

        for idx, line in enumerate(lines):
            if idx in keep_set:
                lengths.append(length_fn(code, line))

        hist = _histogram_from_lengths(lengths)
        if not hist.empty:
            hist_by_code[code] = hist

    return hist_by_code


def _write_filtered_sheet_no_whitespace(
    writer: ExcelWriter,
    base_dir: str,
    codes: list[str],
    textbook_lines: list[str],
    simple_indices: set[int],
    total_unfiltered_count: int,
    braille_by_code: dict[str, list[str]],
) -> None:

    keep_lines = sorted(simple_indices)

    ws_codes = {"LaTeX", "LaTeX-6", "ASCIIMath", "ASCIIMath-6"}

    def no_ws_length_fn(code: str, line: str) -> int:
        if code in ws_codes:
            cleaned, _ = remove_ws_before_non_alnum(line)
            return len(cleaned)
        return len(line)

    hist_by_code = _build_hist_by_code(
        braille_by_code,
        codes,
        keep_lines,
        no_ws_length_fn,
    )

    generate_summary_sheet(
        writer,
        hist_by_code,
        codes,
        sheet_name="no whitespace",
        total_unfiltered_count=total_unfiltered_count,
    )


def _load_braille_by_code(base_dir: str, codes: list[str]) -> dict[str, list[str]]:
    braille_by_code = {}
    for code in codes:
        path = Path(base_dir) / f"{code}.brls"
        if not path.is_file():
            braille_by_code[code] = []
            continue
        with open(path, "r", encoding="utf-8") as f:
            braille_by_code[code] = [line.rstrip("\n") for line in f]
    return braille_by_code


def main() -> None:
    writer = _open_excel_writer("braille-lengths.xlsx")

    try:
        # Load textbook MathML lines (used only for filtering)
        textbook_lines = _load_textbook_mmls()

        # Load codes and base directory
        codes = _get_codes()
        base_dir = "Braille"

        # Load all braille once (no repeated file reads)
        braille_by_code = _load_braille_by_code(base_dir, codes)

        # Load unfiltered histograms for the TOTAL sheet
        hist_by_code, total_unfiltered_count = _load_unfiltered_histograms(
            base_dir, codes
        )

        # Write TOTAL sheet
        _write_total_sheet(
            writer,
            hist_by_code,
            codes,
            total_unfiltered_count,
        )

        # SIMPLE sheet (raw lengths)
        simple_expr_count, simple_indices = _write_filtered_sheet_simple(
            writer,
            base_dir,
            codes,
            textbook_lines,
            total_unfiltered_count,
            braille_by_code,     # <-- NEW: in-memory braille
        )

        # NO WHITESPACE sheet (adjusted lengths for LaTeX/ASCIIMath)
        _write_filtered_sheet_no_whitespace(
            writer,
            base_dir,
            codes,
            textbook_lines,
            simple_indices,       # <-- SAME indices as simple sheet
            total_unfiltered_count,
            braille_by_code,      # <-- same in-memory braille
        )

        # 2D structure analysis (unchanged)
        mathml_lines = _parse_mathml_lines(textbook_lines)
        _write_2d_structure_analysis(writer, mathml_lines)

        # 2D expressions filter (unchanged)
        _write_filtered_sheet_2d(
            writer,
            base_dir,
            codes,
            textbook_lines,
            total_unfiltered_count,
        )

    finally:
        _close_excel_writer(writer)


if __name__ == "__main__":
    main()
