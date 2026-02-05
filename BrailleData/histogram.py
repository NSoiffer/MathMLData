import pandas as pd
import glob
import os
import statistics
from typing import List, NamedTuple
import sys
sys.stdout.reconfigure(encoding='utf-8')  # in case print statements are used for debugging


class Statistics(NamedTuple):
    """Statistics calculated from a list of line lengths."""
    total_lines: int
    total_chars: int
    mean_len: float
    median_len: float
    max_len: int


def read_line_lengths(directory: str, file_pattern: str = "*.txt") -> List[int]:
    """
    Read files from a directory and calculate line lengths.

    Args:
        directory: Directory path to search for files
        file_pattern: File pattern to match (e.g., "*.txt", "*.brls")

    Returns:
        List of line lengths from all matching files. Returns empty list if
        no files found or no valid data.
    """
    all_lengths: List[int] = []

    search_path = os.path.join(directory, file_pattern)
    files = glob.glob(search_path)

    if not files:
        print(f"No files found matching pattern: {search_path}")
        return []

    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                # Count characters per line, excluding trailing whitespace
                lengths = [len(line.strip()) for line in f if len(line.strip()) > 2]
                all_lengths.extend(lengths)
        except Exception as e:
            print(f"Could not read file {file}: {e}")

    return all_lengths


def calculate_statistics(lengths: List[int]) -> Statistics:
    """
    Calculate statistics from a list of line lengths.

    Args:
        lengths: List of line lengths

    Returns:
        Statistics named tuple containing total_lines, total_chars, mean_len,
        median_len, and max_len
    """
    return Statistics(
        total_lines=len(lengths),
        total_chars=sum(lengths),
        mean_len=statistics.mean(lengths),
        median_len=statistics.median(lengths),
        max_len=max(lengths)
    )


def _combine_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame | None:
    """
    Combine multiple datasets by summing frequencies and recalculating percentages.

    Args:
        datasets: List of hist_data DataFrames to combine

    Returns:
        Combined DataFrame with Frequency % and Cumulative %, or None if empty
    """
    if not datasets:
        return None

    combined = pd.DataFrame({'Character Count': range(1, 81)})
    combined['Frequency'] = 0
    total_lines = 0

    for hist_data in datasets:
        total_lines += hist_data['Frequency'].sum()
        temp_df = hist_data[['Character Count', 'Frequency']].copy()
        combined = combined.merge(
            temp_df,
            on='Character Count',
            how='left',
            suffixes=('', '_new')
        )
        combined['Frequency'] = combined['Frequency'].fillna(0) + combined['Frequency_new'].fillna(0)
        combined = combined.drop(columns=['Frequency_new'])

    combined = combined.fillna(0)
    combined['Frequency'] = combined['Frequency'].astype(int)
    combined['Cumulative Count'] = combined['Frequency'].cumsum()
    combined['Frequency %'] = (combined['Frequency'] / total_lines * 100).round(2)
    combined['Cumulative %'] = (combined['Cumulative Count'] / total_lines * 100).round(2)

    return combined


def _create_chart_with_series(
    workbook,
    sheet_name: str,
    start_row: int,
    end_row: int,
    series_configs: list[dict],
    title: str,
    chart_type: str = 'line'
) -> object:
    """
    Create a chart with multiple series and standard configuration.

    Args:
        workbook: xlsxwriter workbook object
        sheet_name: Name of the sheet containing data
        start_row: Starting row for data (0-based)
        end_row: Ending row for data (0-based)
        series_configs: List of dicts with 'name', 'col_idx', 'color' keys
        title: Chart title
        chart_type: Type of chart ('line' or 'column')

    Returns:
        Chart object
    """
    chart = workbook.add_chart({'type': chart_type})

    for config in series_configs:
        chart.add_series({
            'name': config['name'],
            'categories': [sheet_name, start_row, 0, end_row, 0],
            'values': [sheet_name, start_row, config['col_idx'], end_row, config['col_idx']],
            'line': {'color': config['color'], 'width': 2}
        })

    chart.set_title({'name': title})
    chart.set_x_axis({
        'name': 'Number of Characters (UTF-8)',
        'max': 80,
        'interval_unit': 5,
        'interval_tick': 5,
        'major_gridlines': {'visible': True}
    })
    chart.set_y_axis({'name': 'Percentage (%)'})
    chart.set_legend({'position': 'bottom'})

    return chart


def generate_summary_sheet(
    writer: pd.ExcelWriter,
    all_data: list[tuple[pd.DataFrame, str]]
) -> None:
    """
    Create a summary sheet with overlay charts showing all datasets.

    Args:
        writer: ExcelWriter object
        all_data: List of tuples containing (hist_data DataFrame, sheet_name)
    """
    sheet_name = "Summary"
    workbook = writer.book

    # Create a combined DataFrame with Character Count and all Frequency % columns
    # Start with Character Count column
    combined_df = pd.DataFrame({'Character Count': range(1, 81)})

    # Add Frequency % and Cumulative % columns for each dataset
    for hist_data, name in all_data:
        # Extract the columns we need and rename them to include dataset name
        data_subset = hist_data[['Character Count', 'Frequency %', 'Cumulative %']].copy()
        data_subset = data_subset.rename(columns={
            'Frequency %': f'Frequency %_{name}',
            'Cumulative %': f'Cumulative %_{name}'
        })

        # Merge to ensure we have all character counts
        combined_df = combined_df.merge(
            data_subset,
            on='Character Count',
            how='left'
        )

    # Fill NaN values with 0
    combined_df = combined_df.fillna(0)

    # Write to summary sheet
    combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

    worksheet = writer.sheets[sheet_name]

    # Filter to 1-80 for charts
    chart_data = combined_df[(combined_df['Character Count'] >= 1) &
                              (combined_df['Character Count'] <= 80)].copy()
    max_chart_row = len(chart_data)

    # Row references: header at row 0, data starts at row 1
    start_row = 1
    end_row = start_row + max_chart_row - 1 if max_chart_row > 0 else start_row

    # Find column indices for Frequency % and Cumulative % columns
    freq_col_names = [col for col in combined_df.columns if col.startswith('Frequency %_')]
    cum_col_names = [col for col in combined_df.columns if col.startswith('Cumulative %_')]

    # Get column indices
    freq_col_indices = [combined_df.columns.get_loc(col) for col in freq_col_names]
    cum_col_indices = [combined_df.columns.get_loc(col) for col in cum_col_names]

    # Create distribution overlay chart
    colors = ['#4F81BD', '#C0504D', '#9BBB59', '#8064A2', '#F79646', '#1F497D']
    dist_series_configs = [
        {'name': name, 'col_idx': freq_col_indices[idx], 'color': colors[idx % len(colors)]}
        for idx, (hist_data, name) in enumerate(all_data)
        if idx < len(freq_col_indices)
    ]
    dist_chart = _create_chart_with_series(
        workbook, sheet_name, start_row, end_row, dist_series_configs,
        'Distribution of Line Lengths - All Datasets'
    )

    # Create cumulative distribution overlay chart
    cum_series_configs = [
        {'name': name, 'col_idx': cum_col_indices[idx], 'color': colors[idx % len(colors)]}
        for idx, (hist_data, name) in enumerate(all_data)
        if idx < len(cum_col_indices)
    ]
    cum_chart = _create_chart_with_series(
        workbook, sheet_name, start_row, end_row, cum_series_configs,
        'Cumulative Distribution of Line Lengths - All Datasets'
    )

    # Place charts side by side
    chart_row_start = max_chart_row + 3
    dist_chart_position = f'D{chart_row_start}'
    cum_chart_position = f'L{chart_row_start}'

    worksheet.insert_chart(dist_chart_position, dist_chart)
    worksheet.insert_chart(cum_chart_position, cum_chart)

    # Create combined datasets: Nemeth (highschool + college) and UEB (highschool + college)
    nemeth_data = [hist_data for hist_data, name in all_data if 'Nemeth' in name]
    ueb_data = [hist_data for hist_data, name in all_data if 'UEB' in name]

    combined_nemeth = _combine_datasets(nemeth_data)
    combined_ueb = _combine_datasets(ueb_data)

    # Add combined data columns to the summary sheet
    if combined_nemeth is not None:
        combined_df = combined_df.merge(
            combined_nemeth[['Character Count', 'Frequency %', 'Cumulative %']].rename(columns={
                'Frequency %': 'Frequency %_Nemeth_Combined',
                'Cumulative %': 'Cumulative %_Nemeth_Combined'
            }),
            on='Character Count',
            how='left'
        )

    if combined_ueb is not None:
        combined_df = combined_df.merge(
            combined_ueb[['Character Count', 'Frequency %', 'Cumulative %']].rename(columns={
                'Frequency %': 'Frequency %_UEB_Combined',
                'Cumulative %': 'Cumulative %_UEB_Combined'
            }),
            on='Character Count',
            how='left'
        )

    # Rewrite the sheet with combined data
    combined_df = combined_df.fillna(0)
    combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]

    # Create combined distribution chart (Nemeth vs UEB)
    if combined_nemeth is not None and combined_ueb is not None:
        nemeth_freq_col = combined_df.columns.get_loc('Frequency %_Nemeth_Combined')
        ueb_freq_col = combined_df.columns.get_loc('Frequency %_UEB_Combined')
        nemeth_cum_col = combined_df.columns.get_loc('Cumulative %_Nemeth_Combined')
        ueb_cum_col = combined_df.columns.get_loc('Cumulative %_UEB_Combined')

        combined_dist_series = [
            {'name': 'Nemeth Combined', 'col_idx': nemeth_freq_col, 'color': '#4F81BD'},
            {'name': 'UEB Combined', 'col_idx': ueb_freq_col, 'color': '#C0504D'}
        ]
        combined_dist_chart = _create_chart_with_series(
            workbook, sheet_name, start_row, end_row, combined_dist_series,
            'Distribution of Line Lengths - Combined Datasets'
        )

        combined_cum_series = [
            {'name': 'Nemeth Combined', 'col_idx': nemeth_cum_col, 'color': '#4F81BD'},
            {'name': 'UEB Combined', 'col_idx': ueb_cum_col, 'color': '#C0504D'}
        ]
        combined_cum_chart = _create_chart_with_series(
            workbook, sheet_name, start_row, end_row, combined_cum_series,
            'Cumulative Distribution of Line Lengths - Combined Datasets'
        )

        # Place combined charts below the original charts
        combined_chart_row_start = chart_row_start + 20  # Place below original charts
        combined_dist_chart_position = f'D{combined_chart_row_start}'
        combined_cum_chart_position = f'L{combined_chart_row_start}'

        worksheet.insert_chart(combined_dist_chart_position, combined_dist_chart)
        worksheet.insert_chart(combined_cum_chart_position, combined_cum_chart)

    # Create highschool-only comparison charts (Nemeth highschool vs UEB highschool)
    nemeth_hs_name = next((name for hist_data, name in all_data if 'Nemeth' in name and 'highschool' in name), None)
    ueb_hs_name = next((name for hist_data, name in all_data if 'UEB' in name and 'highschool' in name), None)

    if nemeth_hs_name and ueb_hs_name:
        # Find column indices for highschool data
        col_mapping = {
            'Frequency %_Nemeth_highschool': 'nemeth_freq',
            'Cumulative %_Nemeth_highschool': 'nemeth_cum',
            'Frequency %_UEB_highschool': 'ueb_freq',
            'Cumulative %_UEB_highschool': 'ueb_cum'
        }
        hs_cols = {}
        for col_name, key in col_mapping.items():
            if col_name in combined_df.columns:
                hs_cols[key] = combined_df.columns.get_loc(col_name)

        if 'nemeth_freq' in hs_cols and 'ueb_freq' in hs_cols:
            hs_dist_series = [
                {'name': 'Nemeth Highschool', 'col_idx': hs_cols['nemeth_freq'], 'color': '#4F81BD'},
                {'name': 'UEB Highschool', 'col_idx': hs_cols['ueb_freq'], 'color': '#C0504D'}
            ]
            hs_dist_chart = _create_chart_with_series(
                workbook, sheet_name, start_row, end_row, hs_dist_series,
                'Distribution of Line Lengths - Highschool Only'
            )

            if 'nemeth_cum' in hs_cols and 'ueb_cum' in hs_cols:
                hs_cum_series = [
                    {'name': 'Nemeth Highschool', 'col_idx': hs_cols['nemeth_cum'], 'color': '#4F81BD'},
                    {'name': 'UEB Highschool', 'col_idx': hs_cols['ueb_cum'], 'color': '#C0504D'}
                ]
                hs_cum_chart = _create_chart_with_series(
                    workbook, sheet_name, start_row, end_row, hs_cum_series,
                    'Cumulative Distribution of Line Lengths - Highschool Only'
                )
            else:
                hs_cum_chart = None
        else:
            hs_dist_chart = None
            hs_cum_chart = None
    else:
        hs_dist_chart = None
        hs_cum_chart = None

    # Place highschool charts below combined charts
    if hs_dist_chart is not None and hs_cum_chart is not None:
        hs_chart_row_start = combined_chart_row_start + 20 if combined_nemeth is not None and combined_ueb is not None else chart_row_start + 20
        hs_dist_chart_position = f'D{hs_chart_row_start}'
        hs_cum_chart_position = f'L{hs_chart_row_start}'

        worksheet.insert_chart(hs_dist_chart_position, hs_dist_chart)
        worksheet.insert_chart(hs_cum_chart_position, hs_cum_chart)

    print(f"Successfully added summary sheet '{sheet_name}' to workbook.")


def generate_line_histogram(
    directory: str,
    file_pattern: str = "*.brls",
    writer: pd.ExcelWriter | None = None,
    sheet_name: str | None = None
) -> tuple[pd.ExcelWriter, pd.DataFrame, str]:
    """
    Reads multiple UTF-8 files, prints summary statistics, and exports a
    histogram to an Excel file.

    Args:
        directory: Directory path to search for files
        file_pattern: File pattern to match (e.g., "*.txt", "*.brls")
        writer: Optional ExcelWriter object. If None, creates a new one.
        sheet_name: Optional sheet name. If None, generates from directory.

    Returns:
        Tuple of (ExcelWriter object, hist_data DataFrame, sheet_name string)
    """
    # Generate sheet name from directory if not provided
    if sheet_name is None:
        # Handle both forward and backslash separators
        normalized = directory.replace('/', os.sep).replace('\\', os.sep)
        path_parts = [p for p in normalized.split(os.sep) if p]
        if len(path_parts) >= 2:
            # Use last 2 parts for sheet name (e.g., "Nemeth_highschool")
            sheet_name = '_'.join(path_parts[-2:])
        elif len(path_parts) == 1:
            sheet_name = path_parts[0]
        else:
            sheet_name = "Data"

    # Ensure sheet name is valid (Excel sheet names have restrictions)
    # Replace invalid characters and limit length
    sheet_name = sheet_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    sheet_name = sheet_name.replace('?', '_').replace('*', '_').replace('[', '_').replace(']', '_')
    if len(sheet_name) > 31:  # Excel sheet name limit
        sheet_name = sheet_name[:31]

    # Create writer if not provided
    create_writer = writer is None
    if create_writer:
        output_name = "braille-lengths.xlsx"
        writer = pd.ExcelWriter(output_name, engine='xlsxwriter')

    all_lengths = read_line_lengths(directory, file_pattern)

    if not all_lengths:
        print(f"No data found in the files for '{directory}'.")
        return writer, pd.DataFrame(), sheet_name

    # --- Calculation of Statistics ---
    stats = calculate_statistics(all_lengths)

    print("-" * 30)
    print(f"Summary Statistics for '{directory}':")
    print(f"Total Lines:      {stats.total_lines}")
    print(f"Total Characters: {stats.total_chars}")
    print(f"Mean Length:      {stats.mean_len:.2f}")
    print(f"Median Length:    {stats.median_len}")
    print(f"Max Length:    {stats.max_len}")
    print("-" * 30)

    # Extract subdirectory names for title
    # Handle both forward and backslash separators
    normalized = directory.replace('/', os.sep).replace('\\', os.sep)
    path_parts = [p for p in normalized.split(os.sep) if p]
    # Get the last 2-3 meaningful parts (skip common prefixes like "Braille")
    if len(path_parts) >= 2:
        # Take the last 2 parts (e.g., "Nemeth", "highschool")
        subdirs = ' '.join(path_parts[-2:])
        title_suffix = f" for {subdirs} textbooks"
    elif len(path_parts) == 1:
        title_suffix = f" for {path_parts[0]}"
    else:
        title_suffix = ""

    # --- Excel Generation ---
    df = pd.DataFrame(all_lengths, columns=['LineLength'])
    hist_data = df['LineLength'].value_counts().sort_index().reset_index()
    hist_data.columns = ['Character Count', 'Frequency']

    # Ensure we have entries for all values from 1 to 80 (fill missing with 0 frequency)
    # This ensures consistent category positions for label calculations
    all_counts = pd.DataFrame({'Character Count': range(1, 81)})
    hist_data = all_counts.merge(hist_data, on='Character Count', how='left').fillna(0)
    hist_data['Frequency'] = hist_data['Frequency'].astype(int)

    # Recalculate cumulative distribution after filling missing values
    hist_data['Cumulative Count'] = hist_data['Frequency'].cumsum()

    # Calculate percentages (use original total_lines, not affected by filled zeros)
    total_lines = stats.total_lines
    hist_data['Frequency %'] = (hist_data['Frequency'] / total_lines * 100).round(2)
    hist_data['Cumulative %'] = (hist_data['Cumulative Count'] / total_lines * 100).round(2)

    # Create a column with explicit labels: show only 5, 10, 15, ..., 80
    # This column will be used for x-axis category labels
    hist_data['X-Axis Label'] = hist_data['Character Count'].apply(
        lambda x: x if x >= 5 and x % 5 == 0 else ''
    )

    # Filter data to only include character counts from 1 to 80 for charts
    # Data is sorted by Character Count, so rows from 1 to 80 are contiguous
    hist_data_filtered = hist_data[
        (hist_data['Character Count'] >= 1) &
        (hist_data['Character Count'] <= 80)
    ].copy()
    max_chart_row = len(hist_data_filtered)

    # Create statistics DataFrame to write to sheet
    stats_df = pd.DataFrame({
        'Statistic': ['Total Lines', 'Total Characters', 'Mean Length', 'Median Length', 'Max Length'],
        'Value': [
            stats.total_lines,
            stats.total_chars,
            f'{stats.mean_len:.2f}',
            stats.median_len,
            stats.max_len
        ]
    })

    # Write statistics and data to the sheet
    # Statistics start at row 0 (header at row 0, data at rows 1-5)
    stats_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

    # Write histogram data starting a few rows after statistics
    # stats_df has 5 data rows, written with startrow=0:
    #   Row 0: Header
    #   Rows 1-5: Data (5 rows)
    #   Total: 6 rows used (rows 0-5)
    # Add 1 blank row, so histogram starts at row 7
    data_start_row = len(stats_df) + 2  # 5 data rows + 1 header row + 1 blank row = row 7
    hist_data.to_excel(writer, sheet_name=sheet_name, index=False, startrow=data_start_row)

    # Ensure the workbook and worksheet are accessible
    workbook = writer.book
    if sheet_name not in writer.sheets:
        # This shouldn't happen, but handle it just in case
        raise ValueError(f"Sheet '{sheet_name}' was not created successfully")
    worksheet = writer.sheets[sheet_name]

    # Adjust chart data row references to account for statistics section
    # Row 0 is stats header, row 1 is stats data, row 2 is blank, row 3 is hist_data header
    # So hist_data starts at row data_start_row, header is at data_start_row, data starts at data_start_row + 1
    start_row = data_start_row + 1  # Data starts after the histogram header
    end_row = start_row + max_chart_row - 1 if max_chart_row > 0 else start_row

    # Create line chart for distribution
    chart = workbook.add_chart({'type': 'line'})
    if max_chart_row > 0:
        chart.add_series({
            'name':       'Line Length Frequency',
            'categories': [sheet_name, start_row, 0, end_row, 0],  # Use Character Count column
            'values':     [sheet_name, start_row, 3, end_row, 3],  # Use Frequency % column
            'line':       {'color': '#4F81BD', 'width': 2}
        })

    chart.set_title({'name': f'Distribution of Line Lengths{title_suffix}'})
    chart.set_x_axis({
        'name': 'Number of Characters (UTF-8)',
        'max': 80,
        'interval_unit': 5,  # Show labels and gridlines every 5 units
        'interval_tick': 5,  # Start at position 5 (1-based), which is value 5
        'major_gridlines': {'visible': True}
    })
    chart.set_y_axis({'name': 'Percentage (%)'})
    chart.set_legend({'position': 'none'})

    # Create cumulative distribution chart
    cum_chart = workbook.add_chart({'type': 'line'})
    if max_chart_row > 0:
        cum_chart.add_series({
            'name':       'Cumulative Distribution',
            'categories': [sheet_name, start_row, 0, end_row, 0],  # Use Character Count column
            'values':     [sheet_name, start_row, 4, end_row, 4],  # Use Cumulative % column
            'line':       {'color': '#C0504D', 'width': 2}
        })

    cum_chart.set_title({'name': f'Cumulative Distribution of Line Lengths{title_suffix}'})
    cum_chart.set_x_axis({
        'name': 'Number of Characters (UTF-8)',
        'max': 80,
        'interval_unit': 5,  # Show labels and gridlines every 5 units
        'interval_tick': 5,  # Start at position 5 (1-based), which is value 5
        'major_gridlines': {'visible': True}
    })
    cum_chart.set_y_axis({'name': 'Percentage (%)'})
    cum_chart.set_legend({'position': 'none'})

    # Place charts on the sheet
    # Charts are positioned side by side, below the data
    chart_row_start = data_start_row + max_chart_row + 3  # Start charts below data with some spacing
    chart_position = f'D{chart_row_start}'
    # Place cumulative chart to the right of the histogram (8 columns over for more spacing)
    cum_chart_position = f'L{chart_row_start}'

    worksheet.insert_chart(chart_position, chart)
    worksheet.insert_chart(cum_chart_position, cum_chart)

    print(f"Successfully added sheet '{sheet_name}' to workbook.")

    return writer, hist_data, sheet_name


# --- Example Usage ---
if __name__ == "__main__":
    # Create a single writer for all datasets
    writer = None
    all_data = []  # Store (hist_data, sheet_name) tuples for summary sheet

    try:
        writer, hist_data, sheet_name = generate_line_histogram(
            directory="Braille/Nemeth/highschool", file_pattern="*.brls", writer=writer
        )
        if not hist_data.empty:
            all_data.append((hist_data, sheet_name))

        writer, hist_data, sheet_name = generate_line_histogram(
            directory="Braille/Nemeth/college", file_pattern="*.brls", writer=writer
        )
        if not hist_data.empty:
            all_data.append((hist_data, sheet_name))

        writer, hist_data, sheet_name = generate_line_histogram(
            directory="Braille/UEB/highschool", file_pattern="*.brls", writer=writer
        )
        if not hist_data.empty:
            all_data.append((hist_data, sheet_name))

        writer, hist_data, sheet_name = generate_line_histogram(
            directory="Braille/UEB/college", file_pattern="*.brls", writer=writer
        )
        if not hist_data.empty:
            all_data.append((hist_data, sheet_name))

        # Create summary sheet with overlay charts
        if all_data and writer is not None:
            generate_summary_sheet(writer, all_data)
    finally:
        if writer is not None:
            writer.close()
            print("All sheets added to 'braille-lengths.xlsx'.")
