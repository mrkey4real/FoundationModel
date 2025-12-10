import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json
import re
import sys

# Configuration
DATA_PATH = Path(r"..\data\merged_East_labview_egauge_1min.csv")
# 基础输出目录
BASE_DIR = Path(r"..\data\quality_analysis")
OUTPUT_PATH = BASE_DIR / "manual_column_selection.json"
PROGRESS_PATH = BASE_DIR / "selection_progress.json"
# 图片存放目录
PLOTS_DIR = BASE_DIR / "plots_cache"


def get_safe_filename(col_name):
    """Convert column name to safe filename."""
    # Replace non-alphanumeric characters with underscore
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', col_name)
    # Avoid overly long filenames
    return f"{clean_name[:100]}.png"


def load_data():
    """Load the merged dataset."""
    print("Loading data...")
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'], index_col='timestamp')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_progress():
    """Load previous selection progress if exists."""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'essential': [],
        'optional': [],
        'toss': [],
        'last_index': 0
    }


def save_progress(selections):
    """Save current progress."""
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump(selections, f, indent=2, ensure_ascii=False)
    print(f"Progress saved to {PROGRESS_PATH}")


def save_final(selections):
    """Save final selection."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(selections, f, indent=2, ensure_ascii=False)
    print(f"Final selection saved to {OUTPUT_PATH}")


def compute_statistics(series):
    """Compute comprehensive statistics for a column."""
    stats = {}

    # Basic info
    stats['total_count'] = len(series)
    stats['missing_count'] = series.isna().sum()
    stats['missing_pct'] = 100 * stats['missing_count'] / stats['total_count']
    stats['valid_count'] = stats['total_count'] - stats['missing_count']

    # Get valid data
    valid_data = series.dropna()

    if len(valid_data) == 0:
        stats['status'] = 'ALL_MISSING'
        return stats

    # Check if numeric
    if not np.issubdtype(valid_data.dtype, np.number):
        stats['status'] = 'NON_NUMERIC'
        stats['unique_values'] = valid_data.nunique()
        stats['sample_values'] = valid_data.value_counts().head(5).to_dict()
        return stats

    stats['status'] = 'NUMERIC'

    # Numeric statistics
    stats['min'] = valid_data.min()
    stats['max'] = valid_data.max()
    stats['mean'] = valid_data.mean()
    stats['median'] = valid_data.median()
    stats['std'] = valid_data.std()

    # Percentiles
    stats['p25'] = valid_data.quantile(0.25)
    stats['p75'] = valid_data.quantile(0.75)
    stats['p95'] = valid_data.quantile(0.95)
    stats['p99'] = valid_data.quantile(0.99)

    # Check for constant
    stats['unique_values'] = valid_data.nunique()
    stats['is_constant'] = stats['unique_values'] <= 1 or stats['std'] < 1e-10

    # Check for zeros
    stats['zero_count'] = (valid_data == 0).sum()
    stats['zero_pct'] = 100 * stats['zero_count'] / len(valid_data)

    # Check for negative values
    stats['negative_count'] = (valid_data < 0).sum()
    stats['negative_pct'] = 100 * stats['negative_count'] / len(valid_data)

    # Non-zero time (useful for power data)
    stats['nonzero_pct'] = 100 * (valid_data.abs() > 0.001).sum() / len(valid_data)

    return stats


def plot_column(df, col, stats):
    """Create visualization for a column."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{col}', fontsize=14, fontweight='bold')

    valid_data = df[col].dropna()

    if len(valid_data) == 0:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        return fig

    # 1. Time series (downsampled for speed)
    ax1 = axes[0, 0]
    # Resample to hourly for faster plotting
    hourly = df[col].resample('1H').mean()
    ax1.plot(hourly.index, hourly.values, linewidth=0.5, alpha=0.8)
    ax1.set_title('Time Series (hourly mean)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)

    # Mark missing periods
    missing_mask = df[col].isna().resample('1H').mean()
    missing_periods = missing_mask[missing_mask > 0.5]
    for idx in missing_periods.index:
        ax1.axvline(idx, color='red', alpha=0.1, linewidth=0.5)

    # 2. Histogram
    ax2 = axes[0, 1]
    try:
        # Remove extreme outliers for better visualization
        q01 = valid_data.quantile(0.01)
        q99 = valid_data.quantile(0.99)
        plot_data = valid_data[(valid_data >= q01) & (valid_data <= q99)]
        ax2.hist(plot_data, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(stats.get('mean', 0), color='red', linestyle='--', label=f"Mean: {stats.get('mean', 0):.2f}")
        ax2.axvline(stats.get('median', 0), color='green', linestyle='--', label=f"Median: {stats.get('median', 0):.2f}")
        ax2.legend()
    except:
        ax2.text(0.5, 0.5, 'Cannot plot histogram', ha='center', va='center')
    ax2.set_title('Distribution (1%-99% range)')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')

    # 3. Daily pattern (if enough data)
    ax3 = axes[1, 0]
    try:
        hourly_pattern = df[col].groupby(df.index.hour).mean()
        ax3.bar(hourly_pattern.index, hourly_pattern.values, alpha=0.7)
        ax3.set_title('Hourly Pattern (average)')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Mean Value')
        ax3.set_xticks(range(0, 24, 3))
    except:
        ax3.text(0.5, 0.5, 'Cannot compute pattern', ha='center', va='center')

    # 4. Monthly trend
    ax4 = axes[1, 1]
    try:
        monthly = df[col].resample('M').mean()
        ax4.bar(range(len(monthly)), monthly.values, alpha=0.7)
        ax4.set_title('Monthly Mean')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Mean Value')
        month_labels = [d.strftime('%b') for d in monthly.index]
        ax4.set_xticks(range(len(monthly)))
        ax4.set_xticklabels(month_labels, rotation=45)
    except:
        ax4.text(0.5, 0.5, 'Cannot compute monthly', ha='center', va='center')

    plt.tight_layout()
    return fig


def print_statistics(col, stats, idx, total):
    """Print formatted statistics."""
    print("\n" + "="*80)
    print(f"Column {idx+1}/{total}: {col}")
    print("="*80)

    print(f"\n[Data Availability]")
    print(f"  Total points:    {stats['total_count']:,}")
    print(f"  Valid points:    {stats['valid_count']:,}")
    print(f"  Missing:         {stats['missing_count']:,} ({stats['missing_pct']:.1f}%)")

    if stats['status'] == 'ALL_MISSING':
        print(f"\n  *** ALL DATA IS MISSING ***")
        return

    if stats['status'] == 'NON_NUMERIC':
        print(f"\n[Non-numeric Data]")
        print(f"  Unique values: {stats['unique_values']}")
        print(f"  Sample values: {stats['sample_values']}")
        return

    print(f"\n[Value Range]")
    print(f"  Min:     {stats['min']:.4f}")
    print(f"  Max:     {stats['max']:.4f}")
    print(f"  Range:   {stats['max'] - stats['min']:.4f}")

    print(f"\n[Central Tendency]")
    print(f"  Mean:    {stats['mean']:.4f}")
    print(f"  Median:  {stats['median']:.4f}")
    print(f"  Std:     {stats['std']:.4f}")

    print(f"\n[Percentiles]")
    print(f"  25th:    {stats['p25']:.4f}")
    print(f"  75th:    {stats['p75']:.4f}")
    print(f"  95th:    {stats['p95']:.4f}")
    print(f"  99th:    {stats['p99']:.4f}")

    print(f"\n[Special Values]")
    print(f"  Unique values:  {stats['unique_values']:,}")
    print(f"  Zero values:    {stats['zero_count']:,} ({stats['zero_pct']:.1f}%)")
    print(f"  Negative:       {stats['negative_count']:,} ({stats['negative_pct']:.1f}%)")
    print(f"  Non-zero time:  {stats['nonzero_pct']:.1f}%")

    if stats['is_constant']:
        print(f"\n  *** WARNING: CONSTANT OR NEAR-CONSTANT ***")


def generate_all_plots(df):
    """Batch generate plots for all columns and save to disk."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    columns = list(df.columns)
    total = len(columns)

    print(f"\nStarting batch generation of {total} plots...")
    print(f"Saving to: {PLOTS_DIR}")

    for idx, col in enumerate(columns):
        filename = get_safe_filename(col)
        filepath = PLOTS_DIR / filename

        # Skip if already exists (resume capability)
        if filepath.exists():
            print(f"[{idx+1}/{total}] Skipping {col} (already exists)")
            continue

        print(f"[{idx+1}/{total}] Plotting {col}...")
        
        # Calculate stats (needed for plot lines)
        stats = compute_statistics(df[col])
        
        # Create plot
        fig = plot_column(df, col, stats)
        
        # Save and close immediately to free memory
        try:
            fig.savefig(filepath, dpi=100) # dpi=100 is enough for screen review
        except Exception as e:
            print(f"  Error saving {col}: {e}")
        finally:
            plt.close(fig) # Critical: close figure to prevent memory leak

    print("\nBatch generation complete!")


def show_saved_image(col):
    """Load and display a saved image."""
    filename = get_safe_filename(col)
    filepath = PLOTS_DIR / filename

    if not filepath.exists():
        return False, None

    try:
        img = mpimg.imread(filepath)
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img)
        ax.axis('off') # Hide axes since they are in the image
        ax.set_title(f"Viewing cached plot: {col}", fontsize=10)
        plt.tight_layout()
        return True, fig
    except Exception as e:
        print(f"Error loading image: {e}")
        return False, None


def interactive_selection(df):
    """Main interactive selection loop."""
    columns = list(df.columns)
    total = len(columns)

    # Load previous progress
    selections = load_progress()
    start_idx = selections['last_index']

    # Already classified columns
    classified = set(selections['essential'] + selections['optional'] + selections['toss'])

    print(f"\nStarting from column {start_idx + 1}")
    print(f"Already classified: {len(classified)} columns")
    print(f"  Essential: {len(selections['essential'])}")
    print(f"  Optional:  {len(selections['optional'])}")
    print(f"  Toss:      {len(selections['toss'])}")

    print("\n" + "-"*60)
    print("Commands:")
    print("  e = Essential (core HVAC logic)")
    print("  o = Optional (nice to have)")
    print("  t = Toss (drop this column)")
    print("  s = Skip (review later)")
    print("  b = Back (go to previous column)")
    print("  j = Jump to specific column number")
    print("  v = View plot again")
    print("  q = Quit and save progress")
    print("  f = Finish and save final selection")
    print("-"*60)

    idx = start_idx

    while idx < total:
        col = columns[idx]

        # Skip already classified
        if col in classified:
            idx += 1
            continue

        # Compute statistics (fast enough to do on the fly for text output)
        stats = compute_statistics(df[col])
        print_statistics(col, stats, idx, total)

        # Try to show saved image first
        success, fig = show_saved_image(col)
        
        # If no image found, generate on the fly
        if not success:
            print("  (Cached plot not found, generating live...)")
            fig = plot_column(df, col, stats)
        
        plt.show(block=False)
        plt.pause(0.1)

        # Get user input
        while True:
            choice = input(f"\nClassify '{col}' [e/o/t/s/b/j/v/q/f]: ").strip().lower()

            if choice == 'e':
                selections['essential'].append(col)
                classified.add(col)
                print(f"  -> Added to ESSENTIAL")
                plt.close(fig)
                idx += 1
                break
            elif choice == 'o':
                selections['optional'].append(col)
                classified.add(col)
                print(f"  -> Added to OPTIONAL")
                plt.close(fig)
                idx += 1
                break
            elif choice == 't':
                selections['toss'].append(col)
                classified.add(col)
                print(f"  -> Added to TOSS")
                plt.close(fig)
                idx += 1
                break
            elif choice == 's':
                print(f"  -> Skipped")
                plt.close(fig)
                idx += 1
                break
            elif choice == 'b':
                if idx > 0:
                    idx -= 1
                    # Remove from classifications if exists
                    prev_col = columns[idx]
                    for cat in ['essential', 'optional', 'toss']:
                        if prev_col in selections[cat]:
                            selections[cat].remove(prev_col)
                            classified.discard(prev_col)
                            print(f"  -> Going back to '{prev_col}'")
                            break
                plt.close(fig)
                break
            elif choice == 'j':
                try:
                    new_idx = int(input("  Jump to column number (1-based): ")) - 1
                    if 0 <= new_idx < total:
                        idx = new_idx
                        plt.close(fig)
                        break
                    else:
                        print(f"  Invalid index. Must be 1-{total}")
                except ValueError:
                    print("  Invalid number")
            elif choice == 'v':
                plt.close(fig)
                # Retry showing
                success, fig = show_saved_image(col)
                if not success:
                    fig = plot_column(df, col, stats)
                plt.show(block=False)
                plt.pause(0.1)
            elif choice == 'q':
                selections['last_index'] = idx
                save_progress(selections)
                plt.close('all')
                print("\nProgress saved. Run again to continue.")
                return selections
            elif choice == 'f':
                save_final(selections)
                plt.close('all')
                print("\nFinal selection saved!")
                return selections
            else:
                print("  Invalid choice. Use e/o/t/s/b/j/v/q/f")

        # Auto-save progress every 10 columns
        if idx % 10 == 0:
            selections['last_index'] = idx
            save_progress(selections)

    # Finished all columns
    print("\n" + "="*80)
    print("ALL COLUMNS REVIEWED!")
    print("="*80)
    print(f"  Essential: {len(selections['essential'])}")
    print(f"  Optional:  {len(selections['optional'])}")
    print(f"  Toss:      {len(selections['toss'])}")

    save_final(selections)
    return selections


def show_summary(selections):
    """Show summary of selections."""
    print("\n" + "="*80)
    print("SELECTION SUMMARY")
    print("="*80)

    print(f"\n[Essential - {len(selections['essential'])} columns]")
    for col in sorted(selections['essential']):
        print(f"  + {col}")

    print(f"\n[Optional - {len(selections['optional'])} columns]")
    for col in sorted(selections['optional']):
        print(f"  ~ {col}")

    print(f"\n[Toss - {len(selections['toss'])} columns]")
    for col in sorted(selections['toss']):
        print(f"  - {col}")


def main():
    """Main entry point."""
    print("="*60)
    print("INTERACTIVE COLUMN SELECTION TOOL")
    print("="*60)

    # Mode Selection
    print("\nSelect Mode:")
    print("  1. GENERATE PLOTS (Run this first, non-interactive)")
    print("  2. START SELECTION (Interactive, loads generated plots)")
    print("  3. View Summary only")
    
    mode = input("\nEnter mode [1/2/3]: ").strip()

    if mode == '1':
        # Batch generation mode
        df = load_data()
        generate_all_plots(df)
        print("\nAll plots generated. You can now restart and run Mode 2.")
    
    elif mode == '2':
        # Interactive mode
        # Check for existing progress
        if PROGRESS_PATH.exists():
            selections = load_progress()
            print(f"\nFound existing progress:")
            print(f"  Essential: {len(selections['essential'])}")
            print(f"  Optional:  {len(selections['optional'])}")
            print(f"  Toss:      {len(selections['toss'])}")
            print(f"  Last index: {selections['last_index']}")

            choice = input("\n[c]ontinue, [r]estart? ").strip().lower()
            if choice == 'r':
                selections = {'essential': [], 'optional': [], 'toss': [], 'last_index': 0}
        
        # Load data
        df = load_data()

        # Start interactive selection
        plt.ion()  # Enable interactive mode
        selections = interactive_selection(df)
        plt.ioff()
        
        # Show final summary
        show_summary(selections)
    
    elif mode == '3':
        if PROGRESS_PATH.exists():
            selections = load_progress()
            show_summary(selections)
        else:
            print("No progress file found.")
    
    else:
        print("Invalid mode selected.")

if __name__ == '__main__':
    main()