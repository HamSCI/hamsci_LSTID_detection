#!/usr/bin/env python3

import polars as pl
from datetime import datetime
from matplotlib.ticker import FuncFormatter    
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import AxesImage
from matplotlib.colors import LogNorm
from matplotlib import cm
from scipy.ndimage import gaussian_filter
import numpy as np
import os
import io
from PIL import Image
from scripts.utils_geo import *
from scripts.utils import *
from matplotlib.patches import Polygon

############################ Tier 1 plots ############################

def plot_dist_vs_ut(ax, df_box: pl.DataFrame, time_bin_size_secs: int = 600, dist_bin_size: int = 2):
    # Convert to pandas to use datetime functionality
    df_pd = df_box.to_pandas()

    # Extract seconds since epoch from datetime column
    times_unix = df_pd["date"].astype("int64") // 10**9  # UNIX time in seconds
    distances = df_pd["dist_Km"].to_numpy()

    # Define histogram bin edges
    time_edges = np.arange(times_unix.min(), times_unix.max() + time_bin_size_secs, time_bin_size_secs)
    dist_edges = np.arange(distances.min(), distances.max() + dist_bin_size, dist_bin_size)

    # Build 2D histogram
    heatmap, _, _ = np.histogram2d(times_unix, distances, bins=[time_edges, dist_edges])
    norm = LogNorm(vmin=np.nanmin(heatmap[heatmap > 0]), vmax=np.nanmax(heatmap))

    # Set colormap
    cmap = cm.viridis.copy()
    cmap.set_bad(color='black')

    # Plot the image
    img = ax.imshow(
        heatmap.T, origin='lower', aspect='auto', cmap=cmap, alpha=0.7,
        norm=norm,
        extent=[time_edges[0], time_edges[-1], dist_edges[0], dist_edges[-1]]
    )


    def format_ut(x, pos):
        try:
            return datetime.utcfromtimestamp(x).strftime('%H:%M')
        except Exception:
            return ""

    ax.xaxis.set_major_formatter(FuncFormatter(format_ut))

    # Labels and title
    ax.set_xlabel("Time (UT)", fontsize=10)
    ax.set_ylabel("Distance (km)", fontsize=10)
    ax.set_title("Distance vs Time (UT, Log Scale)", fontsize=12)

    return img

def plot_map(ax, df_box: pl.DataFrame, region: dict):
    lat_min, lat_max = region['lat_lim']
    lon_min, lon_max = region['lon_lim']

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_facecolor("black")
    ax.add_feature(cfeature.COASTLINE, edgecolor="white")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="white")
    ax.add_feature(cfeature.STATES, linestyle=":", edgecolor="white")

    lats = df_box['mid_lat'].to_numpy()
    lons = df_box['mid_long'].to_numpy()

    bins = 100
    heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bins,
                                             range=[[lon_min, lon_max], [lat_min, lat_max]])
    heatmap = np.log1p(heatmap)
    heatmap = gaussian_filter(heatmap, sigma=2)

    img = ax.imshow(heatmap.T, extent=[lon_min, lon_max, lat_min, lat_max],
                    origin='lower', cmap='inferno', alpha=0.7, aspect='auto')

    ax.set_title('Spot Midpoint Locations (Geocentric Coordinates)', fontsize=12)
    return img


def plot_dist_vs_lon(ax, df_box: pl.DataFrame, altitude: float):
    times = df_box[f"geomag_lon_{altitude}"].to_numpy()
    distances = df_box["dist_Km"].to_numpy()

    time_bin_size = 1
    dist_bin_size = 50

    time_edges = np.arange(times.min(), times.max() + time_bin_size, time_bin_size)
    dist_edges = np.arange(distances.min(), distances.max() + dist_bin_size, dist_bin_size)

    heatmap, _, _ = np.histogram2d(times, distances, bins=[time_edges, dist_edges])
    norm = LogNorm(vmin=np.nanmin(heatmap[heatmap > 0]), vmax=np.nanmax(heatmap))

    cmap = cm.viridis.copy()
    cmap.set_bad(color='black')

    img = ax.imshow(
        heatmap.T, origin='lower', aspect='auto', cmap=cmap, alpha=0.7,
        norm=norm, extent=[time_edges[0], time_edges[-1], dist_edges[0], dist_edges[-1]]
    )

    ax.set_xlabel("Geomagnetic Longitude", fontsize=10)
    ax.set_ylabel("Distance (km)", fontsize=10)
    ax.set_title("Distance vs Geomagnetic Longitude (Log Scale)", fontsize=12)

    return img


def plot_spot_count(ax, df_box: pl.DataFrame, altitude: float):
    lons = df_box[f"geomag_lon_{altitude}"].to_numpy()
    lon_bins = np.linspace(lons.min(), lons.max(), 50)
    spot_counts, _ = np.histogram(lons, bins=lon_bins)

    line, = ax.plot(lon_bins[:-1], spot_counts, marker="o", linestyle='-', color='tab:blue')

    ax.set_xlabel("Geomagnetic Longitude", fontsize=10)
    ax.set_ylabel("Total Spot Count", fontsize=10)
    ax.set_title(f"Geomagnetic Longitude vs. Total Spot Counts at {altitude} km", fontsize=12)

    return line


def plot_text_box(ax, df: pl.DataFrame, date_col: str = "date"):
    date_str = df[date_col][0].strftime("%Y-%m-%d") if len(df) > 0 else "No Data"
    percentages = calculate_source_percent(df)
    spot_count = df.height

    text_lines = [f"Date: {date_str}", f"Spot count: {spot_count}"]
    text_lines += [f"{source}: {percent:.1f}%" for source, percent in percentages.items()]
    text = "\n".join(text_lines)

    txt = ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12,
                  transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.axis("off")

    return txt


def plot_globe(ax, df, region, date_str):
    altitude = 0
    geomag_lons = np.linspace(-180, 180, 360)
    geomag_lats = np.zeros_like(geomag_lons)
    line_lats, line_lons = convert_geomagnetic_to_geocentric(geomag_lats, geomag_lons, altitude, date_str)

    # Plot red dotted line (equator)
    ax.plot(line_lons, line_lats, color='red', linestyle='dotted', linewidth=2,
            transform=ccrs.Geodetic(), alpha=0.75)

    ecuator_label_position = 148
    ecuator_height_adjust = 1
    ax.text(line_lons[ecuator_label_position], line_lats[ecuator_label_position] + ecuator_height_adjust,
            "0°", color='red', fontsize=10, ha='center', transform=ccrs.Geodetic())

    gl = ax.gridlines(draw_labels=True, color='gray', linewidth=0.5, linestyle='--')
    gl.xlocator = plt.MultipleLocator(30)
    gl.ylocator = plt.MultipleLocator(15)
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.ylabels_right = False

    ax.coastlines(resolution='50m', color='blue', linewidth=1.5, alpha=0.5)
    ax.set_global()
    ax.set_facecolor('white')

    lat_min, lat_max = region['lat_lim']
    lon_min, lon_max = region['lon_lim']

    lats = df['mid_lat'].to_numpy()
    lons = df['mid_long'].to_numpy()

    bins = 100
    heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bins,
                                             range=[[lon_min, lon_max], [lat_min, lat_max]])
    heatmap = np.log1p(heatmap)
    heatmap = gaussian_filter(heatmap, sigma=2)

    img = ax.imshow(heatmap.T, extent=[lon_min, lon_max, lat_min, lat_max],
                    origin='lower', cmap='inferno', alpha=1.0, aspect='auto', transform=ccrs.PlateCarree())

    ax.set_title('Spot Midpoints (Geographic Coordinates)', fontsize=14)
    return img


def plot_map_geomag(ax, df_box: pl.DataFrame, region: dict, alt: float, date_str: str):
    latlim, lonlim = convert_geocentric_to_geomagnetic(region['lat_lim'], region['lon_lim'], alt, date_str)

    lats = df_box[f"geomag_lat_{alt}"].to_numpy()
    lons = df_box[f"geomag_lon_{alt}"].to_numpy()
    
    lat_min = min(latlim[0], lats.min())
    lat_max = max(latlim[1], lats.max())
    lon_min = min(lonlim[0], lons.min())
    lon_max = max(lonlim[1], lons.max())
    
    bins = 100
    heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bins,
                                             range=[[lon_min, lon_max], [lat_min, lat_max]])

    heatmap = gaussian_filter(heatmap, sigma=1)
    heatmap_masked = np.ma.masked_where(heatmap == 0, np.log1p(heatmap) * 3)
    
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color='black')

    img = ax.imshow(heatmap_masked.T, extent=[lon_min, lon_max, lat_min, lat_max],
                    origin='lower', cmap=cmap, alpha=1, aspect='auto')

    ax.axhline(y=0, color='red', linestyle=':', linewidth=2.5, alpha=0.8)
    ax.set_title('Spot Midpoints (Geomagnetic Coordinates)', fontsize=14)
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)

    return img

############################ Tier 2 Subplots ############################

def plot_map_distTime_spot(df: pl.DataFrame, region: dict, altitude: float = 300):
    """Generate a composite plot of map, time-distance, spot count, and colorbar."""
    sDate = df['date'].min()
    eDate = df['date'].max()

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 0.05], height_ratios=[1, 1])

    ax_map = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_dist_time = fig.add_subplot(gs[0, 1])
    ax_spot = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[0, 0])
    ax_colorbar = fig.add_subplot(gs[0, 2])

    fig.suptitle(f"{sDate} - {eDate}", fontsize=16)

    plot_map(ax_map, df, region)
    dist_plot = plot_dist_vs_lon(ax_dist_time, df, altitude)
    spot_plot = plot_spot_count(ax_spot, df, altitude)
    plot_text_box(ax_text, df)

    ax_colorbar.cla()
    plt.colorbar(dist_plot, cax=ax_colorbar, orientation='vertical')
    ax_colorbar.set_ylabel("Spot Count (Log Density)", fontsize=10)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    return fig


def plot_globe_geomagMap(df: pl.DataFrame, region: dict, altitude: float = 300):
    """Generate a globe and geomagnetic projection plot with shared colorbars."""
    sDate = df['date'].min()
    eDate = df['date'].max()
    if df.is_empty():
        fig = plt.figure()
        return fig
    date_str = df['date'].dt.strftime('%Y-%m-%d')[0]

    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1])

    ax_globe = fig.add_subplot(gs[0, 0], projection=ccrs.Orthographic(central_longitude=-60, central_latitude=0))
    ax_map_geomag = fig.add_subplot(gs[1, 0])
    ax_cbar_1 = fig.add_subplot(gs[0, 1])
    ax_cbar_2 = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"{sDate} - {eDate}", fontsize=16)

    globe_img = plot_globe(ax_globe, df, region, date_str)
    geomag_img = plot_map_geomag(ax_map_geomag, df, region, altitude, date_str)

    fig.colorbar(globe_img, cax=ax_cbar_1, orientation='vertical')
    ax_cbar_1.set_ylabel("Spot Count (Log Density)", fontsize=10)

    fig.colorbar(geomag_img, cax=ax_cbar_2, orientation='vertical')
    ax_cbar_2.set_ylabel("Spot Count (Log Density)", fontsize=10)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    return fig

def plot_distTime_only(df: pl.DataFrame, time_bin_size_secs: int = 600, dist_bin_size: int = 2):
    """Create a standalone distance vs time (UT) plot with colorbar."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colorbar import Colorbar

    # Determine time range for the title
    sDate = df['date'].min()
    eDate = df['date'].max()

    # Set up the figure and axes
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])  # main plot + colorbar

    ax_main = fig.add_subplot(gs[0])
    ax_cbar = fig.add_subplot(gs[1])

    # Create the plot
    img = plot_dist_vs_ut(ax_main, df, time_bin_size_secs, dist_bin_size)

    # Attach the colorbar
    plt.colorbar(img, cax=ax_cbar, orientation='vertical')
    ax_cbar.set_ylabel("Spot Count (Log Density)", fontsize=10)

    # Title
    fig.suptitle(f"{sDate} — {eDate} UTC", fontsize=14)

    plt.tight_layout()
    return fig


############################ Tier 3 Combined PLots ############################

def plot_combined_plot_2(dfs, region, altitude, output_folder, freq, n_rows=4, n_cols=2):
    figs = []
    for df_sub in dfs:
        fig = plot_globe_geomagMap(df_sub, region, altitude)
        figs.append(fig)

    n_plots = len(figs)

    # Dynamically determine layout if not provided
    if n_rows is None and n_cols is None:
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
    elif n_rows is None:
        n_rows = (n_plots + n_cols - 1) // n_cols
    elif n_cols is None:
        n_cols = (n_plots + n_rows - 1) // n_rows

    fig_combined, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows))
    axs = axs.flatten() if n_plots > 1 else [axs]

    for ax, fig in zip(axs, figs):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        ax.imshow(np.array(img))
        ax.axis('off')

    for ax in axs[n_plots:]:
        ax.axis('off')

    fig_combined.suptitle(freq + ", 0-10000km Dist", fontsize=16)
    fig_combined.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, "combined_plot_2.png")
    fig_combined.savefig(out_path, dpi=200)

    print(f"Wrote {out_path}")

def plot_single_distTime_combined(df, output_folder, freq, time_bin_size_secs=600, dist_bin_size=2):
    """
    Generate and save a single distance vs time (UT) plot embedded in a combined-style figure.
    """
    fig_single = plot_distTime_only(df, time_bin_size_secs, dist_bin_size)

    # Create one-panel combined figure
    fig_combined, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Save single figure to buffer, draw into ax as image
    buf = io.BytesIO()
    fig_single.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig_single)
    buf.seek(0)
    img = Image.open(buf)
    ax.imshow(np.array(img))
    ax.axis('off')
    
    fig_combined.suptitle("Distance vs Time (UT) " + freq + 'MHz', fontsize=16)
    fig_combined.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, "dist_time_single.png")
    fig_combined.savefig(out_path, dpi=200)

    print(f"Wrote {out_path}")

    
# Example usage for testing purposes:
if __name__ == "__main__":
    # Read the DataFrame
    df = pl.read_parquet("../cache/df_gen/2017-07-01_lat-30_30_lon-100_-30_0.00MHz_30.00MHz_dist0_20000km_altitudes_0_100_300.parquet")

    # Define region and altitude for plotting
    region = {
        'lat_lim': [-30, 30],  
        'lon_lim': [-100, -30]  
    }
    
    altitude = 0  # Example altitude

    # Create output folder if it doesn't exist
    output_folder = "../output"
    test_plots_folder = f"{output_folder}/tests"
    os.makedirs(test_plots_folder, exist_ok=True)

    # Plotting and saving individual plots
    # Plot Map
    fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    plot_map(ax1, df, region)
    fig1.savefig(f"{test_plots_folder}/plot_map.png")
    print(f"Saved plot_map.png")

    # Plot Distance vs Longitude
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_dist_vs_lon(ax2, df, altitude)
    fig2.savefig(f"{test_plots_folder}/plot_dist_vs_lon.png")
    print(f"Saved plot_dist_vs_lon.png")

    # Plot Spot Count
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    plot_spot_count(ax3, df, altitude)
    fig3.savefig(f"{test_plots_folder}/plot_spot_count.png")
    print(f"Saved plot_spot_count.png")

    # Plot Text Box
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    plot_text_box(ax4, df)
    fig4.savefig(f"{test_plots_folder}/plot_text_box.png")
    print(f"Saved plot_text_box.png")

    # Plot Globe
    fig5, ax5 = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.Orthographic(
        central_longitude=-60, central_latitude=0)})
    
    # Call the plot_globe function
    plot_globe(ax5, df, region, date_str='2017-03-01')
    
    # Save the plot
    fig5.savefig(f"{test_plots_folder}/plot_globe.png")
    print(f"Saved plot_globe.png")

    fig6, ax6 = plt.subplots(figsize=(8, 6))
    plot_map_geomag(ax6, df, region, altitude, date_str='2017-07-01')
    fig6.savefig(f"{test_plots_folder}/plot_map_geomag.png")
    print(f"Saved plot_map_geomag.png")