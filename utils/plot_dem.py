"""
DEM Visualization and Frequency Analysis
-----------------------------------------
Loads a Digital Elevation Model (DEM), renders it in 3D, and analyses
terrain reconstruction quality at different FFT frequency resolutions.
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = "./datasets/sen1floods11_v1.1/data/CopernicusDEM"
SAVE_PATH = "./plots"
MODES_LIST = [8, 4, 2, 1]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dem(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the first GeoTIFF found under `root` and return the elevation
    array together with coordinate grids X and Y."""
    file = glob.glob(f"{root}/*.tif")[0]
    with rasterio.open(file) as src:
        dem = src.read(1)
        transform = src.transform

    x = np.arange(dem.shape[1]) * transform[0] + transform[2]
    y = np.arange(dem.shape[0]) * transform[4] + transform[5]
    X, Y = np.meshgrid(x, y)

    return dem, X, Y


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------

def compute_fft(dem: np.ndarray) -> torch.Tensor:
    """Return the centred 2-D FFT of `dem` as a complex tensor."""
    dem_tensor = torch.from_numpy(dem).float()
    return torch.fft.fftshift(torch.fft.fft2(dem_tensor))


def reconstruct_from_modes(fft_shifted: torch.Tensor, modes: int) -> np.ndarray:
    """Reconstruct a DEM from the central `modes` × `modes` frequency block.

    Parameters
    ----------
    fft_shifted : torch.Tensor
        Centred FFT of the original DEM.
    modes : int
        Half-width (in pixels) of the low-frequency window to keep.

    Returns
    -------
    np.ndarray
        Real-valued reconstruction with the same shape as the input DEM.
    """
    h, w = fft_shifted.shape
    ch, cw = h // 2, w // 2

    mask = torch.zeros((h, w), dtype=torch.complex64)
    mask[ch - modes : ch + modes, cw - modes : cw + modes] = 1

    filtered = torch.fft.ifftshift(fft_shifted * mask)
    return torch.fft.ifft2(filtered).real.numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_3d_dem(dem: np.ndarray, X: np.ndarray, Y: np.ndarray,
                save_path: str, step: int = 5) -> None:
    """Render `dem` as a 3-D surface and save the figure."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X[::step, ::step], Y[::step, ::step], dem[::step, ::step],
        cmap="Purples", linewidth=0, antialiased=False,
    )
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Elevation")
    ax.set_title("3D DEM Visualization")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fft_panel(dem: np.ndarray, fft_shifted: torch.Tensor,
                   modes_list: list[int], save_path: str) -> None:
    """2-D panel: original DEM, FFT magnitude, and one tile per mode count."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Original DEM
    axes[0].imshow(dem, cmap="Purples")
    axes[0].set_title(f"Original DEM\n{dem.shape[0]}×{dem.shape[1]}")
    axes[0].axis("off")

    # FFT magnitude spectrum
    magnitude = torch.log(torch.abs(fft_shifted) + 1).numpy()
    axes[1].imshow(magnitude, cmap="hot")
    axes[1].set_title("FFT Magnitude Spectrum")
    axes[1].axis("off")

    # Reconstructions
    for i, modes in enumerate(modes_list):
        recon = reconstruct_from_modes(fft_shifted, modes)
        mse = np.mean((dem - recon) ** 2)

        axes[i + 2].imshow(recon, cmap="Purples")
        axes[i + 2].set_title(f"{modes}×{modes} modes")
        axes[i + 2].axis("off")
        axes[i + 2].text(
            0.05, 0.05, f"MSE: {mse:.2f}",
            transform=axes[i + 2].transAxes,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_3d_reconstructions(fft_shifted: torch.Tensor, X: np.ndarray,
                             Y: np.ndarray, modes_list: list[int],
                             save_path: str, step: int = 3) -> None:
    """3-D surface grid showing terrain reconstructed at each mode count."""
    fig = plt.figure(figsize=(16, 12))

    for idx, modes in enumerate(modes_list):
        recon = reconstruct_from_modes(fft_shifted, modes)
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")

        surf = ax.plot_surface(
            X[::step, ::step], Y[::step, ::step], recon[::step, ::step],
            cmap="Purples", linewidth=0, antialiased=True, alpha=0.8,
        )
        ax.set_title(f"{modes}×{modes} modes ({(2 * modes + 1) ** 2} frequencies)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Elevation")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_frequency_summary(dem: np.ndarray, modes_list: list[int]) -> None:
    """Print a concise summary of pixel counts vs. retained frequencies."""
    print("\nFrequency Analysis Summary:")
    print(f"Original DEM: {dem.shape[0]}×{dem.shape[1]} = {dem.size:,} pixels")
    print("\nReconstructions:")
    for modes in modes_list:
        freqs = (2 * modes + 1) ** 2
        pct = 100 * freqs / dem.size
        print(f"  {modes}×{modes} modes: {freqs:,} frequencies ({pct:.2f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dem, X, Y = load_dem(ROOT)
    fft_shifted = compute_fft(dem)

    plot_3d_dem(dem, X, Y, save_path=f"{SAVE_PATH}/DEM.png")
    plot_fft_panel(dem, fft_shifted, MODES_LIST, save_path=f"{SAVE_PATH}/FFT.png")
    plot_3d_reconstructions(fft_shifted, X, Y, MODES_LIST, save_path=f"{SAVE_PATH}/RECON.png")
    print_frequency_summary(dem, MODES_LIST)


if __name__ == "__main__":
    main()