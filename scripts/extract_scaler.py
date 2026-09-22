"""
extract_scaler.py
Extract scaler_min/scaler_max from patches NPZ files into a tiny standalone
file for upload to Google Colab/Drive.

Usage:
    python scripts/extract_scaler.py

Output:
    data/scaler_stats_7x7.npz   (~2 KB)
    data/scaler_stats_15x15.npz
    data/scaler_stats_21x21.npz
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import PATCH_SIZES, DATA_DIR, BAND_NAMES

def main():
    data_dir = Path(DATA_DIR)

    for ps in PATCH_SIZES:
        # Try both years, take the first one found
        for year in [2018, 2024]:
            npz_path = data_dir / f"patches_{year}_{ps}x{ps}.npz"
            if not npz_path.exists():
                continue

            print(f"Loading scaler from: {npz_path}")
            data = np.load(npz_path, allow_pickle=True)

            scaler_min = data['scaler_min']
            scaler_max = data['scaler_max']

            out_path = data_dir / f"scaler_stats_{ps}x{ps}.npz"
            np.savez_compressed(
                out_path,
                scaler_min=scaler_min,
                scaler_max=scaler_max,
                band_names=BAND_NAMES,
                patch_size=ps,
            )

            size_kb = out_path.stat().st_size / 1024
            print(f"  Saved: {out_path} ({size_kb:.1f} KB)")
            print(f"  scaler_min shape: {scaler_min.shape}")
            print(f"  scaler_max shape: {scaler_max.shape}")
            data.close()
            break
        else:
            print(f"  WARNING: No patches NPZ found for patch_size={ps}")

    print("\nDone! Upload these tiny .npz files to Google Drive.")


if __name__ == '__main__':
    main()
