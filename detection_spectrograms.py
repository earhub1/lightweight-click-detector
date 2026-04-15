"""Generate spectrogram images for windows that contain click detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt, spectrogram
from soundfile import info as sf_info
from soundfile import read as sf_read

import plot_setup
from event_detection import combine_odfs_additive, compute_z_contrast, extract_odfs

REQUIRED_COLUMNS = {"time_sec", "high_click_events", "low_click_events"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-window focused-band spectrogram plots and "
            "diagnostics using the same Z-score logic as event_detection.py."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to detection CSV.")
    parser.add_argument("--wav", type=Path, required=True, help="Path to source WAV file.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory where images and manifest.csv are written.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=10.0,
        help="Window duration in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=10.0,
        help="Window hop in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--only-with-events",
        dest="only_with_events",
        action="store_true",
        help="Export only windows containing at least one event (default behavior).",
    )
    parser.add_argument(
        "--include-empty-windows",
        dest="only_with_events",
        action="store_false",
        help="Also export windows without events.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Image DPI for saved spectrograms (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="Image format/extension for output files (default: %(default)s).",
    )
    parser.add_argument(
        "--band-low-hz",
        type=float,
        default=20_000.0,
        help="Lower bound (Hz) for focused band subplot (default: %(default)s).",
    )
    parser.add_argument(
        "--band-high-hz",
        type=float,
        default=50_000.0,
        help="Upper bound (Hz) for focused band subplot (default: %(default)s).",
    )
    parser.add_argument(
        "--short-term-duration",
        type=float,
        default=0.001,
        help="Short-term duration used by event_detection.py to compute ODFs and Z.",
    )
    parser.add_argument("--w-hfc", type=float, default=0.35, help="Peso do dHFC.")
    parser.add_argument("--w-sf", type=float, default=0.35, help="Peso do SF.")
    parser.add_argument("--w-wpd", type=float, default=0.15, help="Peso do WPD.")
    parser.add_argument("--w-cd", type=float, default=0.15, help="Peso do CD.")
    parser.add_argument(
        "--norm-percentile",
        type=float,
        default=95.0,
        help="Percentil usado na normalização robusta do detector.",
    )
    parser.add_argument(
        "--z-mode",
        type=str,
        choices=["raw", "contrast", "hybrid"],
        default="raw",
        help=(
            "Modo do score Z — deve ser igual ao usado na detecção: "
            "raw, contrast ou hybrid (padrão: %(default)s)."
        ),
    )
    parser.add_argument(
        "--contrast-win-ms",
        type=float,
        default=100.0,
        help="Janela de contraste local (ms) — igual ao detector (padrão: %(default)s).",
    )
    parser.add_argument(
        "--contrast-alpha",
        type=float,
        default=0.5,
        help="Peso do Z raw no modo hybrid — igual ao detector (padrão: %(default)s).",
    )
    parser.add_argument(
        "--z-smooth-mode",
        type=str,
        choices=["none", "mean", "median"],
        default="none",
        help="Modo de suavização aplicado ao Z usado apenas para visualização.",
    )
    parser.add_argument(
        "--z-smooth-size",
        type=int,
        default=5,
        help="Tamanho da janela de suavização do Z (em frames). Use valores ímpares para median.",
    )
    parser.add_argument(
        "--thr-low",
        type=float,
        default=None,
        help="Limiar low fixo para exibir como linha horizontal nos plots do Z.",
    )
    parser.add_argument(
        "--thr-high",
        type=float,
        default=None,
        help="Limiar high fixo para exibir como linha horizontal nos plots do Z.",
    )
    parser.add_argument(
        "--window-manifest",
        type=Path,
        default=None,
        help=(
            "Caminho para o _window_manifest.csv gerado pelo detector. "
            "Quando fornecido, lê low_thr_used/high_thr_used por janela "
            "e os exibe como linhas de threshold nos plots (tem precedência sobre --thr-low/--thr-high)."
        ),
    )
    parser.set_defaults(only_with_events=True)
    return parser.parse_args()


def validate_csv_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {missing_str}")


def collect_candidate_windows(
    df: pd.DataFrame,
    *,
    hop_sec: float,
    window_sec: float,
    only_with_events: bool,
) -> list[int]:
    event_mask = (df["high_click_events"] == 1) | (df["low_click_events"] == 1)

    if only_with_events:
        event_times = df.loc[event_mask, "time_sec"].to_numpy(dtype=float)
        if event_times.size == 0:
            return []
        window_indices = np.floor(event_times / hop_sec).astype(int)
        return sorted(set(window_indices.tolist()))

    max_time = float(df["time_sec"].max()) if not df.empty else 0.0
    if max_time <= 0:
        return [0]

    num_windows = int(np.floor(max_time / hop_sec)) + 1
    return list(range(num_windows))


def event_times_in_window(
    df: pd.DataFrame,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    in_window = (df["time_sec"] >= start_sec) & (df["time_sec"] < end_sec)
    high_times = df.loc[in_window & (df["high_click_events"] == 1), "time_sec"].to_numpy(dtype=float)
    low_times = df.loc[in_window & (df["low_click_events"] == 1), "time_sec"].to_numpy(dtype=float)
    return high_times, low_times


def read_audio_window(wav_path: Path, start_sec: float, window_sec: float) -> tuple[np.ndarray, int]:
    wav_meta = sf_info(wav_path)
    fs = int(wav_meta.samplerate)

    start_sample = int(round(start_sec * fs))
    num_samples = int(round(window_sec * fs))

    data, _ = sf_read(wav_path, start=start_sample, frames=num_samples, always_2d=True)
    if data.size == 0:
        return np.array([], dtype=np.float32), fs

    audio = data[:, 0]
    return audio, fs


def format_window_tag(seconds_value: float) -> str:
    millis = int(round(seconds_value * 1000))
    return f"{millis:010d}ms"


def serialize_times(values: Iterable[float]) -> str:
    values_rounded = [round(float(v), 6) for v in values]
    return json.dumps(values_rounded, ensure_ascii=False)


def _safe_minmax_for_plot(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    min_x = np.min(x)
    max_x = np.max(x)
    denom = max_x - min_x
    if denom <= np.finfo(float).eps:
        return np.zeros_like(x, dtype=float)
    return (x - min_x) / denom


def _smooth_z(
    z: np.ndarray,
    mode: str = "none",
    size: int = 5,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.size == 0 or mode == "none":
        return z

    size = max(1, int(size))

    if mode == "mean":
        return uniform_filter1d(z, size=size, mode="nearest")

    if mode == "median":
        if size % 2 == 0:
            size += 1
        return medfilt(z, kernel_size=size)

    return z


def plot_window_spectrogram(
    *,
    audio: np.ndarray,
    fs: int,
    start_sec: float,
    end_sec: float,
    output_file: Path,
    title_prefix: str,
    dpi: int,
    band_low_hz: float,
    band_high_hz: float,
    rms: float,
    peak: float,
    duration_sec_real: float,
    high_times_abs: np.ndarray,
    low_times_abs: np.ndarray,
    short_term_duration: float,
    w_hfc: float = 0.35,
    w_sf: float = 0.35,
    w_wpd: float = 0.15,
    w_cd: float = 0.15,
    norm_percentile: float = 95.0,
    z_mode: str = "raw",
    contrast_win_ms: float = 100.0,
    contrast_alpha: float = 0.5,
    z_smooth_mode: str = "none",
    z_smooth_size: int = 5,
    low_thr: float | None = None,
    high_thr: float | None = None,
) -> dict[str, float]:
    """
    Gera espectrograma focado + curvas diagnósticas usando o mesmo Z do detector.

    Subplots:
    1. Espectrograma focado
    2. dHFC
    3. SF
    4. WPD
    5. CD
    6. Z raw
    7. Z used (z_mode ativo — idêntico ao detector)
    8. Z smooth (suavização visual de Z used)
    9. Eventos detectados (linhas verticais)

    Linhas horizontais de threshold (gold=low, crimson=high) são desenhadas
    nos subplots Z used e Z smooth quando low_thr/high_thr são fornecidos.
    """
    if audio.size == 0:
        return {
            "spec_db_p95": float("nan"),
            "spec_db_median": float("nan"),
            "snr_db_approx": float("nan"),
            "z_raw_max": float("nan"),
            "z_raw_p95": float("nan"),
        }

    nperseg_vis = min(1024, audio.size)
    noverlap_vis = int(0.5 * nperseg_vis)

    freqs, times, sxx = spectrogram(
        audio,
        fs=fs,
        nperseg=nperseg_vis,
        noverlap=noverlap_vis,
        scaling="density",
        mode="magnitude",
    )

    sxx_db = 10 * np.log10(sxx + np.finfo(float).eps)

    spec_db_p95 = float(np.percentile(sxx_db, 95))
    spec_db_median = float(np.median(sxx_db))
    noise_floor_db = float(np.percentile(sxx_db, 5))
    snr_db_approx = spec_db_p95 - noise_floor_db

    dHFC, SF, WPD, CD = extract_odfs(
        audio,
        fs,
        short_term_duration,
        flow=band_low_hz,
        fhig=band_high_hz,
    )

    z_raw = combine_odfs_additive(
        dHFC,
        SF,
        WPD,
        CD,
        w_hfc=w_hfc,
        w_sf=w_sf,
        w_wpd=w_wpd,
        w_cd=w_cd,
        norm_percentile=norm_percentile,
    )

    z_contrast = compute_z_contrast(
        z_raw,
        frame_dt=short_term_duration,
        contrast_win_ms=contrast_win_ms,
    )
    if z_mode == "contrast":
        z_used = z_contrast
    elif z_mode == "hybrid":
        alpha = float(np.clip(contrast_alpha, 0.0, 1.0))
        z_used = alpha * z_raw + (1.0 - alpha) * z_contrast
    else:
        z_used = z_raw

    t_det = (np.arange(len(z_raw)) + 0.5) * short_term_duration

    z_smooth = _smooth_z(z_used, mode=z_smooth_mode, size=z_smooth_size)
    z_smooth_scaled = _safe_minmax_for_plot(z_smooth)

    high_times_local = np.asarray(high_times_abs, dtype=float) - start_sec
    low_times_local = np.asarray(low_times_abs, dtype=float) - start_sec

    fig, axes = plt.subplots(
        9,
        1,
        figsize=(12, 15),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1, 1, 1, 1, 1, 0.8]},
    )
    ax_band, ax_hfc, ax_sf, ax_wpd, ax_cd, ax_z_raw, ax_z_used, ax_z_smooth, ax_events = axes

    band_mask_spec = (freqs >= band_low_hz) & (freqs <= band_high_hz)
    if np.any(band_mask_spec):
        ax_band.pcolormesh(
            times,
            freqs[band_mask_spec] / 1000.0,
            sxx_db[band_mask_spec, :],
            shading="auto",
            cmap="magma",
        )
        ax_band.set_ylabel("Banda (kHz)")
        ax_band.set_title(f"Banda focada: {band_low_hz/1000:.1f}–{band_high_hz/1000:.1f} kHz")
    else:
        ax_band.text(
            0.5,
            0.5,
            "Sem bins na banda selecionada",
            transform=ax_band.transAxes,
            ha="center",
            va="center",
        )
        ax_band.set_ylabel("Banda (kHz)")

    ax_hfc.plot(t_det, dHFC, color="tab:blue", linewidth=1.0)
    ax_hfc.set_ylabel("dHFC")
    ax_hfc.grid(alpha=0.25, linewidth=0.5)

    ax_sf.plot(t_det, SF, color="tab:orange", linewidth=1.0)
    ax_sf.set_ylabel("SF")
    ax_sf.grid(alpha=0.25, linewidth=0.5)

    ax_wpd.plot(t_det, WPD, color="tab:green", linewidth=1.0)
    ax_wpd.set_ylabel("WPD")
    ax_wpd.grid(alpha=0.25, linewidth=0.5)

    ax_cd.plot(t_det, CD, color="tab:red", linewidth=1.0)
    ax_cd.set_ylabel("CD")
    ax_cd.grid(alpha=0.25, linewidth=0.5)

    ax_z_raw.plot(t_det, z_raw, color="tab:gray", linewidth=1.0)
    ax_z_raw.set_ylabel("Z raw")
    ax_z_raw.grid(alpha=0.25, linewidth=0.5)

    ax_z_used.plot(t_det, z_used, color="tab:purple", linewidth=1.2)
    ax_z_used.set_ylabel(f"Z {z_mode}")
    ax_z_used.grid(alpha=0.25, linewidth=0.5)

    if low_thr is not None:
        ax_z_used.axhline(
            low_thr,
            color="gold",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
            label=f"low_thr={low_thr:.4f}",
        )
    if high_thr is not None:
        ax_z_used.axhline(
            high_thr,
            color="crimson",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
            label=f"high_thr={high_thr:.4f}",
        )
    if low_thr is not None or high_thr is not None:
        ax_z_used.legend(fontsize=7, loc="upper right")

    ax_z_smooth.plot(t_det, z_smooth, color="tab:brown", linewidth=1.2)
    ax_z_smooth.set_ylabel("Z smooth")
    ax_z_smooth.grid(alpha=0.25, linewidth=0.5)
    
    if low_thr is not None:
        ax_z_smooth.axhline(
            low_thr,
            color="gold",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )
    
    if high_thr is not None:
        ax_z_smooth.axhline(
            high_thr,
            color="crimson",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )

    for ax in ( ax_z_used, ax_z_smooth):
        """
        for t in low_times_local:
            if 0.0 <= t <= (end_sec - start_sec):
                ax.axvline(t, color="green", linewidth=0.8, alpha=0.6)
        """
        for t in high_times_local:
            if 0.0 <= t <= (end_sec - start_sec):
                ax.axvline(t, color="red", linewidth=0.8, alpha=0.6)

    for t in low_times_local:
        if 0.0 <= t <= (end_sec - start_sec):
            ax_events.axvline(t, color="green", linewidth=1.0, alpha=0.9)
    for t in high_times_local:
        if 0.0 <= t <= (end_sec - start_sec):
            ax_events.axvline(t, color="red", linewidth=1.0, alpha=0.9)

    ax_events.set_ylabel("Eventos")
    ax_events.set_xlabel("Tempo local da janela (s)")
    ax_events.set_ylim(0, 1)
    ax_events.set_yticks([])
    ax_events.grid(alpha=0.25, linewidth=0.5)

    ax_band.set_xlim(0.0, end_sec - start_sec)

    thr_info = ""
    if low_thr is not None or high_thr is not None:
        lo_str = f"{low_thr:.4f}" if low_thr is not None else "—"
        hi_str = f"{high_thr:.4f}" if high_thr is not None else "—"
        thr_info = f" | thr=({lo_str}, {hi_str})"

    fig.suptitle(
        (
            f"{title_prefix} | janela [{start_sec:.3f}, {end_sec:.3f}) s | "
            f"rms={rms:.5f} | peak={peak:.5f} | duração={duration_sec_real:.3f}s | "
            f"p95={spec_db_p95:.2f} dB | mediana={spec_db_median:.2f} dB | "
            f"snr≈{snr_db_approx:.2f} dB | "
            f"z_mode={z_mode} | "
            f"pesos=({w_hfc:.2f}, {w_sf:.2f}, {w_wpd:.2f}, {w_cd:.2f}) | "
            f"norm_p={norm_percentile:.1f} | "
            f"short_term={short_term_duration:.6f}s | "
            f"z_smooth=({z_smooth_mode}, {z_smooth_size})"
            f"{thr_info}"
        ),
        fontsize=9,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "spec_db_p95": spec_db_p95,
        "spec_db_median": spec_db_median,
        "snr_db_approx": snr_db_approx,
        "z_raw_max": float(np.max(z_raw)) if z_raw.size else float("nan"),
        "z_raw_p95": float(np.percentile(z_raw, 95)) if z_raw.size else float("nan"),
    }


def main() -> None:
    args = parse_args()

    if args.window_sec <= 0:
        raise ValueError("--window-sec must be > 0")
    if args.hop_sec <= 0:
        raise ValueError("--hop-sec must be > 0")

    plot_setup.configure_plots()
    plt.rcParams["text.usetex"] = False

    df = pd.read_csv(args.csv)
    validate_csv_columns(df)
    df = df.sort_values("time_sec").reset_index(drop=True)

    manifest_win_df: pd.DataFrame | None = None
    if args.window_manifest is not None:
        manifest_win_df = pd.read_csv(args.window_manifest)

    window_indices = collect_candidate_windows(
        df,
        hop_sec=args.hop_sec,
        window_sec=args.window_sec,
        only_with_events=args.only_with_events,
    )

    wav_meta = sf_info(args.wav)
    fs_file = int(wav_meta.samplerate)
    if not (0 < args.band_low_hz < args.band_high_hz <= fs_file / 2):
        raise ValueError(
            "--band-low-hz and --band-high-hz must satisfy: "
            "0 < band-low-hz < band-high-hz <= fs/2"
        )

    audio_duration = float(wav_meta.duration)
    output_dir = args.output.resolve() / args.wav.stem
    manifest_rows: list[dict] = []

    for window_idx in window_indices:
        start_sec = float(window_idx * args.hop_sec)
        end_sec = start_sec + float(args.window_sec)

        if start_sec >= audio_duration:
            continue

        high_times_abs, low_times_abs = event_times_in_window(df, start_sec, end_sec)

        if args.only_with_events and high_times_abs.size == 0 and low_times_abs.size == 0:
            continue

        audio, fs = read_audio_window(args.wav, start_sec, args.window_sec)
        if audio.size == 0:
            continue

        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        duration_sec_real = float(audio.size / fs)

        low_thr_plot: float | None = args.thr_low
        high_thr_plot: float | None = args.thr_high

        if (
            manifest_win_df is not None
            and "low_thr_used" in manifest_win_df.columns
            and "high_thr_used" in manifest_win_df.columns
        ):
            row_mask = np.isclose(
                manifest_win_df["window_start_sec"].to_numpy(dtype=float),
                start_sec,
                atol=0.01,
            )
            if row_mask.any():
                mrow = manifest_win_df.loc[row_mask].iloc[0]
                low_thr_plot = float(mrow["low_thr_used"])
                high_thr_plot = float(mrow["high_thr_used"])

        file_name = (
            f"{args.wav.stem}__wstart_{format_window_tag(start_sec)}"
            f"__wend_{format_window_tag(end_sec)}.{args.format.lstrip('.')}"
        )
        image_path = output_dir / file_name

        spec_stats = plot_window_spectrogram(
            audio=audio,
            fs=fs,
            start_sec=start_sec,
            end_sec=end_sec,
            output_file=image_path,
            title_prefix=args.wav.stem,
            dpi=args.dpi,
            band_low_hz=args.band_low_hz,
            band_high_hz=args.band_high_hz,
            rms=rms,
            peak=peak,
            duration_sec_real=duration_sec_real,
            high_times_abs=high_times_abs,
            low_times_abs=low_times_abs,
            short_term_duration=args.short_term_duration,
            w_hfc=args.w_hfc,
            w_sf=args.w_sf,
            w_wpd=args.w_wpd,
            w_cd=args.w_cd,
            norm_percentile=args.norm_percentile,
            z_mode=args.z_mode,
            contrast_win_ms=args.contrast_win_ms,
            contrast_alpha=args.contrast_alpha,
            z_smooth_mode=args.z_smooth_mode,
            z_smooth_size=args.z_smooth_size,
            low_thr=low_thr_plot,
            high_thr=high_thr_plot,
        )

        manifest_rows.append(
            {
                "image_path": str(image_path.resolve()),
                "wav_path": str(args.wav.resolve()),
                "window_start_sec": start_sec,
                "window_end_sec": end_sec,
                "num_high": int(high_times_abs.size),
                "num_low": int(low_times_abs.size),
                "high_times_sec": serialize_times(high_times_abs.tolist()),
                "low_times_sec": serialize_times(low_times_abs.tolist()),
                "rms": rms,
                "peak": peak,
                "duration_sec_real": duration_sec_real,
                "spec_db_p95": spec_stats["spec_db_p95"],
                "spec_db_median": spec_stats["spec_db_median"],
                "snr_db_approx": spec_stats["snr_db_approx"],
                "z_raw_max": spec_stats["z_raw_max"],
                "z_raw_p95": spec_stats["z_raw_p95"],
            }
        )

    manifest_df = pd.DataFrame(
        manifest_rows,
        columns=[
            "image_path",
            "wav_path",
            "window_start_sec",
            "window_end_sec",
            "num_high",
            "num_low",
            "high_times_sec",
            "low_times_sec",
            "rms",
            "peak",
            "duration_sec_real",
            "spec_db_p95",
            "spec_db_median",
            "snr_db_approx",
            "z_raw_max",
            "z_raw_p95",
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
