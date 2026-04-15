"""
event_detection.py

Detecta eventos de clique em lotes de arquivos de áudio (.wav) usando
funções de detecção de onset (ODFs) extraídas por STFT em resolução curta.

Fluxo:
1. Localiza arquivos de áudio no diretório de entrada.
2. Para cada arquivo, processa o sinal em janelas de médio prazo.
3. Em cada janela, extrai as ODFs:
   - dHFC
   - Spectral Flux (SF)
   - Weighted Phase Deviation (WPD)
   - Complex Domain (CD)
4. Combina as ODFs em um score composto z por fusão aditiva após
   normalização robusta por percentil.
5. Aplica limiares fixos ou adaptativos sobre z.
6. Salva (append streaming em CSVs consolidados no diretório de saída):
   - all_events.csv  — eventos de todos os áudios, com coluna audio_filepath
   - all_chains.csv  — cadeias de cliques de todos os áudios, com coluna audio_filepath
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import ShortTimeFFT, windows

import core.signal_processing as sp

logger = logging.getLogger(__name__)


def setup_logging(log_stats: bool) -> None:
    level = logging.INFO if log_stats else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def append_to_consolidated(
    df: pd.DataFrame,
    dest_path: Path,
    audio_filepath: str,
) -> None:
    if df.empty:
        return
    out = df.copy()
    out.insert(0, "audio_filepath", audio_filepath)
    write_header = not (dest_path.exists() and dest_path.stat().st_size > 0)
    with open(dest_path, "a", encoding="utf-8", newline="") as f:
        out.to_csv(f, index=False, header=write_header)
    logger.debug("Appended %d rows to %s", len(out), dest_path)


def robust_percentile_normalize(
    x: np.ndarray,
    q: float = 95.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normaliza um vetor 1D usando um percentil alto como referência de escala.
    """
    x = np.asarray(x, dtype=float)
    ref = float(np.percentile(x, q))
    return x / (ref + eps)


def combine_odfs_additive(
    dHFC: np.ndarray,
    SF: np.ndarray,
    WPD: np.ndarray,
    CD: np.ndarray,
    *,
    w_hfc: float = 0.35,
    w_sf: float = 0.35,
    w_wpd: float = 0.15,
    w_cd: float = 0.15,
    norm_percentile: float = 95.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Combina as ODFs por soma ponderada após normalização robusta.
    """
    dHFC_n = robust_percentile_normalize(dHFC, q=norm_percentile, eps=eps)
    SF_n = robust_percentile_normalize(SF, q=norm_percentile, eps=eps)
    WPD_n = robust_percentile_normalize(WPD, q=norm_percentile, eps=eps)
    CD_n = robust_percentile_normalize(CD, q=norm_percentile, eps=eps)

    z = (
        w_hfc * dHFC_n
        + w_sf * SF_n
        + w_wpd * WPD_n
        + w_cd * CD_n
    )
    return z


def compute_adaptive_threshold_chunk(
    z: np.ndarray,
    *,
    k_low: float = 3.0,
    k_high: float = 6.0,
    low_thr_floor: float = 1e-5,
    high_thr_floor: float = 5e-5,
    trim_top: float = 0.02,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """
    Calcula limiares adaptativos locais para um chunk via mediana + MAD.

    Remove a cauda superior (trim_top) antes de estimar o ruído de fundo,
    tornando a estimativa robusta a cliques densos dentro da própria janela.
    """
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return float(low_thr_floor), float(high_thr_floor)

    z_sorted = np.sort(z)
    cut = max(1, int(np.floor((1.0 - trim_top) * len(z_sorted))))
    z_bg = z_sorted[:cut]

    bg_med = float(np.median(z_bg))
    bg_mad = 1.4826 * float(np.median(np.abs(z_bg - bg_med)))

    low_thr = max(bg_med + k_low * bg_mad, float(low_thr_floor))
    high_thr = max(bg_med + k_high * bg_mad, float(high_thr_floor))

    if high_thr <= low_thr:
        high_thr = low_thr + eps

    return low_thr, high_thr


def compute_z_contrast(
    z: np.ndarray,
    *,
    frame_dt: float,
    contrast_win_ms: float = 100.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute local robust contrast for z using local median and MAD.
    """
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return z

    frame_dt = max(float(frame_dt), eps)
    win_frames = max(1, int(round((contrast_win_ms / 1000.0) / frame_dt)))
    if win_frames % 2 == 0:
        win_frames += 1

    z_local_med = median_filter(z, size=win_frames, mode="nearest")
    abs_dev = np.abs(z - z_local_med)
    z_local_mad = median_filter(abs_dev, size=win_frames, mode="nearest")
    z_contrast = (z - z_local_med) / (z_local_mad + eps)
    return z_contrast


def detect_onsets(z: np.ndarray, high_thr: float, low_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta eventos high e low a partir do score z.
    """
    high_click = (z >= high_thr).astype(int)
    low_click = ((z >= low_thr) & (z < high_thr)).astype(int)
    return high_click, low_click


def extract_odfs(
    sound_data: np.ndarray,
    fs: int,
    time_resolution: float,
    *,
    flow: float = 75e3,
    fhig: float = 125e3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrai ODFs a partir da STFT unilateral.
    """
    num_samples = len(sound_data)
    W = int(time_resolution * fs)
    nfft = W

    if W <= 0:
        raise ValueError("short_term_duration * fs must be >= 1 sample.")

    window = windows.hann(W, sym=False)

    stft_transform = ShortTimeFFT(
        mfft=nfft,
        hop=W,
        win=window,
        fft_mode="onesided",
        fs=fs,
    )
    STFT_oneside = stft_transform.stft(sound_data, p0=0, p1=num_samples // W, k_offset=0)
    Sxx = np.abs(STFT_oneside)

    freqs_onesided = np.linspace(0, fs / 2, nfft // 2 + 1)

    nyquist = fs / 2
    if fhig > nyquist:
        logger.warning(
            "fhig=%.1f Hz > Nyquist=%.1f Hz. Clamping fhig to Nyquist.",
            fhig,
            nyquist,
        )
        fhig = nyquist

    if flow >= fhig:
        raise ValueError(
            f"Invalid frequency band: flow={flow:.1f} Hz must be lower than fhig={fhig:.1f} Hz."
        )

    dHFC = sp.compute_hfc(Sxx, freqs_onesided, flow, fhig)
    SF = sp.compute_spectral_flux(Sxx, freqs_onesided, flow, fhig)
    WPD = sp.compute_wpd(STFT_oneside)
    CD = sp.compute_cd(STFT_oneside)

    return dHFC, SF, WPD, CD, Sxx, freqs_onesided


def compute_window_metrics(
    sound_data: np.ndarray,
    z: np.ndarray,
    high_click: np.ndarray,
    low_click: np.ndarray,
    *,
    mid_term_duration: float,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Calcula métricas por janela para qualificação acústica.
    """
    audio = np.asarray(sound_data, dtype=float)

    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    crest_factor = float(peak / (rms + eps)) if audio.size else 0.0

    z_raw_mean = float(np.mean(z)) if z.size else 0.0
    z_raw_std = float(np.std(z)) if z.size else 0.0
    z_raw_p95 = float(np.percentile(z, 95)) if z.size else 0.0
    z_raw_max = float(np.max(z)) if z.size else 0.0
    peak_to_p95_ratio = float(z_raw_max / (z_raw_p95 + eps)) if z.size else 0.0

    num_high = int(np.sum(high_click))
    num_low = int(np.sum(low_click))
    event_rate = float((num_high + num_low) / max(mid_term_duration, eps))

    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "z_raw_mean": z_raw_mean,
        "z_raw_std": z_raw_std,
        "z_raw_p95": z_raw_p95,
        "z_raw_max": z_raw_max,
        "peak_to_p95_ratio": peak_to_p95_ratio,
        "num_high": num_high,
        "num_low": num_low,
        "event_rate": event_rate,
    }


def _group_contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index runs where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for cur in idx[1:]:
        cur = int(cur)
        if cur == prev + 1:
            prev = cur
            continue
        runs.append((start, prev))
        start = cur
        prev = cur
    runs.append((start, prev))
    return runs


def _event_from_run(
    run: tuple[int, int],
    event_type: str,
    time_sec: np.ndarray,
    frame_dt: float,
    z_all: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Build a single event dictionary from an inclusive run."""
    start_idx, end_idx = run
    peak_idx = start_idx

    if z_all is not None and len(z_all) > 0:
        z_slice = z_all[start_idx:end_idx + 1]
        if z_slice.size > 0:
            peak_rel = int(np.argmax(z_slice))
            peak_idx = start_idx + peak_rel
            peak_z = float(z_all[peak_idx])
        else:
            peak_z = float("nan")
    else:
        peak_z = float("nan")

    start_time = float(max(0.0, time_sec[start_idx] - 0.5 * frame_dt))
    end_time = float(time_sec[end_idx] + 0.5 * frame_dt)
    duration_ms = float(max(0.0, (end_time - start_time) * 1000.0))

    return {
        "event_type": event_type,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_sec": start_time,
        "end_sec": end_time,
        "duration_ms": duration_ms,
        "peak_idx": peak_idx,
        "peak_time_sec": float(time_sec[peak_idx]),
        "peak_z": peak_z,
    }


def _merge_close_events(
    events: list[dict[str, float | str]],
    merge_gap_sec: float,
    time_sec: np.ndarray,
    z_all: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    """Merge adjacent events if temporal gap is below threshold."""
    if len(events) <= 1:
        return events

    merged: list[dict[str, float | str]] = [events[0].copy()]
    for event in events[1:]:
        current = merged[-1]
        gap = float(event["start_sec"]) - float(current["end_sec"])
        if gap <= merge_gap_sec:
            current["end_idx"] = int(event["end_idx"])
            current["end_sec"] = float(event["end_sec"])
            current["duration_ms"] = float((float(current["end_sec"]) - float(current["start_sec"])) * 1000.0)

            if z_all is not None and len(z_all) > 0:
                old_peak_idx = int(current["peak_idx"])
                new_peak_idx = int(event["peak_idx"])
                old_peak_z = float(z_all[old_peak_idx])
                new_peak_z = float(z_all[new_peak_idx])
                if new_peak_z > old_peak_z:
                    current["peak_idx"] = new_peak_idx
                    current["peak_time_sec"] = float(time_sec[new_peak_idx])
                    current["peak_z"] = new_peak_z
                else:
                    current["peak_z"] = old_peak_z
            continue

        merged.append(event.copy())

    return merged


def _apply_refractory(
    events: list[dict[str, float | str]],
    refractory_sec: float,
    z_all: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    """Apply refractory period by keeping the strongest (or earliest) event."""
    if len(events) <= 1 or refractory_sec <= 0:
        return events

    kept: list[dict[str, float | str]] = [events[0].copy()]
    for event in events[1:]:
        prev = kept[-1]
        dt_start = float(event["start_sec"]) - float(prev["start_sec"])
        if dt_start >= refractory_sec:
            kept.append(event.copy())
            continue

        if z_all is not None and len(z_all) > 0:
            prev_peak_z = float(z_all[int(prev["peak_idx"])])
            curr_peak_z = float(z_all[int(event["peak_idx"])])
            if curr_peak_z > prev_peak_z:
                kept[-1] = event.copy()
        # sem z_all, mantém o mais antigo

    return kept


def _events_to_mask(events: list[dict[str, float | str]], n_frames: int) -> np.ndarray:
    """Convert event spans back to a binary frame mask."""
    mask = np.zeros(n_frames, dtype=int)
    for event in events:
        start_idx = int(event["start_idx"])
        end_idx = int(event["end_idx"])
        mask[start_idx:end_idx + 1] = 1
    return mask


def _build_events_from_mask(
    mask: np.ndarray,
    *,
    event_type: str,
    time_sec: np.ndarray,
    frame_dt: float,
    z_all: np.ndarray | None,
    min_event_duration_ms: float,
    merge_gap_ms: float,
    refractory_ms: float,
) -> list[dict[str, float | str]]:
    """Build and postprocess event dictionaries from a binary mask."""
    merge_gap_sec = max(0.0, float(merge_gap_ms) / 1000.0)
    refractory_sec = max(0.0, float(refractory_ms) / 1000.0)
    min_event_duration_ms = max(0.0, float(min_event_duration_ms))

    runs = _group_contiguous_runs(mask)
    events = [_event_from_run(run, event_type, time_sec, frame_dt, z_all=z_all) for run in runs]
    events = [ev for ev in events if float(ev["duration_ms"]) >= min_event_duration_ms]
    events = _merge_close_events(events, merge_gap_sec, time_sec, z_all=z_all)
    events = _apply_refractory(events, refractory_sec, z_all=z_all)
    return events


def postprocess_binary_masks(
    high_click_events: np.ndarray,
    low_click_events: np.ndarray,
    *,
    time_sec: np.ndarray,
    frame_dt: float,
    min_event_duration_ms: float,
    merge_gap_ms: float,
    refractory_ms: float,
    z_all: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Postprocess frame-level masks and return cleaned binary high/low arrays.
    """
    time_sec = np.asarray(time_sec, dtype=float)
    high_click_events = np.asarray(high_click_events, dtype=int)
    low_click_events = np.asarray(low_click_events, dtype=int)

    high_events = _build_events_from_mask(
        high_click_events == 1,
        event_type="high",
        time_sec=time_sec,
        frame_dt=frame_dt,
        z_all=z_all,
        min_event_duration_ms=min_event_duration_ms,
        merge_gap_ms=merge_gap_ms,
        refractory_ms=refractory_ms,
    )
    low_events = _build_events_from_mask(
        low_click_events == 1,
        event_type="low",
        time_sec=time_sec,
        frame_dt=frame_dt,
        z_all=z_all,
        min_event_duration_ms=min_event_duration_ms,
        merge_gap_ms=merge_gap_ms,
        refractory_ms=refractory_ms,
    )

    high_pp = _events_to_mask(high_events, len(time_sec))
    low_pp = _events_to_mask(low_events, len(time_sec))
    low_pp[high_pp == 1] = 0
    return high_pp, low_pp


def compute_spectral_descriptors(
    Sxx_all: np.ndarray,
    freqs: np.ndarray,
    peak_idx: int,
    *,
    flow: float,
    fhig: float,
    n_bands: int = 10,
    eps: float = 1e-12,
) -> dict[str, float]:
    nan_result: dict[str, float] = {
        "dominant_freq_hz": float("nan"),
        "dominant_amp": float("nan"),
        "bw_3db_hz": float("nan"),
        "bw_10db_hz": float("nan"),
    }
    for i in range(1, n_bands + 1):
        nan_result[f"band_energy_{i}"] = float("nan")
        nan_result[f"band_energy_rel_{i}"] = float("nan")

    n_frames = Sxx_all.shape[1]
    peak_idx = int(np.clip(peak_idx, 0, n_frames - 1))
    lo = max(0, peak_idx - 1)
    hi = min(n_frames, peak_idx + 2)
    spec = Sxx_all[:, lo:hi].mean(axis=1)

    band_mask = (freqs >= flow) & (freqs <= fhig)
    spec_band = spec[band_mask]
    freqs_band = freqs[band_mask]

    if spec_band.size == 0:
        return nan_result

    dom_idx = int(np.argmax(spec_band))
    dominant_freq_hz = float(freqs_band[dom_idx])
    dominant_amp = float(spec_band[dom_idx])

    spec_db = 20.0 * np.log10(spec_band + eps)
    peak_db = float(spec_db[dom_idx])

    def _bandwidth(drop_db: float) -> float:
        threshold = peak_db - drop_db
        left = float(freqs_band[dom_idx])
        for i in range(dom_idx, -1, -1):
            if spec_db[i] < threshold:
                left = float(freqs_band[i])
                break
        right = float(freqs_band[dom_idx])
        for i in range(dom_idx, len(spec_db)):
            if spec_db[i] < threshold:
                right = float(freqs_band[i])
                break
        return float(right - left)

    bw_3db = _bandwidth(3.0)
    bw_10db = _bandwidth(10.0)

    power_band = spec_band ** 2
    band_edges = np.linspace(flow, fhig, n_bands + 1)
    band_energies: list[float] = []
    for b in range(n_bands):
        mask_b = (freqs_band >= band_edges[b]) & (freqs_band < band_edges[b + 1])
        band_energies.append(float(np.sum(power_band[mask_b])))

    total_energy = sum(band_energies) + eps
    band_energies_rel = [e / total_energy for e in band_energies]

    result: dict[str, float] = {
        "dominant_freq_hz": dominant_freq_hz,
        "dominant_amp": dominant_amp,
        "bw_3db_hz": bw_3db,
        "bw_10db_hz": bw_10db,
    }
    for i, e in enumerate(band_energies, start=1):
        result[f"band_energy_{i}"] = e
    for i, e in enumerate(band_energies_rel, start=1):
        result[f"band_energy_rel_{i}"] = e
    return result


def enrich_events_with_spectral(
    df_events: pd.DataFrame,
    Sxx_all: np.ndarray,
    freqs: np.ndarray,
    *,
    flow: float,
    fhig: float,
    short_term_duration: float,
    n_bands: int = 10,
) -> pd.DataFrame:
    if df_events.empty or Sxx_all.size == 0:
        return df_events

    n_frames = Sxx_all.shape[1]
    records: list[dict[str, float]] = []
    for _, row in df_events.iterrows():
        peak_idx = int(np.clip(
            round(float(row["peak_time_sec"]) / short_term_duration),
            0, n_frames - 1,
        ))
        desc = compute_spectral_descriptors(
            Sxx_all, freqs, peak_idx,
            flow=flow, fhig=fhig, n_bands=n_bands,
        )
        records.append(desc)

    spectral_df = pd.DataFrame(records, index=df_events.index)
    return pd.concat([df_events, spectral_df], axis=1)


def postprocess_events(
    time_sec: np.ndarray,
    high_click_events: np.ndarray,
    low_click_events: np.ndarray,
    frame_dt: float,
    min_event_duration_ms: float,
    merge_gap_ms: float,
    refractory_ms: float,
    z_all: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Convert frame-level detections into robust event-level segments.
    """
    time_sec = np.asarray(time_sec, dtype=float)
    high_click_events = np.asarray(high_click_events, dtype=int)
    low_click_events = np.asarray(low_click_events, dtype=int)

    columns = [
        "event_id",
        "event_type",
        "start_sec",
        "end_sec",
        "duration_ms",
        "peak_time_sec",
        "peak_z",
    ]
    if time_sec.size == 0:
        return pd.DataFrame(columns=columns)

    frame_dt = float(frame_dt) if frame_dt > 0 else 0.001
    high_pp, low_pp = postprocess_binary_masks(
        high_click_events,
        low_click_events,
        time_sec=time_sec,
        frame_dt=frame_dt,
        min_event_duration_ms=min_event_duration_ms,
        merge_gap_ms=merge_gap_ms,
        refractory_ms=refractory_ms,
        z_all=z_all,
    )

    event_rows: list[dict[str, float | str]] = []
    for run in _group_contiguous_runs(high_pp == 1):
        event_rows.append(_event_from_run(run, "high", time_sec, frame_dt, z_all=z_all))
    for run in _group_contiguous_runs(low_pp == 1):
        event_rows.append(_event_from_run(run, "low", time_sec, frame_dt, z_all=z_all))

    if not event_rows:
        return pd.DataFrame(columns=columns)

    event_rows = sorted(event_rows, key=lambda r: (float(r["start_sec"]), str(r["event_type"])))
    output_rows: list[dict[str, float | str | int]] = []
    for idx, row in enumerate(event_rows, start=1):
        output_rows.append(
            {
                "event_id": idx,
                "event_type": row["event_type"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "duration_ms": row["duration_ms"],
                "peak_time_sec": row["peak_time_sec"],
                "peak_z": row["peak_z"],
            }
        )

    return pd.DataFrame(output_rows, columns=columns)


def _aggregate_spectral(
    chain_df: pd.DataFrame,
    *,
    flow: float,
    fhig: float,
    n_bands: int = 10,
    eps: float = 1e-12,
) -> dict[str, float]:
    agg: dict[str, float] = {}

    scalar_cols = {
        "dominant_freq_hz": True,
        "dominant_amp": False,
        "bw_3db_hz": False,
        "bw_10db_hz": False,
    }

    for col, full_stats in scalar_cols.items():
        if col in chain_df.columns:
            vals = chain_df[col].dropna().to_numpy(dtype=float)
        else:
            vals = np.array([])

        if vals.size > 0:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
        else:
            mean_val = float("nan")
            std_val = float("nan")

        agg[f"mean_{col}"] = mean_val
        agg[f"std_{col}"] = std_val

        if full_stats:
            cv = std_val / (mean_val + eps) if not np.isnan(mean_val) else float("nan")
            agg["cv_dominant_freq_hz"] = cv
            agg["min_dominant_freq_hz"] = float(np.min(vals)) if vals.size else float("nan")
            agg["max_dominant_freq_hz"] = float(np.max(vals)) if vals.size else float("nan")

    band_means: list[float] = []
    for i in range(1, n_bands + 1):
        col = f"band_energy_{i}"
        col_rel = f"band_energy_rel_{i}"

        if col in chain_df.columns:
            vals = chain_df[col].dropna().to_numpy(dtype=float)
            agg[f"mean_band_energy_{i}"] = float(np.mean(vals)) if vals.size else float("nan")
            band_means.append(agg[f"mean_band_energy_{i}"])
        else:
            agg[f"mean_band_energy_{i}"] = float("nan")
            band_means.append(float("nan"))

        if col_rel in chain_df.columns:
            vals_rel = chain_df[col_rel].dropna().to_numpy(dtype=float)
            agg[f"mean_band_energy_rel_{i}"] = float(np.mean(vals_rel)) if vals_rel.size else float("nan")
        else:
            agg[f"mean_band_energy_rel_{i}"] = float("nan")

    if not any(np.isnan(v) for v in band_means):
        band_width = (fhig - flow) / n_bands
        band_centers = np.array([flow + (i - 0.5) * band_width for i in range(1, n_bands + 1)])
        total = sum(band_means) + eps
        agg["spectral_centroid_hz"] = float(
            sum(c * e for c, e in zip(band_centers, band_means)) / total
        )
    else:
        agg["spectral_centroid_hz"] = float("nan")

    return agg


def build_click_chains(
    df_events: pd.DataFrame,
    *,
    min_clicks_in_chain: int = 5,
    max_gap_factor: float = 2.0,
    filepath: str | None = None,
    event_types: tuple[str, ...] = ("high",),
    flow: float = 75e3,
    fhig: float = 125e3,
) -> pd.DataFrame:
    """
    Agrupa eventos em cadeias de cliques com base em ICI e extrai
    features temporais e de intensidade por cadeia.

    Regras:
    - considera apenas os tipos listados em event_types
    - cadeia deve conter pelo menos min_clicks_in_chain cliques
    - um novo clique pertence à cadeia atual se:
          gap <= max_gap_factor * ici_medio_da_cadeia
      onde gap é a diferença entre peak_time_sec consecutivos
    """
    columns = [
        "filepath",
        "chain_id",
        "start_time",
        "end_time",
        "n_clicks",
        "duration_sec",
        "click_rate_hz",
        "mean_ici_ms",
        "std_ici_ms",
        "cv_ici",
        "median_ici_ms",
        "min_ici_ms",
        "max_ici_ms",
        "p25_ici_ms",
        "p75_ici_ms",
        "ici_trend_slope",
        "mean_peak_z",
        "std_peak_z",
        "cv_peak_z",
        "min_peak_z",
        "max_peak_z",
    ]

    if df_events.empty:
        return pd.DataFrame(columns=columns)

    df = df_events[df_events["event_type"].isin(event_types)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df.sort_values("peak_time_sec").reset_index(drop=True)

    chains_idx: list[list[int]] = []
    current_chain_idx = [0]

    for i in range(1, len(df)):
        current_time = float(df.loc[i, "peak_time_sec"])
        prev_time = float(df.loc[i - 1, "peak_time_sec"])
        gap = current_time - prev_time

        if len(current_chain_idx) >= 2:
            chain_times = df.loc[current_chain_idx, "peak_time_sec"].to_numpy(dtype=float)
            ici_chain = np.diff(chain_times)
            mean_ici = float(np.mean(ici_chain))
        else:
            mean_ici = gap

        if gap <= max_gap_factor * mean_ici:
            current_chain_idx.append(i)
        else:
            if len(current_chain_idx) >= min_clicks_in_chain:
                chains_idx.append(current_chain_idx)
            current_chain_idx = [i]

    if len(current_chain_idx) >= min_clicks_in_chain:
        chains_idx.append(current_chain_idx)

    rows = []

    for chain_id, idxs in enumerate(chains_idx, start=1):
        chain_df = df.loc[idxs].copy().reset_index(drop=True)

        chain_times = chain_df["peak_time_sec"].to_numpy(dtype=float)
        peak_z_vals = chain_df["peak_z"].to_numpy(dtype=float)

        icis = np.diff(chain_times)
        icis_ms = icis * 1000.0

        start_time = float(chain_times[0])
        end_time = float(chain_times[-1])
        duration_sec = float(max(0.0, end_time - start_time))
        n_clicks = int(len(chain_times))
        click_rate_hz = float(n_clicks / duration_sec) if duration_sec > 0 else float("nan")

        if icis_ms.size > 0:
            mean_ici_ms = float(np.mean(icis_ms))
            std_ici_ms = float(np.std(icis_ms))
            cv_ici = float(std_ici_ms / mean_ici_ms) if mean_ici_ms > 0 else float("nan")
            median_ici_ms = float(np.median(icis_ms))
            min_ici_ms = float(np.min(icis_ms))
            max_ici_ms = float(np.max(icis_ms))
            p25_ici_ms = float(np.percentile(icis_ms, 25))
            p75_ici_ms = float(np.percentile(icis_ms, 75))

            if icis_ms.size >= 2:
                x = np.arange(len(icis_ms), dtype=float)
                ici_trend_slope = float(np.polyfit(x, icis_ms, 1)[0])
            else:
                ici_trend_slope = float("nan")
        else:
            mean_ici_ms = float("nan")
            std_ici_ms = float("nan")
            cv_ici = float("nan")
            median_ici_ms = float("nan")
            min_ici_ms = float("nan")
            max_ici_ms = float("nan")
            p25_ici_ms = float("nan")
            p75_ici_ms = float("nan")
            ici_trend_slope = float("nan")

        if peak_z_vals.size > 0:
            mean_peak_z = float(np.mean(peak_z_vals))
            std_peak_z = float(np.std(peak_z_vals))
            cv_peak_z = float(std_peak_z / mean_peak_z) if mean_peak_z > 0 else float("nan")
            min_peak_z = float(np.min(peak_z_vals))
            max_peak_z = float(np.max(peak_z_vals))
        else:
            mean_peak_z = float("nan")
            std_peak_z = float("nan")
            cv_peak_z = float("nan")
            min_peak_z = float("nan")
            max_peak_z = float("nan")

        rows.append(
            {
                "filepath": filepath if filepath is not None else "",
                "chain_id": chain_id,
                "start_time": start_time,
                "end_time": end_time,
                "n_clicks": n_clicks,
                "duration_sec": duration_sec,
                "click_rate_hz": click_rate_hz,
                "mean_ici_ms": mean_ici_ms,
                "std_ici_ms": std_ici_ms,
                "cv_ici": cv_ici,
                "median_ici_ms": median_ici_ms,
                "min_ici_ms": min_ici_ms,
                "max_ici_ms": max_ici_ms,
                "p25_ici_ms": p25_ici_ms,
                "p75_ici_ms": p75_ici_ms,
                "ici_trend_slope": ici_trend_slope,
                "mean_peak_z": mean_peak_z,
                "std_peak_z": std_peak_z,
                "cv_peak_z": cv_peak_z,
                "min_peak_z": min_peak_z,
                "max_peak_z": max_peak_z,
                **_aggregate_spectral(chain_df, flow=flow, fhig=fhig),
            }
        )

    return pd.DataFrame(rows, columns=columns)

def process_audio_file(
    file_path: Path,
    *,
    mid_term_duration: float = 1.0,
    short_term_duration: float = 0.001,
    high_thr: float = 0.2,
    low_thr: float = 0.05,
    flow: float = 75e3,
    fhig: float = 125e3,
    log_stats: bool = False,
    thr_mode: str = "fixed",
    low_quantile: float = 0.95,
    high_quantile: float = 0.99,
    low_thr_floor: float = 1e-5,
    high_thr_floor: float = 5e-5,
    w_hfc: float = 0.35,
    w_sf: float = 0.35,
    w_wpd: float = 0.15,
    w_cd: float = 0.15,
    norm_percentile: float = 95.0,
    min_event_duration_ms: float = 1.0,
    merge_gap_ms: float = 1.0,
    refractory_ms: float = 2.0,
    z_mode: str = "raw",
    contrast_win_ms: float = 100.0,
    contrast_alpha: float = 0.5,
    bg_k_low: float = 3.0,
    bg_k_high: float = 6.0,
    smooth_beta: float = 0.8,
    penalty_event_rate_limit: float = 50.0,
    penalty_ratio_limit: float = 1.5,
    penalty_factor_high: float = 1.15,
    penalty_factor_low: float = 1.10,
    reward_factor: float = 0.95,
    rms_clean_limit: float = 0.01,
    ratio_good_limit: float = 3.0,
    low_thr_ratio: float = 0.75,
    contrast_low_thr: float = 2.5,
    disable_low: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Processa um arquivo e retorna:
    1. DataFrame por frame com eventos low/high
    2. DataFrame por janela com métricas de qualidade acústica
    3. DataFrame por evento com segmentos robustos (event-level)
    """
    print(f"Processing file: {file_path}")
    if z_mode not in {"raw", "contrast", "hybrid"}:
        raise ValueError("z_mode must be one of: raw, contrast, hybrid.")
    if z_mode in {"contrast", "hybrid"} and thr_mode == "fixed":
        logger.warning(
            "Using z_mode=%s with fixed thresholds may require recalibration. "
            "For first validation prefer thr_mode='adaptive'.",
            z_mode,
        )

    fs = sp.get_sample_rate(file_path)
    num_samples_mid = int(mid_term_duration * fs)

    eps = 1e-12

    time_sec_list: list[float] = []
    z_raw_chunks: list[np.ndarray] = []
    z_used_chunks: list[np.ndarray] = []
    sound_chunks: list[np.ndarray] = []
    Sxx_chunks: list[np.ndarray] = []
    freqs_out: np.ndarray | None = None
    time_chunks: list[np.ndarray] = []
    high_click_chunks: list[np.ndarray] = []
    low_click_chunks: list[np.ndarray] = []

    low_thr_prev: float | None = None
    high_thr_prev: float | None = None
    prev_event_rate: float | None = None
    prev_peak_to_p95_ratio: float | None = None
    prev_rms: float | None = None

    low_thr_eff: float = low_thr
    high_thr_eff: float = high_thr
    thr_per_chunk: list[tuple[float, float]] = []

    start_sample = 0
    window_idx = 0

    while True:
        sound_data = sp.read_audio_segment(file_path, start_sample, num_samples_mid)
        if sound_data is None or len(sound_data) == 0:
            break

        dHFC, SF, WPD, CD, Sxx_chunk, freqs = extract_odfs(
            sound_data,
            fs,
            short_term_duration,
            flow=flow,
            fhig=fhig,
        )
        Sxx_chunks.append(Sxx_chunk)
        if freqs_out is None:
            freqs_out = freqs

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

        num_short_frames = len(z_used)
        time_offset = start_sample / fs
        time_chunk = time_offset + (np.arange(num_short_frames) + 0.5) * short_term_duration

        if log_stats:
            logger.info(
                (
                    "file=%s window=%d t0=%.3fs "
                    "z_raw[min/mean/max/p95]=%.8f/%.8f/%.8f/%.8f "
                    "z_used[min/mean/max/p95]=%.8f/%.8f/%.8f/%.8f mode=%s "
                    "dHFC[min/mean/max/p95]=%.8f/%.8f/%.8f/%.8f "
                    "SF[min/mean/max/p95]=%.8f/%.8f/%.8f/%.8f "
                    "WPD[min/mean/max/p95]=%.8f/%.8f/%.8f/%.8f "
                    "CD[min/mean/max/p95]=%.8f/%.8f/%.8f/%.8f"
                ),
                file_path.name,
                window_idx,
                time_offset,
                float(np.min(z_raw)),
                float(np.mean(z_raw)),
                float(np.max(z_raw)),
                float(np.percentile(z_raw, 95)),
                float(np.min(z_used)),
                float(np.mean(z_used)),
                float(np.max(z_used)),
                float(np.percentile(z_used, 95)),
                z_mode,
                float(np.min(dHFC)),
                float(np.mean(dHFC)),
                float(np.max(dHFC)),
                float(np.percentile(dHFC, 95)),
                float(np.min(SF)),
                float(np.mean(SF)),
                float(np.max(SF)),
                float(np.percentile(SF, 95)),
                float(np.min(WPD)),
                float(np.mean(WPD)),
                float(np.max(WPD)),
                float(np.percentile(WPD, 95)),
                float(np.min(CD)),
                float(np.mean(CD)),
                float(np.max(CD)),
                float(np.percentile(CD, 95)),
            )

        if thr_mode == "adaptive":
            low_thr_local, high_thr_local = compute_adaptive_threshold_chunk(
                z_used,
                k_low=bg_k_low,
                k_high=bg_k_high,
                low_thr_floor=low_thr_floor,
                high_thr_floor=high_thr_floor,
            )

            low_thr_local = max(low_thr_local, low_thr_ratio * high_thr_local)

            if low_thr_prev is None:
                low_thr_chunk = low_thr_local
                high_thr_chunk = high_thr_local
            else:
                beta = float(np.clip(smooth_beta, 0.0, 1.0))
                low_thr_chunk = beta * low_thr_prev + (1.0 - beta) * low_thr_local
                high_thr_chunk = beta * high_thr_prev + (1.0 - beta) * high_thr_local

            if (
                prev_event_rate is not None
                and prev_peak_to_p95_ratio is not None
                and prev_event_rate > penalty_event_rate_limit
                and prev_peak_to_p95_ratio < penalty_ratio_limit
            ):
                high_thr_chunk *= penalty_factor_high
                low_thr_chunk *= penalty_factor_low
                if log_stats:
                    logger.info(
                        "file=%s window=%d penalty applied (event_rate=%.2f, ratio=%.4f)",
                        file_path.name,
                        window_idx,
                        prev_event_rate,
                        prev_peak_to_p95_ratio,
                    )
            elif (
                prev_rms is not None
                and prev_peak_to_p95_ratio is not None
                and prev_rms < rms_clean_limit
                and prev_peak_to_p95_ratio > ratio_good_limit
            ):
                high_thr_chunk *= reward_factor
                low_thr_chunk *= reward_factor
                if log_stats:
                    logger.info(
                        "file=%s window=%d reward applied (rms=%.6f, ratio=%.4f)",
                        file_path.name,
                        window_idx,
                        prev_rms,
                        prev_peak_to_p95_ratio,
                    )

            low_thr_eff = low_thr_chunk
            high_thr_eff = high_thr_chunk
            low_thr_prev = low_thr_chunk
            high_thr_prev = high_thr_chunk

        thr_per_chunk.append((low_thr_eff, high_thr_eff))

        if thr_mode == "adaptive":
            high_chunk = (z_used >= high_thr_eff).astype(int)
            low_chunk = (
                (z_raw >= low_thr_eff)
                & (z_raw < high_thr_eff)
                & (z_contrast >= contrast_low_thr)
            ).astype(int)
        else:
            high_chunk, low_chunk = detect_onsets(z_used, high_thr_eff, low_thr_eff)

        z_max_chunk = float(np.max(z_used)) if z_used.size else 0.0
        z_p95_chunk = float(np.percentile(z_used, 95)) if z_used.size else 0.0
        prev_peak_to_p95_ratio = z_max_chunk / (z_p95_chunk + eps)
        prev_event_rate = float((high_chunk.sum() + low_chunk.sum()) / max(mid_term_duration, eps))
        prev_rms = float(np.sqrt(np.mean(np.square(np.asarray(sound_data, dtype=float))))) if len(sound_data) > 0 else 0.0

        time_sec_list.extend(time_chunk.tolist())
        z_raw_chunks.append(z_raw)
        z_used_chunks.append(z_used)
        sound_chunks.append(np.asarray(sound_data, dtype=float))
        time_chunks.append(time_chunk)
        high_click_chunks.append(high_chunk)
        low_click_chunks.append(low_chunk)

        start_sample += num_samples_mid
        window_idx += 1

    if len(z_used_chunks) == 0:
        empty_events = pd.DataFrame(columns=["time_sec", "high_click_events", "low_click_events"])
        empty_windows = pd.DataFrame(
            columns=[
                "window_index",
                "window_start_sec",
                "window_end_sec",
                "rms",
                "peak",
                "crest_factor",
                "z_raw_mean",
                "z_raw_std",
                "z_raw_p95",
                "z_raw_max",
                "peak_to_p95_ratio",
                "z_used_mean",
                "z_used_std",
                "z_used_p95",
                "z_used_max",
                "num_high",
                "num_low",
                "event_rate",
                "low_thr_used",
                "high_thr_used",
            ]
        )
        empty_event_level = pd.DataFrame(
            columns=[
                "event_id", "event_type", "start_sec", "end_sec",
                "duration_ms", "peak_time_sec", "peak_z",
                "dominant_freq_hz", "dominant_amp", "bw_3db_hz", "bw_10db_hz",
                *[f"band_energy_{i}" for i in range(1, 11)],
                *[f"band_energy_rel_{i}" for i in range(1, 11)],
            ]
        )
        return empty_events, empty_windows, empty_event_level

    z_raw_all = np.concatenate(z_raw_chunks)
    z_used_all = np.concatenate(z_used_chunks)
    Sxx_all = np.concatenate(Sxx_chunks, axis=1)

    if thr_mode == "adaptive":
        high_click_arr = np.concatenate(high_click_chunks)
        low_click_arr = np.concatenate(low_click_chunks)
    else:
        high_click_arr, low_click_arr = detect_onsets(z_used_all, high_thr_eff, low_thr_eff)

    if disable_low:
        low_click_arr = np.zeros_like(low_click_arr)

    df_events = pd.DataFrame(
        {
            "time_sec": time_sec_list,
            "high_click_events": high_click_arr,
            "low_click_events": low_click_arr,
        }
    )

    if len(time_sec_list) > 1:
        diffs = np.diff(np.asarray(time_sec_list, dtype=float))
        diffs = diffs[diffs > 0]
        frame_dt = float(np.median(diffs)) if diffs.size else short_term_duration
    else:
        frame_dt = short_term_duration

    df_events_level = postprocess_events(
        np.asarray(time_sec_list, dtype=float),
        high_click_arr,
        low_click_arr,
        frame_dt=frame_dt,
        min_event_duration_ms=min_event_duration_ms,
        merge_gap_ms=merge_gap_ms,
        refractory_ms=refractory_ms,
        z_all=z_used_all,
    )

    if freqs_out is not None:
        df_events_level = enrich_events_with_spectral(
            df_events_level,
            Sxx_all,
            freqs_out,
            flow=flow,
            fhig=fhig,
            short_term_duration=short_term_duration,
        )

    # Monta manifest por janela
    window_rows: list[dict[str, float]] = []
    cursor = 0

    for idx, z_raw_chunk in enumerate(z_raw_chunks):
        z_used_chunk = z_used_chunks[idx]
        n = len(z_used_chunk)
        high_chunk = high_click_arr[cursor:cursor + n]
        low_chunk = low_click_arr[cursor:cursor + n]

        window_start_sec = float(time_chunks[idx][0] - 0.5 * short_term_duration)
        window_end_sec = float(time_chunks[idx][-1] + 0.5 * short_term_duration)

        metrics = compute_window_metrics(
            sound_chunks[idx],
            z_raw_chunk,
            high_chunk,
            low_chunk,
            mid_term_duration=mid_term_duration,
        )
        z_used_mean = float(np.mean(z_used_chunk)) if z_used_chunk.size else 0.0
        z_used_std = float(np.std(z_used_chunk)) if z_used_chunk.size else 0.0
        z_used_p95 = float(np.percentile(z_used_chunk, 95)) if z_used_chunk.size else 0.0
        z_used_max = float(np.max(z_used_chunk)) if z_used_chunk.size else 0.0
        low_thr_used, high_thr_used = thr_per_chunk[idx]

        row = {
            "window_index": idx,
            "window_start_sec": window_start_sec,
            "window_end_sec": window_end_sec,
            **metrics,
            "z_used_mean": z_used_mean,
            "z_used_std": z_used_std,
            "z_used_p95": z_used_p95,
            "z_used_max": z_used_max,
            "low_thr_used": low_thr_used,
            "high_thr_used": high_thr_used,
        }
        window_rows.append(row)
        cursor += n

    df_windows = pd.DataFrame(window_rows)

    if log_stats:
        logger.info(
            (
                "file=%s summary: mode=%s low_thr_eff=%.8f high_thr_eff=%.8f "
                "frames=%d high_total=%d low_total=%d z_mode=%s "
                "z_used[min/mean/max/p95]=%.8f/%.8f/%.8f/%.8f "
                "weights=(%.3f, %.3f, %.3f, %.3f) norm_p=%.1f"
            ),
            file_path.name,
            thr_mode,
            low_thr_eff,
            high_thr_eff,
            len(df_events),
            int(df_events["high_click_events"].sum()),
            int(df_events["low_click_events"].sum()),
            z_mode,
            float(np.min(z_used_all)),
            float(np.mean(z_used_all)),
            float(np.max(z_used_all)),
            float(np.percentile(z_used_all, 95)),
            w_hfc,
            w_sf,
            w_wpd,
            w_cd,
            norm_percentile,
        )
        if thr_mode == "adaptive":
            logger.info(
                (
                    "file=%s adaptive params: k_low=%.2f k_high=%.2f smooth_beta=%.2f "
                    "penalty_event_rate_limit=%.1f penalty_ratio_limit=%.4f "
                    "penalty_factor_high=%.3f penalty_factor_low=%.3f "
                    "reward_factor=%.3f rms_clean_limit=%.6f ratio_good_limit=%.2f"
                ),
                file_path.name,
                bg_k_low,
                bg_k_high,
                smooth_beta,
                penalty_event_rate_limit,
                penalty_ratio_limit,
                penalty_factor_high,
                penalty_factor_low,
                reward_factor,
                rms_clean_limit,
                ratio_good_limit,
            )

    return df_events, df_windows, df_events_level


def main(
    input_base_dir: Path,
    output_base_dir: Path,
    *,
    mid_term_duration: float = 1.0,
    short_term_duration: float = 0.001,
    high_thr: float = 0.2,
    low_thr: float = 0.05,
    flow: float = 75e3,
    fhig: float = 125e3,
    log_stats: bool = False,
    thr_mode: str = "fixed",
    low_quantile: float = 0.95,
    high_quantile: float = 0.99,
    low_thr_floor: float = 1e-5,
    high_thr_floor: float = 5e-5,
    recursive: bool = True,
    pattern: str = "*.wav",
    w_hfc: float = 0.35,
    w_sf: float = 0.35,
    w_wpd: float = 0.15,
    w_cd: float = 0.15,
    norm_percentile: float = 95.0,
    min_event_duration_ms: float = 1.0,
    merge_gap_ms: float = 1.0,
    refractory_ms: float = 2.0,
    z_mode: str = "raw",
    contrast_win_ms: float = 100.0,
    contrast_alpha: float = 0.5,
    min_clicks_in_chain: int = 5,
    chain_gap_factor: float = 2.0,
    bg_k_low: float = 3.0,
    bg_k_high: float = 6.0,
    smooth_beta: float = 0.8,
    penalty_event_rate_limit: float = 50.0,
    penalty_ratio_limit: float = 1.5,
    penalty_factor_high: float = 1.15,
    penalty_factor_low: float = 1.10,
    reward_factor: float = 0.95,
    rms_clean_limit: float = 0.01,
    ratio_good_limit: float = 3.0,
    low_thr_ratio: float = 0.75,
    contrast_low_thr: float = 2.5,
    disable_low: bool = False,
) -> None:
    setup_logging(log_stats)

    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    consolidated_chains_path = output_base_dir / "all_chains.csv"

    if recursive:
        audio_iter = input_base_dir.rglob(pattern)
    else:
        audio_iter = input_base_dir.glob(pattern)

    for audio_file in sorted(audio_iter):
        if not audio_file.is_file():
            continue

        _, _, df_event_level = process_audio_file(
            audio_file,
            mid_term_duration=mid_term_duration,
            short_term_duration=short_term_duration,
            high_thr=high_thr,
            low_thr=low_thr,
            flow=flow,
            fhig=fhig,
            log_stats=log_stats,
            thr_mode=thr_mode,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            low_thr_floor=low_thr_floor,
            high_thr_floor=high_thr_floor,
            w_hfc=w_hfc,
            w_sf=w_sf,
            w_wpd=w_wpd,
            w_cd=w_cd,
            norm_percentile=norm_percentile,
            min_event_duration_ms=min_event_duration_ms,
            merge_gap_ms=merge_gap_ms,
            refractory_ms=refractory_ms,
            z_mode=z_mode,
            contrast_win_ms=contrast_win_ms,
            contrast_alpha=contrast_alpha,
            bg_k_low=bg_k_low,
            bg_k_high=bg_k_high,
            smooth_beta=smooth_beta,
            penalty_event_rate_limit=penalty_event_rate_limit,
            penalty_ratio_limit=penalty_ratio_limit,
            penalty_factor_high=penalty_factor_high,
            penalty_factor_low=penalty_factor_low,
            reward_factor=reward_factor,
            rms_clean_limit=rms_clean_limit,
            ratio_good_limit=ratio_good_limit,
            low_thr_ratio=low_thr_ratio,
            contrast_low_thr=contrast_low_thr,
            disable_low=disable_low,
        )

        df_chains = build_click_chains(
            df_event_level,
            min_clicks_in_chain=min_clicks_in_chain,
            max_gap_factor=chain_gap_factor,
            filepath=str(audio_file),
            event_types=("high",),
            flow=flow,
            fhig=fhig,
        )

        abs_audio_path = str(audio_file.resolve())
        append_to_consolidated(df_chains, consolidated_chains_path, abs_audio_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect high- and low-intensity click events for batches of WAV files."
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        type=Path,
        default=Path("../data/wav-files"),
        help="Diretório base contendo os arquivos .wav (busca recursiva por padrão).",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("../data/detections"),
        help="Diretório onde os CSVs serão salvos, espelhando a hierarquia do input.",
    )
    parser.add_argument(
        "--mid-term-duration",
        type=float,
        default=1.0,
        help="Duração (s) das janelas de médio prazo usadas na agregação.",
    )
    parser.add_argument(
        "--short-term-duration",
        type=float,
        default=0.001,
        help="Duração (s) de cada frame curto analisado nas ODFs.",
    )
    parser.add_argument(
        "--high-thr",
        type=float,
        default=0.2,
        help="Limiar para marcar cliques de alta intensidade.",
    )
    parser.add_argument(
        "--low-thr",
        type=float,
        default=0.05,
        help="Limiar para cliques de baixa intensidade (entre baixo e alto).",
    )
    parser.add_argument(
        "--thr-mode",
        choices=["fixed", "adaptive"],
        default="fixed",
        help=(
            "Modo de limiar: fixed usa --low-thr/--high-thr; "
            "adaptive calcula por janela via mediana+MAD com suavização e penalização por regime."
        ),
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=0.95,
        help="(legado, sem efeito em thr-mode adaptive) Quantil inferior para limiar adaptativo por arquivo (0-1).",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=0.99,
        help="(legado, sem efeito em thr-mode adaptive) Quantil superior para limiar adaptativo por arquivo (0-1).",
    )
    parser.add_argument(
        "--low-thr-floor",
        type=float,
        default=1e-5,
        help="Piso mínimo do limiar baixo no modo adaptativo.",
    )
    parser.add_argument(
        "--high-thr-floor",
        type=float,
        default=5e-5,
        help="Piso mínimo do limiar alto no modo adaptativo.",
    )
    parser.add_argument(
        "--bg-k-low",
        type=float,
        default=3.0,
        help="Multiplicador do MAD para calcular o limiar low no modo adaptive (padrão: %(default)s).",
    )
    parser.add_argument(
        "--bg-k-high",
        type=float,
        default=6.0,
        help="Multiplicador do MAD para calcular o limiar high no modo adaptive (padrão: %(default)s).",
    )
    parser.add_argument(
        "--smooth-beta",
        type=float,
        default=0.8,
        help="Fator de suavização exponencial dos thresholds entre janelas no modo adaptive (padrão: %(default)s).",
    )
    parser.add_argument(
        "--penalty-event-rate-limit",
        type=float,
        default=50.0,
        help="Taxa de eventos (eventos/s) acima da qual a janela seguinte é penalizada (padrão: %(default)s).",
    )
    parser.add_argument(
        "--penalty-ratio-limit",
        type=float,
        default=1.5,
        help="peak_to_p95_ratio abaixo do qual a janela seguinte é penalizada (padrão: %(default)s).",
    )
    parser.add_argument(
        "--penalty-factor-high",
        type=float,
        default=1.15,
        help="Multiplicador aplicado ao high_thr em janelas suspeitas de ruído (padrão: %(default)s).",
    )
    parser.add_argument(
        "--penalty-factor-low",
        type=float,
        default=1.10,
        help="Multiplicador aplicado ao low_thr em janelas suspeitas de ruído (padrão: %(default)s).",
    )
    parser.add_argument(
        "--reward-factor",
        type=float,
        default=0.95,
        help="Multiplicador aplicado aos thresholds em janelas limpas (padrão: %(default)s).",
    )
    parser.add_argument(
        "--rms-clean-limit",
        type=float,
        default=0.01,
        help="RMS máximo para considerar uma janela como limpa para recompensa (padrão: %(default)s).",
    )
    parser.add_argument(
        "--ratio-good-limit",
        type=float,
        default=3.0,
        help="peak_to_p95_ratio mínimo para considerar uma janela como limpa para recompensa (padrão: %(default)s).",
    )
    parser.add_argument(
        "--low-thr-ratio",
        type=float,
        default=0.75,
        help=(
            "Fração mínima de high_thr que low_thr deve atingir no modo adaptive. "
            "Impede que o low_thr descole demais do high_thr (padrão: %(default)s)."
        ),
    )
    parser.add_argument(
        "--contrast-low-thr",
        type=float,
        default=2.5,
        help=(
            "Threshold mínimo de Z-contrast local exigido para classificar um frame como low "
            "no modo adaptive. Reduz captura de ruído difuso (padrão: %(default)s)."
        ),
    )
    parser.add_argument(
        "--disable-low",
        action="store_true",
        default=False,
        help="Desativa completamente os eventos low. Útil para analisar o high isoladamente.",
    )
    parser.add_argument(
        "--flow",
        type=float,
        default=75e3,
        help="Limite inferior da banda de frequências (Hz) usada em HFC/SF.",
    )
    parser.add_argument(
        "--fhig",
        type=float,
        default=125e3,
        help="Limite superior da banda de frequências (Hz) usada em HFC/SF.",
    )
    parser.add_argument(
        "--w-hfc",
        type=float,
        default=0.35,
        help="Peso do dHFC na fusão aditiva.",
    )
    parser.add_argument(
        "--w-sf",
        type=float,
        default=0.35,
        help="Peso do Spectral Flux na fusão aditiva.",
    )
    parser.add_argument(
        "--w-wpd",
        type=float,
        default=0.15,
        help="Peso do WPD na fusão aditiva.",
    )
    parser.add_argument(
        "--w-cd",
        type=float,
        default=0.15,
        help="Peso do Complex Domain na fusão aditiva.",
    )
    parser.add_argument(
        "--norm-percentile",
        type=float,
        default=95.0,
        help="Percentil usado na normalização robusta das ODFs.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Ativa logs por janela com estatísticas de z e contagem de eventos.",
    )
    parser.add_argument(
        "--min-event-duration-ms",
        type=float,
        default=1.0,
        help="Duração mínima (ms) para manter um evento após agrupamento temporal.",
    )
    parser.add_argument(
        "--merge-gap-ms",
        type=float,
        default=1.0,
        help="Lacuna máxima (ms) para mesclar eventos consecutivos do mesmo tipo.",
    )
    parser.add_argument(
        "--refractory-ms",
        type=float,
        default=2.0,
        help="Período refratário (ms): eventos muito próximos mantêm apenas o mais forte.",
    )
    parser.add_argument(
        "--z-mode",
        choices=["raw", "contrast", "hybrid"],
        default="raw",
        help="Score usado na detecção: raw, contrast ou hybrid (mistura).",
    )
    parser.add_argument(
        "--contrast-win-ms",
        type=float,
        default=100.0,
        help="Janela local (ms) para calcular z_contrast.",
    )
    parser.add_argument(
        "--contrast-alpha",
        type=float,
        default=0.5,
        help="Peso do z_raw no modo hybrid: z_used = alpha*z_raw + (1-alpha)*z_contrast.",
    )
    parser.add_argument(
        "--min-clicks-in-chain",
        type=int,
        default=5,
        help="Número mínimo de cliques para considerar uma cadeia.",
    )
    parser.add_argument(
        "--chain-gap-factor",
        type=float,
        default=2.0,
        help="Fator multiplicativo do ICI médio para encerrar uma cadeia.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.wav",
        help="Padrão glob para selecionar arquivos de áudio.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Desativa a busca recursiva por arquivos dentro do diretório de entrada.",
    )
    parser.set_defaults(recursive=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.input_dir,
        args.output_dir,
        mid_term_duration=args.mid_term_duration,
        short_term_duration=args.short_term_duration,
        high_thr=args.high_thr,
        low_thr=args.low_thr,
        flow=args.flow,
        fhig=args.fhig,
        log_stats=args.log_stats,
        thr_mode=args.thr_mode,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        low_thr_floor=args.low_thr_floor,
        high_thr_floor=args.high_thr_floor,
        recursive=args.recursive,
        pattern=args.pattern,
        w_hfc=args.w_hfc,
        w_sf=args.w_sf,
        w_wpd=args.w_wpd,
        w_cd=args.w_cd,
        norm_percentile=args.norm_percentile,
        min_event_duration_ms=args.min_event_duration_ms,
        merge_gap_ms=args.merge_gap_ms,
        refractory_ms=args.refractory_ms,
        z_mode=args.z_mode,
        contrast_win_ms=args.contrast_win_ms,
        contrast_alpha=args.contrast_alpha,
        min_clicks_in_chain=args.min_clicks_in_chain,
        chain_gap_factor=args.chain_gap_factor,
        bg_k_low=args.bg_k_low,
        bg_k_high=args.bg_k_high,
        smooth_beta=args.smooth_beta,
        penalty_event_rate_limit=args.penalty_event_rate_limit,
        penalty_ratio_limit=args.penalty_ratio_limit,
        penalty_factor_high=args.penalty_factor_high,
        penalty_factor_low=args.penalty_factor_low,
        reward_factor=args.reward_factor,
        rms_clean_limit=args.rms_clean_limit,
        ratio_good_limit=args.ratio_good_limit,
        low_thr_ratio=args.low_thr_ratio,
        contrast_low_thr=args.contrast_low_thr,
        disable_low=args.disable_low,
    )