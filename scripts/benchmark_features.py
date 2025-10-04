#!/usr/bin/env python3
"""
Benchmark: Laufzeit-Messung aller Features in src/features

• Sucht dynamisch alle Module in src/features/
• Unterstützt zwei Patterns:
    - Modul mit Funktion:   compute(signal, sr) -> dict
    - Modul mit Klasse:     DNSMOS(...).compute(signal, sr) -> dict   (Adapter integriert)
• Lädt Audiofiles (wav/mp3/flac), misst Zeit, sammelt Keys/Fehler
• Schreibt details.csv (pro Datei) & summary.csv (pro Feature)

Beispiel:
python scripts/benchmark_features.py \
  --audio_dir audio_files/de/combined/clips \
  --sr 16000 --max_files 30 \
  --details_csv results/benchmark/details.csv \
  --summary_csv results/benchmark/summary.csv \
  --exclude dnsmos  # (optional)
"""

from __future__ import annotations
import argparse
import fnmatch
import importlib
import os
import pkgutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa


# -------- utils: import src/features safely --------
def ensure_project_root_on_path() -> None:
    """Fügt das Repo-Root (Ordner mit 'src') zu sys.path hinzu, falls nötig."""
    cwd = Path.cwd()
    if (cwd / "src").is_dir():
        if str(cwd) not in sys.path:
            sys.path.insert(0, str(cwd))
        return
    # Fallback: bis 3 Ebenen nach oben schauen
    p = cwd
    for _ in range(3):
        p = p.parent
        if (p / "src").is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return


@dataclass
class FeatureRunner:
    name: str
    compute_fn: Callable[[np.ndarray, int], Dict[str, float]]

    def run(self, signal: np.ndarray, sr: int) -> Dict[str, float]:
        return self.compute_fn(signal, sr)


def discover_feature_modules(include: List[str], exclude: List[str]) -> List[FeatureRunner]:
    """
    Findet Module in src.features und baut FeatureRunner.
    Regeln:
      - Modul muss compute(...) enthalten ODER eine Klasse DNSMOS mit .compute(...)
      - include/exclude sind fnmatch-Pattern auf Modulnamen (ohne .py)
    """
    runners: List[FeatureRunner] = []

    try:
        pkg = importlib.import_module("src.features")
    except ModuleNotFoundError:
        ensure_project_root_on_path()
        pkg = importlib.import_module("src.features")

    for modinfo in pkgutil.iter_modules(pkg.__path__):
        mod_name = modinfo.name  # ohne "src.features."
        if mod_name.startswith("_"):
            continue

        # include/exclude filtern
        if include and not any(fnmatch.fnmatch(mod_name, pat) for pat in include):
            continue
        if exclude and any(fnmatch.fnmatch(mod_name, pat) for pat in exclude):
            continue

        full_name = f"src.features.{mod_name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception as e:
            print(f"[WARN] Konnte Modul {full_name} nicht importieren: {e}")
            continue

        # Fall A: compute(signal, sr) als freie Funktion
        if hasattr(mod, "compute") and callable(getattr(mod, "compute")):
            runners.append(FeatureRunner(name=mod_name, compute_fn=getattr(mod, "compute")))
            continue

        # Fall B: DNSMOS-Klasse mit .compute(self, signal, sr)
        if hasattr(mod, "DNSMOS"):
            try:
                inst = getattr(mod, "DNSMOS")()  # default: 16kHz, cpu – deine Klasse hat defaults
                def _adapter(sig, sr, _inst=inst):
                    return _inst.compute(sig, sr)
                runners.append(FeatureRunner(name=mod_name, compute_fn=_adapter))
                continue
            except Exception as e:
                print(f"[WARN] DNSMOS in {mod_name} konnte nicht initialisiert werden: {e}")
                continue

        print(f"[SKIP] {mod_name}: kein compute() gefunden.")
    return runners


def iter_audio_files(audio_dir: Path, max_files: int, patterns: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(audio_dir.rglob(pat)))
    # Deduplizieren, beschränken
    uniq = []
    seen = set()
    for f in files:
        if f.suffix.lower() not in [".wav", ".mp3", ".flac"]:
            continue
        p = f.resolve()
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:max_files]


def time_call(fn: Callable[[], Dict[str, float]], repeat: int) -> Tuple[float, Dict[str, float], Optional[str]]:
    """Misst die Zeit (Sekunden) für fn() (Median über repeat). Liefert (sek, output, error_message)."""
    times = []
    out: Dict[str, float] = {}
    error: Optional[str] = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        try:
            out = fn()
            error = None
        except Exception as e:
            error = str(e)
            out = {}
        t1 = time.perf_counter()
        times.append(t1 - t0)
    # Median stabiler als Mean, wenn Outlier auftreten
    return (float(np.median(times)), out, error)


def main():
    ap = argparse.ArgumentParser(description="Benchmark Feature-Laufzeiten")
    ap.add_argument("--audio_dir", type=str, required=True, help="Ordner mit Audiodateien")
    ap.add_argument("--sr", type=int, default=16000, help="Resample-Samplingrate")
    ap.add_argument("--max_files", type=int, default=20, help="Maximale Zahl der Audiofiles")
    ap.add_argument("--repeat", type=int, default=1, help="Wiederholungen je Messung (Median)")
    ap.add_argument("--details_csv", type=str, default="results/benchmark/details.csv")
    ap.add_argument("--summary_csv", type=str, default="results/benchmark/summary.csv")
    ap.add_argument("--include", nargs="*", default=["*"], help="Feature-Pattern (fnmatch), z.B. mfcc* spectral_*")
    ap.add_argument("--exclude", nargs="*", default=[], help="Feature-Pattern zum Ausschließen, z.B. dnsmos")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    Path(args.details_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    # Feature-Runner finden
    runners = discover_feature_modules(include=args.include, exclude=args.exclude)
    if not runners:
        print("Keine Features gefunden. Prüfe include/exclude oder src/features.")
        sys.exit(1)
    print(f"Gefundene Features ({len(runners)}): " + ", ".join(r.name for r in runners))

    # Audiofiles sammeln
    files = iter_audio_files(audio_dir, max_files=args.max_files, patterns=("*.wav", "*.mp3", "*.flac"))
    if not files:
        print(f"Keine Audiofiles in {audio_dir} gefunden.")
        sys.exit(1)
    print(f"Benchmark mit {len(files)} Dateien aus {audio_dir}")

    details_rows = []

    # Vorab: alle Audios einmal laden (resample auf args.sr)
    cache: Dict[Path, Tuple[np.ndarray, int]] = {}
    for f in files:
        try:
            sig, _sr = librosa.load(str(f), sr=args.sr, mono=True)
            cache[f] = (sig.astype(np.float32, copy=False), args.sr)
        except Exception as e:
            print(f"[WARN] Laden fehlgeschlagen {f.name}: {e}")

    # Benchmark: pro Feature x Datei
    for fr in runners:
        print(f"\n==> Feature: {fr.name}")
        for f in files:
            if f not in cache:
                continue
            signal, sr = cache[f]
            sec, out, err = time_call(lambda: fr.run(signal, sr), repeat=args.repeat)
            details_rows.append({
                "feature": fr.name,
                "file": f.name,
                "secs": sec,
                "n_keys": len(out),
                "ok": err is None,
                "error": "" if err is None else err[:200],  # truncate
                "keys_preview": ",".join(list(out.keys())[:6]) if out else ""
            })
            print(f"  {f.name:40s}  {sec:7.3f}s  "
                  f"{'OK' if err is None else 'ERR'}  "
                  f"keys={len(out)}")

    details_df = pd.DataFrame(details_rows)
    details_df.to_csv(args.details_csv, index=False)

    # Summary je Feature
    def safe_mean(x): return float(np.mean(x)) if len(x) else np.nan
    def safe_std(x):  return float(np.std(x))  if len(x) else np.nan
    def safe_p95(x):  return float(np.percentile(x, 95)) if len(x) else np.nan
    def success_rate(ok_list): 
        return float(np.mean([1 if ok else 0 for ok in ok_list])) if len(ok_list) else np.nan

    summary = (details_df
               .groupby("feature")
               .agg(time_mean_s=("secs", safe_mean),
                    time_std_s=("secs", safe_std),
                    time_p95_s=("secs", safe_p95),
                    avg_keys=("n_keys", safe_mean),
                    success_rate=("ok", success_rate))
               .reset_index()
               .sort_values("time_mean_s", ascending=True))
    summary.to_csv(args.summary_csv, index=False)

    print(f"\nDetails: {args.details_csv}")
    print(f"Summary: {args.summary_csv}")
    print("\nTipp: Features mit hoher time_p95_s zuerst prüfen/optional machen (z. B. dnsmos, formants, mfcc_deltas).")


if __name__ == "__main__":
    main()