# utils/file_utils.py
# ----------------------------------------------------------------------
# File and time parsing utilities used across the PyQt GUI.
#
# Includes:
#   • Excel loader that reads *cell display text* reliably (as strings)
#   • Robust parsing of Excel time display strings into seconds
#   • Formatting helpers for MM:SS(.d...) output
#   • HDF5 (.h5) loader for signals/time/channel names
#   • Legacy/general time parser (parse_tod) for mixed formats
# ----------------------------------------------------------------------

import pandas as pd
import h5py
from PyQt5.QtWidgets import QMessageBox

import re
from decimal import Decimal, ROUND_HALF_UP


# ==============================================================
# Excel loader
# ==============================================================
def load_excel_file(filepath):
    """
    Load an Excel file into a pandas DataFrame while preserving what Excel *displays*.

    Why dtype=str?
      Excel cells can contain time/datetime/timedelta values whose underlying numeric
      representation differs from what the UI shows. Reading as strings keeps the
      human-visible representation so downstream parsing (e.g., mm:ss.s) is consistent.

    Notes:
      • header=1 means row 2 in Excel is treated as the header row.
      • keep_default_na=False prevents empty strings from becoming NaN automatically,
        which makes downstream "NA" handling explicit and consistent.
      • strips whitespace in object columns for cleaner UI display / parsing.
    """
    try:
        df = pd.read_excel(
            filepath,
            header=1,
            dtype=str,          # read what the cell DISPLAYS (as text)
            engine="openpyxl",
            keep_default_na=False
        )

        # Strip whitespace in object columns (safe across pandas versions)
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)

        return df

    except Exception as e:
        QMessageBox.critical(None, "Error", f"Error loading Excel file:\n{e}")
        return None


# ==============================================================
# Excel "display string" time parsing helpers
# ==============================================================
def _to_float(v: str) -> float:
    """
    Convert a numeric string to float using Decimal for precision.
    This helps avoid float artifacts when parsing fractional seconds.
    """
    return float(Decimal(v))


def _decimals_from_display(token: str, default: int = 1) -> int:
    """
    Determine how many decimal places appear in the seconds field of a
    displayed time token.

    Example:
      "12:34.5" -> 1
      "12:34.50" -> 2
      "12:34" -> 0
    """
    if not isinstance(token, str):
        return default
    token = token.strip()
    if not token:
        return default

    last = token.split(":")[-1]  # seconds component (possibly with decimals)
    if "." in last:
        return max(0, len(last.split(".")[1]))
    return 0


def _q(secs: float, dp: int) -> float:
    """
    Quantize seconds to a fixed number of decimals using Excel-like rounding
    (ROUND_HALF_UP), which matches typical spreadsheet display behavior.
    """
    q = Decimal(10) ** (-dp)
    return float(Decimal(str(secs)).quantize(q, rounding=ROUND_HALF_UP))


def parse_excel_display_to_seconds(raw, *, default_dp: int = 1, within_hour: bool = True):
    """
    Parse the *cell display text* (string read from Excel) into (seconds, dp).

    Supported display styles:
      • AM/PM time-of-day:
          "2:13 PM", "2:13:15 PM", "2:13:15.5 PM"
      • 24h/HH:MM:SS(.fff):
          "01:02:03", "01:02:03.25"
      • MM:SS(.fff):
          "12:34", "12:34.5"
      • SS(.fff):
          "9", "9.5"

    Parameters
    ----------
    raw : any
        Value read from pandas (expected to be string-like because load_excel_file uses dtype=str)
    default_dp : int
        If the display token has no decimal places, assume this many decimals (often 1).
    within_hour : bool
        If True, collapse time-of-day into "seconds within the hour" (mod 3600).
        This matches Excel's common "mm:ss" display logic where hours are ignored.

    Returns
    -------
    (secs_snapped, dp) or (None, None) if blank/unparsable.
    """
    if raw is None:
        return None, None

    s = str(raw).strip()
    if s == "" or s.upper() in {"NA", "N/A", "-", "--"}:
        return None, None

    # Use full string for AM/PM detection; use token for regex patterns.
    token = s.split()[0]
    if token == "":
        return None, None

    # Determine decimal precision based on what Excel shows.
    dp_seen = _decimals_from_display(token, default=default_dp)
    dp = dp_seen if dp_seen > 0 else default_dp

    # --- AM/PM time-of-day: "H:MM(:SS(.fff)) AM/PM"
    m_ampm = re.match(
        r"^\s*(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?\s*([AP]M)\s*$",
        s, re.IGNORECASE
    )
    if m_ampm:
        h = int(m_ampm.group(1))   # kept for completeness if within_hour=False
        m = int(m_ampm.group(2))
        sec_s = m_ampm.group(3)
        frac = m_ampm.group(4) or "0"

        # Build seconds with fractional component if present
        ss = _to_float(f"{sec_s or '0'}.{frac}") if (sec_s or frac != "0") else float(sec_s or 0)

        # Excel "mm:ss" behavior typically ignores hours
        secs = (m * 60 + ss) if within_hour else (h * 3600 + m * 60 + ss)
        return _q(secs, dp), dp

    # --- HH:MM:SS(.fff)
    m_hms = re.match(r"^\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*$", token)
    if m_hms:
        h = int(m_hms.group(1))
        m = int(m_hms.group(2))
        sec_s = m_hms.group(3)
        frac = m_hms.group(4) or "0"
        ss = _to_float(f"{sec_s}.{frac}") if frac != "0" else float(sec_s)

        secs_total = h * 3600 + m * 60 + ss
        secs = (secs_total % 3600) if within_hour else secs_total
        return _q(secs, dp), dp

    # --- MM:SS(.fff)
    m_ms = re.match(r"^\s*(\d{1,4}):(\d{2})(?:\.(\d+))?\s*$", token)
    if m_ms:
        m = int(m_ms.group(1))
        sec_s = m_ms.group(2)
        frac = m_ms.group(3) or "0"
        ss = _to_float(f"{sec_s}.{frac}") if frac != "0" else float(sec_s)
        secs = m * 60 + ss
        return _q(secs, dp), dp

    # --- SS(.fff)
    m_s = re.match(r"^\s*(\d+)(?:\.(\d+))?\s*$", token)
    if m_s:
        sec_s = m_s.group(1)
        frac = m_s.group(2) or "0"
        ss = _to_float(f"{sec_s}.{frac}") if frac != "0" else float(sec_s)
        return _q(ss, dp), dp

    return None, None


def fmt_mmss_exact(secs: float, dp: int) -> str:
    """
    Format seconds as 'MM:SS(.d...)' using exactly dp decimals.

    Examples:
      secs=75.0, dp=1 -> "01:15.0"
      secs=5.25, dp=2 -> "00:05.25"
    """
    mm = int(secs // 60)
    ss = secs - mm * 60

    if dp > 0:
        s = f"{ss:.{dp}f}"
        # Ensure seconds field is at least 2 digits before decimal
        if ss < 10:
            s = "0" + s
    else:
        s = f"{int(ss):02d}"

    return f"{mm:02d}:{s}"


# ==============================================================
# HDF5 (.h5) loader
# ==============================================================
def load_h5_file(filepath):
    """
    Load iEEG data stored in an HDF5 file.

    Expected layout:
      • data/Signals: (num_channels, num_samples)
      • data/Time:    (num_samples,)
      • metadata/channel_names (optional): list/array of bytestrings

    Returns
    -------
    signals : np.ndarray | None
    time_array : np.ndarray | None
    channel_names : list[str] | None
    """
    try:
        with h5py.File(filepath, "r") as f:
            signals = f["data/Signals"][:]  # (channels, samples)
            time_array = f["data/Time"][:]  # (samples,)

            if "metadata/channel_names" in f:
                ch_names = f["metadata/channel_names"][:]
                # HDF5 often stores strings as bytes; decode to python str
                channel_names = [name.decode("utf-8") for name in ch_names]
            else:
                channel_names = None

        return signals, time_array, channel_names

    except Exception as e:
        QMessageBox.critical(None, "Error", f"Error loading H5 file:\n{e}")
        return None, None, None


# ==============================================================
# Legacy / general-purpose time parsing (time_utils style)
# ==============================================================
def parse_tod(value, *, within_hour=True):
    """
    Parse a time-of-day/duration-like value to seconds (float).

    This is a more general parser than parse_excel_display_to_seconds and is
    useful when inputs may be:
      • pandas Timedelta formats ("0 days 00:41:37", "0:34:10.600000", etc.)
      • "HH:MM:SS(.fff)", "MM:SS(.fff)", "SS(.fff)"
      • strings with AM/PM suffixes (ignored)

    Parameters
    ----------
    value : any
        Raw value from Excel/pandas/UI
    within_hour : bool
        If True and the parsed duration looks like a time-of-day (< 24h),
        fold into seconds within the hour (mod 3600). This matches prior UI
        semantics where hours are ignored for "mm:ss" display.

    Returns
    -------
    float seconds or None if missing/unparsable.
    """
    import pandas as pd
    import math

    # Missing check
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    s = str(value).strip()

    # 1) Let pandas try first (covers lots of corner cases)
    try:
        td = pd.to_timedelta(s)
        secs = float(td.total_seconds())
    except Exception:
        # 2) Manual parsing fallbacks
        try:
            clean = (
                s.replace("AM", "").replace("PM", "")
                 .replace("am", "").replace("pm", "")
                 .strip()
            )
            parts = clean.split(":")
            if len(parts) == 3:  # HH:MM:SS(.fff)
                h = float(parts[0]); m = float(parts[1]); ss = float(parts[2])
                secs = h * 3600 + m * 60 + ss
            elif len(parts) == 2:  # MM:SS(.fff)
                m = float(parts[0]); ss = float(parts[1])
                secs = m * 60 + ss
            else:  # SS(.fff)
                secs = float(parts[0])
        except Exception:
            return None

    # If it looks like a time-of-day, fold into within-hour seconds
    if within_hour and 0 <= secs < 24 * 3600:
        secs = secs % 3600.0

    return float(secs)
