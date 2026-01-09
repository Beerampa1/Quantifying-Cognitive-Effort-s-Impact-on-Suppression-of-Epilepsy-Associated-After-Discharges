import pandas as pd
import h5py
from PyQt5.QtWidgets import QMessageBox

# utils/file_utils.py
import re
from decimal import Decimal, ROUND_HALF_UP

# ---------- Excel loader (see also main_gui fix) ----------
def load_excel_file(filepath):
    import pandas as pd
    from PyQt5.QtWidgets import QMessageBox
    try:
        df = pd.read_excel(
            filepath,
            header=1,
            dtype=str,          # read what the cell DISPLAYS
            engine="openpyxl",
            keep_default_na=False
        )
        # strip only object columns (pandas-safe across versions)
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
        return df
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Error loading Excel file:\n{e}")
        return None

# utils/file_utils.py
import re
from decimal import Decimal, ROUND_HALF_UP

def _to_float(v: str) -> float:
    return float(Decimal(v))

def _decimals_from_display(token: str, default: int = 1) -> int:
    """How many decimals are shown on the seconds component in the *cell text*."""
    if not isinstance(token, str):
        return default
    token = token.strip()
    if not token:
        return default
    last = token.split(":")[-1]
    if "." in last:
        return max(0, len(last.split(".")[1]))
    return 0

def _q(secs: float, dp: int) -> float:
    q = Decimal(10) ** (-dp)
    return float(Decimal(str(secs)).quantize(q, rounding=ROUND_HALF_UP))

def parse_excel_display_to_seconds(raw, *, default_dp: int = 1, within_hour: bool = True):
    """
    Parse the *cell display text* (what pandas read from Excel) to (seconds, dp).
    - If string looks like time-of-day (AM/PM or HH:MM:SS), compute seconds *within the hour*
      when within_hour=True (matches Excel 'mm:ss' display).
    Returns (secs_snapped, dp) or (None, None) for blanks/unparsable.
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    if s == "" or s.upper() in {"NA", "N/A", "-", "--"}:
        return None, None

    # Keep the *full* string for AM/PM detection, but make a token for pattern checks.
    token = s.split()[0]  # first chunk only
    if token == "":
        return None, None

    # If cell does not show decimals, fall back to default_dp (usually 1).
    dp_seen = _decimals_from_display(token, default=default_dp)
    dp = dp_seen if dp_seen > 0 else default_dp

    # --- AM/PM time-of-day: e.g. "2:13:15 PM", "2:13 PM"
    m_ampm = re.match(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?\s*([AP]M)\s*$", s, re.IGNORECASE)
    if m_ampm:
        h = int(m_ampm.group(1)); m = int(m_ampm.group(2))
        sec_s = m_ampm.group(3); frac = m_ampm.group(4) or "0"
        ss = _to_float(f"{sec_s or '0'}.{frac}") if (sec_s or frac != "0") else float(sec_s or 0)
        # Excel 'mm:ss' ignores hours → seconds within the hour:
        secs = m*60 + ss if within_hour else h*3600 + m*60 + ss
        return _q(secs, dp), dp

    # --- HH:MM:SS(.fff)  (24h or duration with hours)
    m_hms = re.match(r"^\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*$", token)
    if m_hms:
        h = int(m_hms.group(1)); m = int(m_hms.group(2))
        sec_s = m_hms.group(3); frac = m_hms.group(4) or "0"
        ss = _to_float(f"{sec_s}.{frac}") if frac != "0" else float(sec_s)
        secs_total = h*3600 + m*60 + ss
        secs = secs_total % 3600 if within_hour else secs_total
        return _q(secs, dp), dp

    # --- MM:SS(.fff)  (duration style)
    m_ms = re.match(r"^\s*(\d{1,4}):(\d{2})(?:\.(\d+))?\s*$", token)
    if m_ms:
        m = int(m_ms.group(1)); sec_s = m_ms.group(2); frac = m_ms.group(3) or "0"
        ss = _to_float(f"{sec_s}.{frac}") if frac != "0" else float(sec_s)
        secs = m*60 + ss
        return _q(secs, dp), dp

    # --- SS(.fff)
    m_s = re.match(r"^\s*(\d+)(?:\.(\d+))?\s*$", token)
    if m_s:
        sec_s = m_s.group(1); frac = m_s.group(2) or "0"
        ss = _to_float(f"{sec_s}.{frac}") if frac != "0" else float(sec_s)
        return _q(ss, dp), dp

    return None, None

def fmt_mmss_exact(secs: float, dp: int) -> str:
    """Format seconds to 'MM:SS(.d...)' with dp decimals."""
    mm = int(secs // 60)
    ss = secs - mm*60
    if dp > 0:
        s = f"{ss:.{dp}f}"
        if ss < 10:
            s = "0" + s
    else:
        s = f"{int(ss):02d}"
    return f"{mm:02d}:{s}"


def load_h5_file(filepath):
    """
    Load an H5 file.
    Assumes signals are stored at 'data/Signals' (shape: (num_channels, num_samples))
    and time at 'data/Time'.
    If available, channel names are at 'metadata/channel_names'.
    Returns signals, time_array, channel_names (or None).
    """
    try:
        with h5py.File(filepath, 'r') as f:
            signals = f['data/Signals'][:]  # (num_channels, num_samples)
            time_array = f['data/Time'][:]    # (num_samples,)
            if 'metadata/channel_names' in f:
                ch_names = f['metadata/channel_names'][:]
                channel_names = [name.decode('utf-8') for name in ch_names]
            else:
                channel_names = None
        return signals, time_array, channel_names
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Error loading H5 file:\n{e}")
        return None, None, None
    
# time_utils.py
def parse_tod(value, *, within_hour=True):
    """
    Parse a time-of-day/duration string to total seconds (float).
    Returns None for missing/unparsable input.

    Accepts: "HH:MM:SS(.fff)", "MM:SS(.fff)", "SS(.fff)", pandas Timedelta formats,
             and ignores AM/PM suffixes.
    """
    import pandas as pd
    import math

    # missing
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    s = str(value).strip()

    # 1) Try pandas to_timedelta first
    try:
        td = pd.to_timedelta(s)
        secs = float(td.total_seconds())
    except Exception:
        # 2) Manual fallbacks
        try:
            clean = (
                s.replace("AM", "").replace("PM", "")
                 .replace("am", "").replace("pm", "")
                 .strip()
            )
            parts = clean.split(":")
            if len(parts) == 3:                     # HH:MM:SS(.fff)
                h = float(parts[0]); m = float(parts[1]); ss = float(parts[2])
                secs = h*3600 + m*60 + ss
            elif len(parts) == 2:                   # MM:SS(.fff)
                m = float(parts[0]); ss = float(parts[1])
                secs = m*60 + ss
            else:                                   # SS(.fff)
                secs = float(parts[0])
        except Exception:
            return None

    # Keep within-hour if it looks like time-of-day; matches your previous behavior
    if within_hour and 0 <= secs < 24*3600:
        secs = secs % 3600.0

    return float(secs)




# def fmt_mmss(secs, *, seconds_decimals=1):
#     """
#     Format seconds (float) as 'MM:SS.s' (default 1 decimal) using ROUND_HALF_UP.
#     Handles rollover (e.g., 59.95 -> 'MM+1:00.0' when seconds_decimals=1).

#     Accepts numeric or a numeric string.
#     """
#     from decimal import Decimal, ROUND_HALF_UP

#     # Coerce to float if someone passes a numeric string
#     if isinstance(secs, str):
#         secs = float(secs)

#     if secs is None:
#         return "NA"

#     # Split minutes / seconds
#     mm = int(secs // 60)
#     ss = secs - mm * 60

#     # Excel-style rounding on the seconds field
#     q = Decimal(10) ** (-seconds_decimals)
#     ss_dec = Decimal(ss).quantize(q, rounding=ROUND_HALF_UP)

#     # Handle 60.0 after rounding
#     if ss_dec >= Decimal(60).quantize(q):
#         mm += 1
#         ss_dec = Decimal(0).quantize(q)

#     # Build "SS" with fixed decimals and zero-padding
#     if seconds_decimals > 0:
#         ss_str = f"{ss_dec:.{seconds_decimals}f}"
#         # Ensure at least 2 digits before decimal (e.g., '04.4')
#         if ss_dec < 10:
#             ss_str = "0" + ss_str
#     else:
#         ss_str = f"{int(ss_dec):02d}"

#     return f"{mm:02d}:{ss_str}"


# def parse_tod(value, *, within_hour=True, decimals=2, return_str=False):
#     """
#     Parse a time-of-day / duration string to seconds (float), or a formatted string.

#     Examples accepted: "20:24.38", "20:24.4", "0:34:10.600000", "34:10.60", "10.60", etc.
#     - within_hour=True  -> collapse HH:MM:SS to MM:SS within the hour (mod 3600), like your original.
#     - decimals=2        -> round seconds to fixed decimal places to avoid 20:24.38 becoming 20:24.379999...
#     - return_str=False  -> return seconds as float; if True, return "MM:SS.xx" string with exactly `decimals` digits.
#     """
#     import pandas as pd
#     import math

#     if value is None or (isinstance(value, float) and math.isnan(value)):
#         return None

#     s = str(value).strip()

#     # Try pandas’ timedelta first
#     try:
#         td = pd.to_timedelta(s)
#         secs = float(td.total_seconds())
#     except Exception:
#         # Manual fallbacks
#         try:
#             clean = (
#                 s.replace("AM", "").replace("PM", "")
#                  .replace("am", "").replace("pm", "")
#                  .strip()
#             )
#             parts = clean.split(":")
#             if len(parts) == 3:  # HH:MM:SS(.xx)
#                 h = float(parts[0]); m = float(parts[1]); ss = float(parts[2])
#                 secs = h*3600 + m*60 + ss
#             elif len(parts) == 2:  # MM:SS(.xx)
#                 m = float(parts[0]); ss = float(parts[1])
#                 secs = m*60 + ss
#             else:  # SS(.xx)
#                 secs = float(parts[0])
#         except Exception:
#             return None

#     # Keep it within the hour if desired (matches your original semantics)
#     if within_hour and 0 <= secs < 24*3600:
#         secs = secs % 3600.0  # e.g., 13:34:32 -> 34:32

#     # 🔧 Critical: round to fixed decimals to stabilize display (e.g., 20:24.38 vs 20:24.4)
#     secs = round(secs, decimals)

#     if not return_str:
#         return secs

#     # Return a stable string "MM:SS.xx" with exactly `decimals` digits
#     total = secs
#     mm = int(total // 60)
#     ss = total - mm*60
#     fmt = f"{{mm:02d}}:{{ss:0{2 + (1 if decimals>0 else 0) + decimals}.{decimals}f}}"
#     return fmt.format(mm=mm, ss=ss)

# def parse_tod(value):
#     import pandas as pd
#     import math

#     if value is None or (isinstance(value, float) and math.isnan(value)):
#         return None

#     s = str(value).strip()

#     # Try pandas’ timedelta first (handles "0:34:10.600000", "34:10.60", "10.60", etc.)
#     try:
#         td = pd.to_timedelta(s)
#         secs = float(td.total_seconds())
#     except Exception:
#         # Manual fallbacks
#         try:
#             parts = s.replace("AM", "").replace("PM", "").replace("am", "").replace("pm", "").strip().split(":")
#             if len(parts) == 3:  # HH:MM:SS(.xx)
#                 h = float(parts[0]); m = float(parts[1]); ss = float(parts[2])
#                 secs = h*3600 + m*60 + ss
#             elif len(parts) == 2:  # MM:SS(.xx)
#                 m = float(parts[0]); ss = float(parts[1])
#                 secs = m*60 + ss
#             else:  # SS(.xx)
#                 secs = float(parts[0])
#         except Exception:
#             return None

#     # 🔑 NEW: if it looks like a time-of-day (within a day), collapse to MM:SS within the hour
#     if 0 <= secs < 24*3600:
#         secs = secs % 3600.0  # e.g., 13:34:32 -> 34:32

#     return secs

# def parse_tod(value):
#     """
#     A more robust function to parse a time-of-day or duration from Excel cells
#     into total seconds.

#     1. If it's NaN or None, return None.
#     2. Try using pd.to_timedelta (handles '0 days 00:41:37', '0:41:37', etc.).
#     3. If it's a Timestamp (datetime), extract hour/minute/second.
#     4. If it's numeric (float or int), assume fraction of a day (Excel style).
#     5. As a final fallback, try manual splitting "H:MM:SS" or "H:MM:SS.xx".
#     """

#     if pd.isna(value):
#         return None

#     try:
#         td = pd.to_timedelta(str(value))
#         if not pd.isna(td):
#             return td.total_seconds()
#     except Exception:
#         pass  

#     if hasattr(value, 'hour') and hasattr(value, 'minute') and hasattr(value, 'second'):
#         return value.hour * 3600 + value.minute * 60 + value.second + (value.microsecond / 1e6)

#     if isinstance(value, (float, int)):
#         return float(value) * 86400

#     try:
#         parts = str(value).split(":")
#         if len(parts) != 3:
#             return None
#         hours = float(parts[0])
#         minutes = float(parts[1])
#         seconds = float(parts[2])
#         return hours * 3600 + minutes * 60 + seconds
#     except Exception:
#         return None
