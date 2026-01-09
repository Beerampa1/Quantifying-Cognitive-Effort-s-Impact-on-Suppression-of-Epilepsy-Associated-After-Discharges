import pandas as pd
from pathlib import Path
from typing import Optional

class ExcelMathScoreLabeler:
    """
    Loads an Excel sheet with columns:
      - Patient_Session_Trial
      - Math_Score

    Maps trial IDs to 1 (M1) or 0 (M0), skipping any other codes.
    """
    def __init__(
        self,
        excel_path: Path,
        id_col: str = "Patient_Session_Trial",
        score_col: str = "Math_Score",
        success_code: str = "M1",
        failure_code: str = "M0"
    ):
        self.mapping = {}
        df = pd.read_excel(excel_path, header=1)
        df = df[[id_col, score_col]].dropna(subset=[id_col])

        for _, row in df.iterrows():
            tid = str(row[id_col]).strip()
            score = str(row[score_col]).strip().upper()
            if score == success_code:
                self.mapping[tid] = 1
            elif score == failure_code:
                self.mapping[tid] = 0
            # else: skip (MC, NaN, etc.)

    def get(self, trial_id: str) -> Optional[int]:
        """ Return 1, 0, or None if unlabeled. """
        return self.mapping.get(trial_id)
