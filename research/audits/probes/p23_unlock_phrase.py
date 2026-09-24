"""Probe 23 (round 2, minor 9): can agent code compose the unlock phrase and unlock a holdout? (temp ledger dir)"""
import tempfile
from pathlib import Path
from quantlab import ledger
d = Path(tempfile.mkdtemp(prefix="ledger_"))
ledger.git_commit = lambda: "abc1234"          # pretend clean tree (only for this temp-ledger probe)
ledger.create_study(ledger_dir=d, study_id="fbs-0001-a1", book="FBS", system="toy", issue=1, attempt=1,
                    dev_window="x", cost_model_version="v", cv_scheme="cpcv")
row = ledger.record_holdout_unlock(book="FBS", system="toy", study_id="fbs-0001-a1", pass_band={"sharpe_p10": 0.1},
                                   user_confirmation=ledger.unlock_phrase("FBS", "toy"), ledger_dir=d)
print("agent-composed phrase accepted:", row["event"], "| is_holdout_unlocked:", ledger.is_holdout_unlocked("FBS", "toy", ledger_dir=d))
