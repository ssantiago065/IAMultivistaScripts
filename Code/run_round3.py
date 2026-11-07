"""Runs the whole round-3 batch unattended and keeps Windows from idle-sleeping meanwhile.

Order: tests, exp_round3 (20 runs), then exp_allsub random and disjoint (20 runs each,
after the all-subtype dataset exists). Every step resumes where it stopped, so the
script can simply be started again after an interruption. Log: Analysis/round3/run_log.txt

    python Code/run_round3.py
"""

import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "Analysis" / "round3" / "run_log.txt"
DATASET_A = "H:/Dataset/dataset_expA_8clases.csv"
DATASET_ALL = Path("H:/Dataset/dataset_allsub_8clases.csv")
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
POLL_SECONDS = 60

STEPS = [
    ["-m", "pytest", "test_round3.py", "test_controls.py", "-q", "-p", "no:cacheprovider"],
    ["Code/exp_round3.py", "--dataset", DATASET_A],
    ["Code/exp_allsub.py", "--dataset", str(DATASET_ALL), "--protocol", "random"],
    ["Code/exp_allsub.py", "--dataset", str(DATASET_ALL), "--protocol", "disjoint"],
]


def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        for step in STEPS:
            if "exp_allsub.py" in step[0]:
                while not DATASET_ALL.exists():
                    time.sleep(POLL_SECONDS)
            log("START " + " ".join(step))
            with LOG.open("a", encoding="utf-8") as f:
                cwd = ROOT / "Code" if step[0] == "-m" else ROOT
                code = subprocess.call([sys.executable, *step], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
            log(f"END code={code}")
            if code != 0:
                log("STOPPED: step failed")
                return code
        log("ALL DONE")
        return 0
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    sys.exit(main())
