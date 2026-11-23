"""Run the assignment in order. Training is the slow step (30 epochs)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def run(rel):
    print("\n===", rel, "===")
    subprocess.check_call([sys.executable, str(ROOT / rel)])


if __name__ == "__main__":
    run("src/download_data.py")
    run("src/load_data.py")
    run("src/plot_windows.py")
    run("src/prepare.py")
    run("src/model.py")
    run("src/train.py")
    run("src/test_model.py")
    run("src/export_web.py")
    run("src/write_report.py")
    print("\nall done.  python app/serve.py")
