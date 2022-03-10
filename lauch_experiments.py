import subprocess
from pathlib import Path

def lauch_experiment(path):
    command = f"sbatch {path}"
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"command {e.cmd} exited with error code {e.returncode}.")

EXP_DIR = Path("./experiments")
for exp in EXP_DIR.iterdir():
    for script in exp.glob("*.sh"):
        lauch_experiment(script.resolve().__str__())
