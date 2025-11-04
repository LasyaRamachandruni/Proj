This project folder focuses on projects relating to optical networking.

There are currently two projects in progress.

The first project is being worked on by Soham, focusing on telemetry data and failure simulations in optical networks using GNPY.

The second project is being work on by Martin, focusing on soft-failure localization
in a abstracted topology using networkX.

## Windows Quickstart (Python 3.12)

- Open PowerShell in `optical-projects-main\collaborative-rl` (the folder that contains `DQN_runner.py`).
- Install the runtime dependencies:
  ```powershell
  pip install -r ..\requirements-win-py312.txt
  ```
- Verify RLlib and Gymnasium load correctly:
  ```powershell
  python - <<'PY'
from ray.rllib.algorithms.dqn import DQNConfig
import gymnasium
print("OK: RLlib + Gymnasium", getattr(gymnasium, "__version__", "unknown"))
PY
  ```
- Point Python at the local modules and launch the runner:
  ```powershell
  $env:PYTHONPATH="$PWD"
  python .\DQN_runner.py
  ```

The runner reads `.env` if present; when variables are missing it falls back to the bundled data in `../data/toy_2-20_rounds-20_reqs`, writes logs to `../logs`, and searches for broken-fiber snapshots in `../data`.

TensorFlow/Keras are optional. Install them only if you need to load the legacy optical-failure classifier:
```powershell
pip install keras==3.4.1 tensorflow==2.16.1
```
