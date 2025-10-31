This project folder focuses on projects relating to optical networking.

There are currently two projects in progress.

The first project is being worked on by Soham, focusing on telemetry data and failure simulations in optical networks using GNPY.

The second project is being work on by Martin, focusing on soft-failure localization
in a abstracted topology using networkX.

## Windows / Python 3.12 quickstart

The `collaborative-rl` runner now includes Windows-friendly defaults. To launch a
local DQN training session on Windows 11 with Python 3.12.5:

1. Open PowerShell in the project directory:
   ```powershell
   cd .\optical-projects-main\collaborative-rl
   ```
2. Install the required packages (Torch stays separate):
   ```powershell
   pip uninstall -y gymnasium
   pip install -r ..\requirements-win-py312.txt
   ```
3. Run the trainer with the folder on `PYTHONPATH` so local modules resolve:
   ```powershell
   $env:PYTHONPATH = "$PWD"
   python .\DQN_runner.py
   ```

The defaults now point to the bundled `data/` directory and create `logs/` on
first run. TensorFlow/Keras is optional?if it is not installed, the environment
falls back to zero-probability predictions while RLlib (Torch backend) continues
to run.
