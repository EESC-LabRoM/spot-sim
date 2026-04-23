# Sets shell to bash to avoid compatibility issues
set shell := ["bash", "-c"]

# Path variables
sim_venv := ".venv/sim_env"

# uv env var so all uv commands use the right venv
export UV_PROJECT_ENVIRONMENT := `pwd` / sim_venv

# Shared ROS domain ID
ros_domain_id := "77"

# ============================================================================
# Main Commands
# ============================================================================

# Lists all available recipes
default:
    @just --list

# Runs an arbitrary Python script inside the simulation environment
# Usage: just run path/to/script.py --arg1 value
run script_path *args:
    source .venv/sim_env/bin/activate && uv run --active python {{script_path}} {{args}}

# Installs via pip using the simulation environment
install +args:
    uv run pip install {{args}} --no-build-isolation

# ============================================================================
# Maintenance
# ============================================================================

# Installs the full simulation environment (submodules, ZED, uv deps)
install-sim:
    bash scripts/tools/install_modules.sh

# Cleans python and simulator caches (if graphical issues occur)
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +

# ============================================================================
# Spot IsaacSim Environment
# ============================================================================

# Runs the IsaacSim Spot spawner — publishes sensor topics (cameras, TF, joint states)
run-spot-sim *args:
    ROS_DOMAIN_ID={{ros_domain_id}} just run scripts/spot_isaacsim/play.py {{args}}

# Runs the IsaacSim Spot spawner via isaacsim --exec
run-spot-isaacsim *args:
    ROS_DOMAIN_ID={{ros_domain_id}} isaacsim --exec "scripts/spot_isaacsim/play.py {{args}}"

# Runs the IsaacSim Spot spawner in headless mode
run-spot-sim-h *args:
    ROS_DOMAIN_ID={{ros_domain_id}} just run scripts/spot_isaacsim/play.py --headless {{args}}
