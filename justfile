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

# Syncs the simulation environment (useful after git pull or dependency changes)
install-sim:
    uv sync

# Cleans python and simulator caches (if graphical issues occur)
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +

# ============================================================================
# Spot IsaacSim Environment
# ============================================================================

# Converts URDF to USD (one-time conversion)
convert-spot-urdf:
    just run scripts/tools/convert_urdf_to_usd.py --urdf assets/spot/spot_with_arm.urdf --output assets/spot/spot_with_arm.usd

# Converts OBJ/GLB file(s) to USD with collision geometry for Isaac Sim
# Usage: just convert-obj <input.obj|input.glb|dir/> [output.usd|dir/] [collision_type]
# Collision types: convexDecomposition (default), convexHull, meshSimplification, none
convert-obj input output="" collision="convexDecomposition":
    just run scripts/tools/convert_obj_to_usd.py --input {{input}} {{ if output != "" { "--output " + output } else { "" } }} --collision {{collision}}

# Runs the IsaacSim Spot spawner — publishes sensor topics (cameras, TF, joint states)
run-spot-sim *args:
    ROS_DOMAIN_ID={{ros_domain_id}} just run scripts/spot_isaacsim/play.py {{args}}

# Runs the IsaacSim Spot spawner via isaacsim --exec
run-spot-isaacsim *args:
    ROS_DOMAIN_ID={{ros_domain_id}} isaacsim --exec "scripts/spot_isaacsim/play.py {{args}}"

# Runs the IsaacSim Spot spawner in headless mode
run-spot-sim-h *args:
    ROS_DOMAIN_ID={{ros_domain_id}} just run scripts/spot_isaacsim/play.py --headless {{args}}
