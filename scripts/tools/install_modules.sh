#!/bin/bash
# =============================================================================
#   Spot IsaacSim - Simulation Environment Setup
# =============================================================================
# Sets up the simulation Python environment using uv.
#
# This script installs only what IsaacSim and IsaacLab need:
#   - Isaac Sim 5.1 + Isaac Lab (via pip)
#   - Relic (Spot URDF/USD assets)
#   - CuRobo (motion planning / IK)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="$SIM_ROOT/.venv/sim_env"
export UV_PROJECT_ENVIRONMENT="$VENV"

echo "=========================================================="
echo "       STARTING ISAAC SIM / LAB DEPENDENCIES SETUP"
echo "=========================================================="

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[0/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 1. System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt install -y cmake build-essential libboost-dev libboost-thread-dev libpcl-dev

# 2. Submodules
echo "[2/5] Initializing submodules..."
cd "$SIM_ROOT"
git submodule update --init --recursive -- external/relic
git submodule update --init --recursive -- external/curobo
git submodule update --init --recursive -- external/zed-isaac-sim
git submodule update --init --recursive -- external/IsaacRobotics

# 3. CUDA environment
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

grep -q "CUDA_HOME" ~/.bashrc || {
    echo "# Isaac DL Grasp Variables" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda-12.8" >> ~/.bashrc
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
}

# 3. Build ZED Isaac Sim extension
echo "[3/6] Building ZED Isaac Sim extension..."
ZED_EXT_DIR="$SIM_ROOT/external/zed-isaac-sim"
ZED_LIB_VERSION="4.2.0"
ZED_LIB_ARCHIVE="$ZED_EXT_DIR/exts/sl.sensor.camera/bin/libsl_zed_linux_x86_64_${ZED_LIB_VERSION}.tar.gz"
ZED_LIB_BUILD_PATH="$ZED_EXT_DIR/_build/linux-x86_64/release/exts/sl.sensor.camera/bin"
ZED_LIB_SO="$ZED_EXT_DIR/exts/sl.sensor.camera/bin/libsl_zed.so"

mkdir -p "$ZED_EXT_DIR/exts/sl.sensor.camera/bin/" "$ZED_LIB_BUILD_PATH"

if [ ! -f "$ZED_LIB_SO" ]; then
    echo "   Downloading libsl_zed ${ZED_LIB_VERSION}..."
    wget --no-check-certificate \
        -O "$ZED_LIB_ARCHIVE" \
        "https://stereolabs.sfo2.cdn.digitaloceanspaces.com/utils/zed_isaac_sim/${ZED_LIB_VERSION}/libsl_zed_linux_x86_64_${ZED_LIB_VERSION}.tar.gz"
    tar -xzf "$ZED_LIB_ARCHIVE" -C "$ZED_EXT_DIR/exts/sl.sensor.camera/bin/"
    cp "$ZED_LIB_SO" "$ZED_LIB_BUILD_PATH/libsl_zed.so"
else
    echo "   libsl_zed.so already present, skipping download."
    # Ensure build path copy exists for build.sh skip check
    [ -f "$ZED_LIB_BUILD_PATH/libsl_zed.so" ] || cp "$ZED_LIB_SO" "$ZED_LIB_BUILD_PATH/libsl_zed.so"
fi

cd "$ZED_EXT_DIR"
./build.sh
cd "$SIM_ROOT"
echo "   ZED extension built successfully."

# 4. Build temp dir (prevents /tmp full errors during curobo compilation)
echo "[4/6] Configuring build environment..."
mkdir -p ~/tmp_build
export TMPDIR=~/tmp_build
echo "   TMPDIR set to: $TMPDIR"

# 5. Install via uv
echo "[5/6] Installing Python environment via uv..."
cd "$SIM_ROOT"
uv venv "$VENV" --python python3.11 --system-site-packages
source $VENV/bin/activate
uv pip install --python "$PYTHON" \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv sync --inexact

# 6. Write variables to activate file
echo "# Isaac AGN Grasp Configs" >> $VENV/bin/activate
echo "export UV_PROJECT_ENVIRONMENT=$VENV" >> $VENV/bin/activate
echo "export ROS_DISTRO=\${ROS_DISTRO:-jazzy}" >> $VENV/bin/activate
echo "export RMW_IMPLEMENTATION=\${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}" >> $VENV/bin/activate
echo "export LD_LIBRARY_PATH=\"\${VIRTUAL_ENV}/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/\${ROS_DISTRO}/lib:\${LD_LIBRARY_PATH}\"" >> $VENV/bin/activate
source $VENV/bin/activate

echo "[6/6] Done."
echo "=========================================================="
echo "   Simulation setup complete!"
echo "=========================================================="
echo ""
echo "Start the simulation:"
echo "   just run-spot-sim"
echo ""
