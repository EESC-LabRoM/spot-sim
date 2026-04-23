"""
Compatibility patches for Isaac Sim / Isaac Lab external packages.

Run automatically by scripts/tools/install_modules.sh during setup.

Patches:
  - Relic actuator_spot.py: named super().__init__() args (Isaac Lab compatibility)
  - Relic assets/__init__.py: pathlib instead of os.path

Container-specific patches (RAM, MGPC, PoinTr, GroundingDINO, SAM2) are handled
separately by agn_grasp/scripts/patch_ros_externals.py.
"""

from pathlib import Path


def patch_relic_actuator(relic_root):
    """
    Patch Relic SpotKneeActuator for Isaac Lab compatibility.
    Fixes dynamic_friction parameter conflict by using explicit named arguments.
    """
    actuator_file = (
        Path(relic_root) / "source" / "relic" / "relic" / "actuators" / "actuator_spot.py"
    )

    if not actuator_file.exists():
        print(f"   ⚠️  Relic actuator_spot.py not found: {actuator_file}")
        return False

    try:
        content = actuator_file.read_text(encoding="utf-8")

        if "# [CORREÇÃO CRÍTICA]" in content or "cfg=cfg," in content:
            print("      ℹ️  Relic actuator already patched")
            return False

        old_super_call = """\
        super().__init__(
            cfg,
            joint_names,
            joint_ids,
            num_envs,
            device,
            stiffness,
            damping,
            armature,
            friction,
            effort_limit,
            velocity_limit,
        )"""

        new_super_call = """\
        # [CORREÇÃO CRÍTICA]
        # Usamos explicitamente argumentos nomeados (ex: cfg=cfg).
        # Isso evita que 'dynamic_friction' seja preenchido acidentalmente por um
        # argumento posicional e depois novamente pelo **kwargs.
        super().__init__(
            cfg=cfg,
            joint_names=joint_names,
            joint_ids=joint_ids,
            num_envs=num_envs,
            device=device,
            stiffness=stiffness,
            damping=damping,
            armature=armature,
            friction=friction,
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
            **kwargs,  # Repassa dynamic_friction, viscous_friction, etc. corretamente
        )"""

        if old_super_call in content:
            actuator_file.write_text(content.replace(old_super_call, new_super_call), encoding="utf-8")
            print("      ✅ Relic actuator_spot.py fixed")
            return True
        else:
            print("      ⚠️  Could not find expected super().__init__() pattern in actuator_spot.py")
            return False

    except Exception as e:
        print(f"      ❌ Error fixing Relic actuator: {e}")
        return False


def patch_relic_assets_init(relic_root):
    """Patch Relic assets/__init__.py to use pathlib instead of os.path."""
    init_file = (
        Path(relic_root) / "source" / "relic" / "relic" / "assets" / "__init__.py"
    )

    if not init_file.exists():
        print(f"   ⚠️  Relic assets/__init__.py not found: {init_file}")
        return False

    try:
        content = init_file.read_text(encoding="utf-8")

        if "from pathlib import Path" in content and "Path(__file__).parent.absolute()" in content:
            print("      ℹ️  Relic assets/__init__.py already uses pathlib")
            return False

        old_code = "import os\n\nASSET_DIR = os.path.dirname(__file__)"
        new_code = "#import os\nfrom pathlib import Path\n\nASSET_DIR = Path(__file__).parent.absolute()"

        if old_code in content:
            init_file.write_text(content.replace(old_code, new_code), encoding="utf-8")
        elif "import os" in content and "os.path.dirname(__file__)" in content:
            content = content.replace("import os", "#import os\nfrom pathlib import Path")
            content = content.replace("os.path.dirname(__file__)", "Path(__file__).parent.absolute()")
            init_file.write_text(content, encoding="utf-8")
        else:
            print("      ℹ️  Relic assets/__init__.py already patched or has unexpected format")
            return False

        print("      ✅ Relic assets/__init__.py fixed")
        return True

    except Exception as e:
        print(f"      ❌ Error fixing Relic assets init: {e}")
        return False


def main():
    print("=== Relic Patches (Isaac Sim / Isaac Lab) ===")

    scan_root = Path(__file__).resolve().parent.parent.parent / "external"

    if not scan_root.exists():
        print(f"❌ 'external' folder not found at {scan_root}")
        return

    print(f"   External dir: {scan_root}")

    print(f"\n📂 Applying patch for Relic (Isaac Lab compatibility)")
    for relic_dir in [scan_root / "relic", scan_root / "Relic"]:
        if relic_dir.exists():
            patch_relic_actuator(relic_dir)
            patch_relic_assets_init(relic_dir)
            break
    else:
        print("   ⚠️  Relic directory not found")

    print("\n=== Completed ===")


if __name__ == "__main__":
    main()
