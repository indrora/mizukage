"""Compile all .proto files in protobuf/ → mizukage/proto/*.py.

Run once during initial setup, or after any proto file changes:
    uv run --extra dev python scripts/compile_protos.py
"""
import sys
import subprocess
from pathlib import Path

root = Path(__file__).parent.parent
proto_dir = root / "protobuf"
out_dir = root / "mizukage" / "proto"
out_dir.mkdir(parents=True, exist_ok=True)

proto_files = sorted(proto_dir.glob("*.proto"))
if not proto_files:
    sys.exit(f"No .proto files found in {proto_dir}")

print(f"Compiling {len(proto_files)} proto files to {out_dir}")
result = subprocess.run(
    [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={out_dir}",
        *[str(f) for f in proto_files],
    ],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)

# Write the __init__.py that patches sys.path for flat proto imports
init_path = out_dir / "__init__.py"
init_path.write_text(
    '"""Generated protobuf bindings for the Light L16 LRI format (ltpb package)."""\n'
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "_here = str(Path(__file__).parent)\n"
    "if _here not in sys.path:\n"
    "    sys.path.insert(0, _here)\n"
)

print(f"Done. {len(list(out_dir.glob('*_pb2.py')))} _pb2 files written.")
print("Commit mizukage/proto/ to include generated files in the repo.")
