"""Stage the probe beside an already extracted nuts-rs-wasm v0.1.0 runtime."""

import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("nuts_checkout", type=Path)
parser.add_argument("site", type=Path)
parser.add_argument("--benchmarks", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
assert (args.site / "runtime/comlink.worker.js").is_file(), "Extract the runtime first"
for source, destination in [
    (root / "wasm/probe.html", "index.html"),
    (root / "wasm/probe.py", "probe.py"),
    (root / "target/nutpie-probe.zip", "nutpie-probe.zip"),
    (args.nuts_checkout / "client.mjs", "client.mjs"),
    (args.nuts_checkout / "browser-artifact/comlink.mjs", "comlink.mjs"),
    (args.nuts_checkout / "examples/mmm/model.py", "mmm_model.py"),
    (args.nuts_checkout / "examples/mmm/mmm_example.csv", "mmm_example.csv"),
]:
    shutil.copyfile(source, args.site / destination)

if args.benchmarks:
    # The baseline checkout must have completed npm ci && npm run build.
    shutil.copytree(
        args.nuts_checkout / "browser-artifact",
        args.site / "nuts",
        dirs_exist_ok=True,
    )
    for name in ["compile_model.py", "results.py"]:
        shutil.copyfile(args.nuts_checkout / name, args.site / name)
    for source in (root / "wasm/benchmarks").glob("*.py"):
        shutil.copyfile(source, args.site / source.name)
    for source in (root / "target/wasm-dist").glob("*"):
        shutil.copyfile(source, args.site / source.name)
