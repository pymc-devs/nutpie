"""Build a tagged experimental wheel and checksum manifest from the release binary."""

import argparse
import base64
import csv
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--allow-dirty", action="store_true", help="For local probes only")
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root))
if dirty and not args.allow_dirty:
    raise SystemExit(
        "Commit the source before packaging, or use --allow-dirty for a local probe"
    )
version = "0.16.12+wasm.1"
tag = "cp313-cp313-emscripten_4_0_9_wasm32"
info = f"nutpie-{version}.dist-info"
revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True
).strip()
files = {
    str(p.relative_to(root / "python")): p.read_bytes()
    for p in sorted((root / "python/nutpie").rglob("*.py"))
}
files["nutpie/_lib.so"] = (
    root / "target/wasm32-unknown-emscripten/release/_lib.wasm"
).read_bytes()
files[f"{info}/METADATA"] = f"""Metadata-Version: 2.4
Name: nutpie
Version: {version}
Summary: Experimental PyMC-only Nutpie extension for the pinned Xeus runtime
Requires-Python: ==3.13.*
Requires-Dist: numpy>=2
Requires-Dist: pandas>=2
Requires-Dist: pymc==6.2.0
Requires-Dist: pytensor==3.2.4
Requires-Dist: numba==0.66.0
Requires-Dist: arviz>=1,<2
Requires-Dist: xarray>=2026.7.0
License-Expression: MIT
License-File: LICENSE
Project-URL: Source, https://github.com/twiecki/nutpie/tree/{revision}

Experimental WASM build. Provides compile_pymc_model and a limited synchronous sample returning
ArviZ/xarray results. The full native sample/Arrow/Stan API is not available. Requires the tested Xeus runtime.
""".encode()
files[f"{info}/WHEEL"] = (
    f"Wheel-Version: 1.0\nGenerator: nutpie-wasm-probe\nRoot-Is-Purelib: false\nTag: {tag}\n".encode()
)
files[f"{info}/licenses/LICENSE"] = (root / "LICENSE").read_bytes()
# Include notices for the target dependency graph, including build dependencies.
meta = json.loads(
    subprocess.check_output(
        [
            "cargo",
            "metadata",
            "--locked",
            "--format-version",
            "1",
            "--filter-platform",
            "wasm32-unknown-emscripten",
        ],
        cwd=root,
    )
)
notices = []
for package in meta["packages"]:
    if package["name"] == "nutpie":
        continue
    folder = Path(package["manifest_path"]).parent
    paths = []
    for pattern in ["LICENSE*", "LICENCE*", "COPYING*", "NOTICE*", "COPYRIGHT*"]:
        for p in sorted(folder.glob(pattern)):
            if p.is_file():
                name = (
                    f"{info}/licenses/{package['name']}-{package['version']}/{p.name}"
                )
                files[name] = p.read_bytes()
                paths.append(p.name)
    notices.append(
        {
            "name": package["name"],
            "version": package["version"],
            "license": package.get("license"),
            "files": paths,
        }
    )
files[f"{info}/licenses/manifest.json"] = (
    json.dumps(notices, indent=2) + "\n"
).encode()
# Rebuilt Rust std also carries these upstream licenses.
sysroot = Path(
    subprocess.check_output(
        ["rustc", "+nightly", "--print", "sysroot"], text=True
    ).strip()
)
for name in ["LICENSE-APACHE", "LICENSE-MIT"]:
    p = sysroot / "share/doc/rust" / name
    if p.exists():
        files[f"{info}/licenses/rust-std/{name}"] = p.read_bytes()
build = {
    "source_revision": revision,
    "source_dirty": dirty,
    "extension_sha256": hashlib.sha256(files["nutpie/_lib.so"]).hexdigest(),
    "version": version,
    "tag": tag,
    "emscripten": "4.0.9",
    "rust": subprocess.check_output(
        ["rustc", "+nightly", "--version"], text=True
    ).strip(),
    "profile": "release",
    "runtime_sha256": "dc73b5f69ef1946e3409f3ab0884a2b17a3f4d1956069ce0b70f0fc51665a6f4",
}
files["nutpie/wasm-build.json"] = (json.dumps(build, indent=2) + "\n").encode()
record = io.StringIO()
writer = csv.writer(record, lineterminator="\n")
for name, data in sorted(files.items()):
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )
    writer.writerow([name, "sha256=" + digest, len(data)])
writer.writerow([f"{info}/RECORD", "", ""])
files[f"{info}/RECORD"] = record.getvalue().encode()
output = root / "target/wasm-dist"
output.mkdir(exist_ok=True)
wheel = output / f"nutpie-{version}-{tag}.whl"
with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
    for name, data in sorted(files.items()):
        entry = zipfile.ZipInfo(name, (2026, 9, 11, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = 0o644 << 16
        archive.writestr(entry, data)
manifest = {
    **build,
    "filename": wheel.name,
    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "bytes": wheel.stat().st_size,
    "status": "experimental",
    "api": "compile_pymc_model, limited sample (ArviZ/xarray), sample_raw; no drop-in JS sampler API",
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(output / "SHA256SUMS").write_text(manifest["sha256"] + "  " + wheel.name + "\n")
print(wheel)
