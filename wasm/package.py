"""Package the prototype extension and the real Nutpie Python sources."""

import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
output = root / "target/nutpie-probe.zip"
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((root / "python/nutpie").rglob("*.py")):
        archive.write(path, str(path.relative_to(root / "python")))
    archive.write(
        root / "target/wasm32-unknown-emscripten/release/_lib.wasm", "nutpie/_lib.so"
    )
print(output)
