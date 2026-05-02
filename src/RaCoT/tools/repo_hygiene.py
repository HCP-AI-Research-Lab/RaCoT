import os
import pathlib

from RaCoT.tools.common import (
    build_base_parser,
    is_path_ignored,
    output_single_check,
)

DEFAULT_MAX_FILE_SIZE_MB = int(os.getenv("REPO_MAX_FILE_SIZE_MB", "20"))
OPTIONAL_ASSET_DIRS = {"models"}
DISALLOWED_SOURCE_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}


def run_check(root: pathlib.Path, max_file_size_mb: int) -> dict:
    violations = []
    scanned = 0
    max_file_size_bytes = max_file_size_mb * 1024 * 1024

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_path_ignored(path):
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in OPTIONAL_ASSET_DIRS:
            continue
        scanned += 1
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if size > max_file_size_bytes:
            violations.append(f"{rel} exceeds {max_file_size_bytes} bytes")
        if rel.parts and rel.parts[0] == "src" and suffix in DISALLOWED_SOURCE_EXTENSIONS:
            violations.append(f"{rel} uses disallowed source extension {suffix}")

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "scanned_files": scanned,
        "max_file_size_mb": max_file_size_mb,
    }


def parse_args(argv=None):
    parser = build_base_parser(
        "Check repository hygiene constraints",
        max_file_size_default=DEFAULT_MAX_FILE_SIZE_MB,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = pathlib.Path(args.root).resolve()
    result = run_check(root, args.max_file_size_mb)
    return output_single_check(
        result=result,
        as_json=args.json,
        success_message=f"Repository hygiene check passed. Scanned {result['scanned_files']} files.",
        failure_title="Repository hygiene check failed:",
    )
