import argparse
import pathlib

from RaCoT.tools.common import (
    ROOT,
    add_base_check_arguments,
)
from RaCoT.tools.model_download import (
    DEFAULT_MODEL_ID,
    DEFAULT_TARGET_DIR,
    main as download_model_main,
)
from RaCoT.tools.quality_gate import (
    run_comment_mode,
    run_hygiene_mode,
    run_quality_mode,
)
from RaCoT.tools.repo_hygiene import DEFAULT_MAX_FILE_SIZE_MB


def build_parser():
    parser = argparse.ArgumentParser(description="RaCoT repository tooling.")
    subparsers = parser.add_subparsers(dest="command")

    quality_parser = subparsers.add_parser("quality", help="Run all quality checks.")
    add_base_check_arguments(
        quality_parser,
        default_root=ROOT,
        max_file_size_default=DEFAULT_MAX_FILE_SIZE_MB,
    )

    comment_parser = subparsers.add_parser("comment", help="Run comment policy check.")
    add_base_check_arguments(comment_parser, default_root=ROOT)

    hygiene_parser = subparsers.add_parser("hygiene", help="Run repository hygiene check.")
    add_base_check_arguments(
        hygiene_parser,
        default_root=ROOT,
        max_file_size_default=DEFAULT_MAX_FILE_SIZE_MB,
    )

    download_parser = subparsers.add_parser("download-model", help="Download optional local model.")
    download_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    download_parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    download_parser.add_argument("--skip-install", action="store_true")
    return parser


def _run_download_model(args) -> int:
    return download_model_main(
        [
            "--model-id",
            args.model_id,
            "--target-dir",
            args.target_dir,
            *(["--skip-install"] if args.skip_install else []),
        ]
    )


def main(argv=None) -> int:
    argv = list(argv or [])
    if not argv or argv[0].startswith("-"):
        argv = ["quality", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root).resolve()
    command_handlers = {
        "quality": lambda: run_quality_mode(root, args.max_file_size_mb, args.json),
        "comment": lambda: run_comment_mode(root, args.json),
        "hygiene": lambda: run_hygiene_mode(root, args.max_file_size_mb, args.json),
        "download-model": lambda: _run_download_model(args),
    }
    return command_handlers[args.command]()
