import argparse
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[3]
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
    "output",
    "tmp",
}
SKIP_SUFFIXES = {".egg-info"}


def is_path_ignored(path: pathlib.Path, extra_skip_dirs=None) -> bool:
    skip_dirs = SKIP_DIRS | set(extra_skip_dirs or ())
    return any(
        part in skip_dirs or any(part.endswith(suffix) for suffix in SKIP_SUFFIXES)
        for part in path.parts
    )


def print_lines(lines: list[str]) -> None:
    for item in lines:
        print(item)


def add_root_argument(parser, default_root: pathlib.Path = ROOT) -> None:
    parser.add_argument(
        "--root",
        default=str(default_root),
        help="Repository root path. Defaults to current project root.",
    )


def add_json_argument(parser) -> None:
    parser.add_argument("--json", action="store_true", help="Output structured JSON.")


def add_max_file_size_argument(parser, default: int) -> None:
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=default,
        help="Maximum allowed file size in megabytes outside optional asset dirs.",
    )


def add_base_check_arguments(
    parser,
    default_root: pathlib.Path = ROOT,
    max_file_size_default: int | None = None,
) -> None:
    add_root_argument(parser, default_root=default_root)
    if max_file_size_default is not None:
        add_max_file_size_argument(parser, default=max_file_size_default)
    add_json_argument(parser)


def build_base_parser(
    description: str,
    default_root: pathlib.Path = ROOT,
    max_file_size_default: int | None = None,
):
    parser = argparse.ArgumentParser(description=description)
    add_base_check_arguments(
        parser,
        default_root=default_root,
        max_file_size_default=max_file_size_default,
    )
    return parser


def output_single_check(
    result: dict,
    as_json: bool,
    success_message: str,
    failure_title: str,
) -> int:
    if as_json:
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["ok"] else 1
    if result["ok"]:
        print(success_message)
        return 0
    print(failure_title)
    print_lines(result["violations"])
    return 1
