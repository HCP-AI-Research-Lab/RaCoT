import json
import pathlib

from RaCoT.tools.comment_policy import run_check as comment_run_check
from RaCoT.tools.common import ROOT, output_single_check, print_lines
from RaCoT.tools.repo_hygiene import run_check as hygiene_run_check


def _print_quality_section(success_message: str, failure_message: str, result: dict) -> None:
    if result["ok"]:
        print(success_message)
        return
    print(failure_message)
    print_lines(result["violations"])


def run_quality(root: pathlib.Path, max_file_size_mb: int) -> dict:
    comment_result = comment_run_check(root)
    hygiene_result = hygiene_run_check(root, max_file_size_mb)
    ok = comment_result["ok"] and hygiene_result["ok"]
    return {
        "ok": ok,
        "comment_policy": comment_result,
        "repository_hygiene": hygiene_result,
    }


def run_comment_mode(root: pathlib.Path, as_json: bool) -> int:
    result = comment_run_check(root)
    return output_single_check(
        result=result,
        as_json=as_json,
        success_message=f"Comment policy check passed. Checked {result['checked_files']} files.",
        failure_title="Found disallowed comment lines:",
    )


def run_hygiene_mode(root: pathlib.Path, max_file_size_mb: int, as_json: bool) -> int:
    result = hygiene_run_check(root, max_file_size_mb)
    return output_single_check(
        result=result,
        as_json=as_json,
        success_message=f"Repository hygiene check passed. Scanned {result['scanned_files']} files.",
        failure_title="Repository hygiene check failed:",
    )


def run_quality_mode(root: pathlib.Path, max_file_size_mb: int, as_json: bool) -> int:
    result = run_quality(root, max_file_size_mb)
    if as_json:
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["ok"] else 1

    print("Running quality gate: comment policy")
    _print_quality_section(
        success_message="Comment policy check passed.",
        failure_message="Comment policy check failed.",
        result=result["comment_policy"],
    )

    print("Running quality gate: repository hygiene")
    _print_quality_section(
        success_message=(
            f"Repository hygiene check passed. Scanned {result['repository_hygiene']['scanned_files']} files."
        ),
        failure_message="Repository hygiene check failed.",
        result=result["repository_hygiene"],
    )

    if result["ok"]:
        print("Quality gate passed.")
        return 0
    return 1


def main(argv=None) -> int:
    from RaCoT.tools.cli import main as cli_main

    return cli_main(argv)
