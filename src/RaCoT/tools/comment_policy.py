import io
import pathlib
import tokenize

from RaCoT.tools.common import (
    build_base_parser,
    is_path_ignored,
    output_single_check,
)

LEGAL_COMMENT_FILES = {
    pathlib.Path("src/RaCoT/evaluator/_bleu.py"),
    pathlib.Path("src/RaCoT/generator/fid.py"),
    pathlib.Path("src/RaCoT/refiner/llmlingua_compressor.py"),
    pathlib.Path("src/RaCoT/refiner/selective_context_compressor.py"),
}
LEGAL_KEYWORDS = (
    "Copyright",
    "License",
    "Licensed",
    "Source:",
    "source:",
    "http://",
    "https://",
    "Implementation of",
    "This software is released",
    "All Rights Reserved",
    "limitations under the License",
    "WITHOUT WARRANTIES OR CONDITIONS",
)
LEGAL_COMMENT_LINES = {
    "#",
    "# ===============================================================================",
    "# ==============================================================================",
}


def is_allowed_comment(rel: pathlib.Path, line: str) -> bool:
    if rel not in LEGAL_COMMENT_FILES:
        return False
    stripped = line.strip()
    if stripped in LEGAL_COMMENT_LINES:
        return True
    return any(keyword in stripped for keyword in LEGAL_KEYWORDS)


def run_check(root: pathlib.Path) -> dict:
    violations = []
    checked_files = 0

    for path in root.rglob("*.py"):
        if is_path_ignored(path):
            continue
        rel = path.relative_to(root)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        checked_files += 1
        reader = io.StringIO(source).readline
        try:
            for tok in tokenize.generate_tokens(reader):
                if tok.type != tokenize.COMMENT:
                    continue
                if is_allowed_comment(rel, tok.string):
                    continue
                violations.append(f"{rel}:{tok.start[0]}")
        except tokenize.TokenError:
            continue

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "checked_files": checked_files,
    }


def parse_args(argv=None):
    parser = build_base_parser("Check repository comment policy.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = pathlib.Path(args.root).resolve()
    result = run_check(root)
    return output_single_check(
        result=result,
        as_json=args.json,
        success_message=f"Comment policy check passed. Checked {result['checked_files']} files.",
        failure_title="Found disallowed comment lines:",
    )
