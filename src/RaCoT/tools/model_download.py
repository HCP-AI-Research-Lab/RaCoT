import argparse
import pathlib
import subprocess

from RaCoT.tools.common import ROOT

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TARGET_DIR = ROOT / "models" / "Qwen_1.5b"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download optional local RaCoT model.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--target-dir",
        default=str(DEFAULT_TARGET_DIR),
        help="Target directory for downloaded model files.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing huggingface_hub CLI.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    target_dir = pathlib.Path(args.target_dir).expanduser().resolve()
    if not args.skip_install:
        subprocess.run(
            ["python", "-m", "pip", "install", "-U", "huggingface_hub[cli]"],
            check=True,
        )
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            args.model_id,
            "--local-dir",
            str(target_dir),
            "--local-dir-use-symlinks",
            "False",
        ],
        check=True,
    )
    print(f"Downloaded {args.model_id} to {target_dir}")
    return 0
