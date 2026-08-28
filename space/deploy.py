"""Create (or update) the Hugging Face Space that hosts the live demo.

Run from this directory, on a machine that can reach huggingface.co:

    pip install "huggingface_hub>=0.26"
    huggingface-cli login          # or export HF_TOKEN=hf_...
    python deploy.py

It uploads the app plus the two statute PDFs; Hugging Face then builds the image,
which is where the vector index gets compiled. First build takes roughly ten
minutes. Re-run the script any time to redeploy.
"""

import argparse
import pathlib
import sys

from huggingface_hub import HfApi

# Files the Space needs, uploaded from this directory.
APP_FILES = ["README.md", "Dockerfile", "requirements.txt",
             "build_index.py", "rag.py", "app.py"]

# Statute PDFs, taken from the repo root and placed under sources/ in the Space.
SOURCE_PDFS = [
    "BNS Delhi police 2nd edition.pdf",
    "Indian Penal Code Book.pdf",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=None,
                        help="Space id, e.g. aasarmehdi/legal-rag-ipc-bns. "
                             "Defaults to <your username>/legal-rag-ipc-bns.")
    parser.add_argument("--repo-root", default="..", type=pathlib.Path,
                        help="Where the statute PDFs live (default: parent directory).")
    parser.add_argument("--private", action="store_true",
                        help="Create the Space private. A recruiter cannot open a "
                             "private Space, so leave this off for a CV link.")
    args = parser.parse_args()

    here = pathlib.Path(__file__).parent
    api = HfApi()

    try:
        username = api.whoami()["name"]
    except Exception:
        sys.exit("Not logged in. Run `huggingface-cli login` or set HF_TOKEN.")

    repo_id = args.repo_id or f"{username}/legal-rag-ipc-bns"

    missing = [p for p in SOURCE_PDFS if not (args.repo_root / p).exists()]
    if missing:
        sys.exit(f"Statute PDFs not found in {args.repo_root.resolve()}: {missing}\n"
                 f"Pass --repo-root pointing at the repository root.")

    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    print(f"space: https://huggingface.co/spaces/{repo_id}")

    for name in APP_FILES:
        api.upload_file(path_or_fileobj=str(here / name), path_in_repo=name,
                        repo_id=repo_id, repo_type="space")
        print(f"  uploaded {name}")

    for name in SOURCE_PDFS:
        api.upload_file(path_or_fileobj=str(args.repo_root / name),
                        path_in_repo=f"sources/{name}",
                        repo_id=repo_id, repo_type="space")
        print(f"  uploaded sources/{name}")

    print("\nBuilding now — watch the Logs tab. Roughly ten minutes for a first build.")
    print("Then set HF_TOKEN under Settings -> Variables and secrets to enable")
    print("answer generation; without it the demo runs retrieval-only.")
    print(f"\nLive URL: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
