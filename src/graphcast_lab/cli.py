from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from graphcast_lab import __version__


UPSTREAM_TAG = "v0.2"
BUCKET = "dm_graphcast"
BUCKET_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
DOWNLOAD_API = f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o"

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / ".cache" / "graphcast"
ASSETS_ROOT = CACHE_ROOT / "assets"
OUTPUTS_ROOT = CACHE_ROOT / "outputs"
VENV_DIR = REPO_ROOT / ".venv"

DEFAULT_STATS = [
    "graphcast/stats/diffs_stddev_by_level.nc",
    "graphcast/stats/mean_by_level.nc",
    "graphcast/stats/stddev_by_level.nc",
]

BUNDLES: dict[str, list[str]] = {
    "graphcast-small": [
        "graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
        "graphcast/dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc",
        *DEFAULT_STATS,
    ],
    "graphcast-operational": [
        "graphcast/params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",
        "graphcast/dataset/source-hres_date-2022-01-01_res-0.25_levels-13_steps-04.nc",
        *DEFAULT_STATS,
    ],
    "graphcast": [
        "graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",
        "graphcast/dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-04.nc",
        *DEFAULT_STATS,
    ],
}


def cache_path_for_object(object_name: str) -> Path:
    return ASSETS_ROOT / object_name


def ensure_dirs() -> None:
    for path in (CACHE_ROOT, ASSETS_ROOT, OUTPUTS_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def list_bucket(prefix: str, max_results: int) -> list[dict]:
    query = urllib.parse.urlencode({"prefix": prefix, "maxResults": str(max_results)})
    payload = fetch_json(f"{BUCKET_API}?{query}")
    return payload.get("items", [])


def download_object(object_name: str, destination: Path) -> Path:
    ensure_dirs()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = urllib.parse.quote(object_name, safe="")
    url = f"{DOWNLOAD_API}/{encoded}?alt=media"
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return destination


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def venv_exists() -> bool:
    return venv_python().exists()


def cmd_init(_: argparse.Namespace) -> int:
    ensure_dirs()
    print(CACHE_ROOT)
    return 0


def cmd_paths(_: argparse.Namespace) -> int:
    ensure_dirs()
    print(f"repo:      {REPO_ROOT}")
    print(f"cache:     {CACHE_ROOT}")
    print(f"assets:    {ASSETS_ROOT}")
    print(f"outputs:   {OUTPUTS_ROOT}")
    print(f"venv:      {VENV_DIR}")
    print(f"python:    {venv_python()}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    items = list_bucket(prefix=args.prefix, max_results=args.max_results)
    for item in items:
        print(f"{item['name']}\t{item['size']}")
    return 0


def cmd_fetch_object(args: argparse.Namespace) -> int:
    object_name = args.object_name
    destination = Path(args.output) if args.output else cache_path_for_object(object_name)
    path = download_object(object_name, destination)
    print(path)
    return 0


def cmd_bundle_list(_: argparse.Namespace) -> int:
    for name in sorted(BUNDLES):
        print(name)
    return 0


def cmd_bundle_fetch(args: argparse.Namespace) -> int:
    bundle = BUNDLES[args.bundle]
    for object_name in bundle:
        path = cache_path_for_object(object_name)
        if path.exists() and not args.force:
            print(f"exists\t{path}")
            continue
        print(f"fetch\t{object_name}")
        download_object(object_name, path)
    return 0


def cmd_env_create(args: argparse.Namespace) -> int:
    ensure_dirs()
    python = args.python
    run(["uv", "venv", "--python", python, str(VENV_DIR)])
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python()),
            f"git+https://github.com/google-deepmind/graphcast.git@{UPSTREAM_TAG}",
            "absl-py",
        ]
    )
    run(["uv", "pip", "install", "--python", str(venv_python()), "-e", ".", "--no-deps"])
    print(venv_python())
    return 0


def cmd_env_check(_: argparse.Namespace) -> int:
    print(f"venv_exists={venv_exists()}")
    if venv_exists():
        run([str(venv_python()), "--version"])
        return 0
    return 1


def resolve_default_assets(bundle_name: str) -> tuple[Path, Path, Path]:
    bundle = BUNDLES[bundle_name]
    checkpoint = next(cache_path_for_object(item) for item in bundle if "/params/" in item)
    dataset = next(cache_path_for_object(item) for item in bundle if "/dataset/" in item)
    stats_dir = cache_path_for_object("graphcast/stats")
    return checkpoint, dataset, stats_dir


def cmd_forecast(args: argparse.Namespace) -> int:
    if not venv_exists():
        raise SystemExit("Run `graphcast-lab env create` first.")

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    dataset = Path(args.dataset) if args.dataset else None
    stats_dir = Path(args.stats_dir) if args.stats_dir else None

    if args.bundle:
        default_checkpoint, default_dataset, default_stats_dir = resolve_default_assets(args.bundle)
        checkpoint = checkpoint or default_checkpoint
        dataset = dataset or default_dataset
        stats_dir = stats_dir or default_stats_dir

    if checkpoint is None or dataset is None or stats_dir is None:
        raise SystemExit("Provide `--bundle` or all of `--checkpoint`, `--dataset`, and `--stats-dir`.")

    output = Path(args.output) if args.output else OUTPUTS_ROOT / f"{dataset.stem}.forecast.nc"
    output.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    cmd = [
        str(venv_python()),
        "-m",
        "graphcast_lab.runtime",
        "forecast",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(dataset),
        "--stats-dir",
        str(stats_dir),
        "--output",
        str(output),
        "--steps",
        str(args.steps),
    ]
    run(cmd, env=env)
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphcast-lab")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create local cache directories")
    init_parser.set_defaults(func=cmd_init)

    paths_parser = subparsers.add_parser("paths", help="Show repo and cache paths")
    paths_parser.set_defaults(func=cmd_paths)

    list_parser = subparsers.add_parser("list", help="List objects in the public GraphCast bucket")
    list_parser.add_argument("--prefix", default="graphcast/")
    list_parser.add_argument("--max-results", type=int, default=50)
    list_parser.set_defaults(func=cmd_list)

    fetch_parser = subparsers.add_parser("fetch-object", help="Download one bucket object into the local cache")
    fetch_parser.add_argument("object_name")
    fetch_parser.add_argument("--output")
    fetch_parser.set_defaults(func=cmd_fetch_object)

    bundle_parser = subparsers.add_parser("bundle", help="List or fetch curated asset bundles")
    bundle_subparsers = bundle_parser.add_subparsers(dest="bundle_command", required=True)

    bundle_list_parser = bundle_subparsers.add_parser("list")
    bundle_list_parser.set_defaults(func=cmd_bundle_list)

    bundle_fetch_parser = bundle_subparsers.add_parser("fetch")
    bundle_fetch_parser.add_argument("bundle", choices=sorted(BUNDLES))
    bundle_fetch_parser.add_argument("--force", action="store_true")
    bundle_fetch_parser.set_defaults(func=cmd_bundle_fetch)

    env_parser = subparsers.add_parser("env", help="Manage the GraphCast runtime environment")
    env_subparsers = env_parser.add_subparsers(dest="env_command", required=True)

    env_create_parser = env_subparsers.add_parser("create")
    env_create_parser.add_argument("--python", default="3.11")
    env_create_parser.set_defaults(func=cmd_env_create)

    env_check_parser = env_subparsers.add_parser("check")
    env_check_parser.set_defaults(func=cmd_env_check)

    forecast_parser = subparsers.add_parser("forecast", help="Run a local forecast using the dedicated GraphCast venv")
    forecast_parser.add_argument("--bundle", choices=sorted(BUNDLES))
    forecast_parser.add_argument("--checkpoint")
    forecast_parser.add_argument("--dataset")
    forecast_parser.add_argument("--stats-dir")
    forecast_parser.add_argument("--output")
    forecast_parser.add_argument("--steps", type=int, default=4)
    forecast_parser.set_defaults(func=cmd_forecast)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
