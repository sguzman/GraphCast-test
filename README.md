# graphcast-test

Local CLI wrapper for experimenting with Google DeepMind GraphCast from this repo.

It does four things:

1. keeps all downloaded assets under `.cache/graphcast`
2. creates a dedicated Python 3.11 runtime in `.venv`
3. downloads public checkpoint, stats, and sample dataset files from the official `dm_graphcast` bucket
4. runs local forecasts from the terminal and writes outputs as NetCDF

## Constraints

Upstream GraphCast currently targets Python 3.10 and 3.11. This repo is pinned to Python 3.11 because the machine default here is Python 3.14, which is not a supported runtime for upstream GraphCast.

The environment installer uses the official source repo at `google-deepmind/graphcast` tag `v0.2` instead of PyPI, to avoid ambiguity around third-party packages named `graphcast`.

## Quick start

This is a `uv` project. Use `uv run` for commands and let `uv` manage the local `.venv`.

```bash
uv run --python 3.11 main.py init
uv run --python 3.11 main.py env create
uv run --python 3.11 main.py bundle fetch graphcast-small
uv run --python 3.11 main.py forecast --bundle graphcast-small
```

That writes a forecast file under `.cache/graphcast/outputs/`.

Fetching is intended to be idempotent:

- `bundle fetch` skips files that already exist
- `fetch-object` now skips existing files unless you pass `--force`
- `env create` is safe to rerun and will converge the local `.venv`

## Commands

Show local paths:

```bash
uv run --python 3.11 main.py paths
```

List public bucket objects:

```bash
uv run --python 3.11 main.py list --prefix graphcast/params/
```

Fetch one specific object:

```bash
uv run --python 3.11 main.py fetch-object \
  "graphcast/stats/mean_by_level.nc"
```

Force a re-download:

```bash
uv run --python 3.11 main.py fetch-object \
  "graphcast/stats/mean_by_level.nc" \
  --force
```

List curated bundles:

```bash
uv run --python 3.11 main.py bundle list
```

Fetch a bundle:

```bash
uv run --python 3.11 main.py bundle fetch graphcast-small
```

Run a forecast:

```bash
uv run --python 3.11 main.py forecast --bundle graphcast-small --steps 4
```

Inspect a forecast file:

```bash
uv run --python 3.11 main.py inspect \
  ".cache/graphcast/outputs/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.forecast.nc"
```

Plot one regional slice:

```bash
uv run --python 3.11 main.py plot \
  ".cache/graphcast/outputs/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.forecast.nc" \
  --variable 2m_temperature \
  --time-index 0 \
  --lat-min 14 \
  --lat-max 33 \
  --lon-min 250 \
  --lon-max 268 \
  --output ".cache/graphcast/outputs/mexico-2m-temperature-t0.png"
```

Use explicit files instead of a bundle:

```bash
uv run --python 3.11 main.py forecast \
  --checkpoint ".cache/graphcast/assets/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --dataset ".cache/graphcast/assets/graphcast/dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc" \
  --stats-dir ".cache/graphcast/assets/graphcast/stats" \
  --output ".cache/graphcast/outputs/graphcast-small.nc"
```

## Bundles

- `graphcast-small`: 1.0 degree GraphCast checkpoint, matching ERA5 sample dataset, stats files
- `graphcast-operational`: operational 0.25 degree checkpoint, matching HRES sample dataset, stats files
- `graphcast`: 0.25 degree 37-level checkpoint, matching ERA5 sample dataset, stats files

## Outputs

Forecast outputs are written as NetCDF files so you can inspect them with `xarray`, `ncdump`, or downstream notebooks.

GraphCast itself is a global forecast model. That means inference runs on the whole world grid, but you can inspect or plot only a regional subset afterward.
