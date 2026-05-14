from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path


def _build_forecast_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    forecast = subparsers.add_parser("forecast")
    forecast.add_argument("--checkpoint", required=True)
    forecast.add_argument("--dataset", required=True)
    forecast.add_argument("--stats-dir", required=True)
    forecast.add_argument("--output", required=True)
    forecast.add_argument("--steps", type=int, default=4)
    forecast.set_defaults(func=forecast_main)


def _build_inspect_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--dataset", required=True)
    inspect.set_defaults(func=inspect_main)


def _build_plot_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    plot = subparsers.add_parser("plot")
    plot.add_argument("--dataset", required=True)
    plot.add_argument("--variable", required=True)
    plot.add_argument("--time-index", type=int, default=0)
    plot.add_argument("--level", type=int)
    plot.add_argument("--lat-min", type=float)
    plot.add_argument("--lat-max", type=float)
    plot.add_argument("--lon-min", type=float)
    plot.add_argument("--lon-max", type=float)
    plot.add_argument("--output", required=True)
    plot.set_defaults(func=plot_main)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m graphcast_lab.runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_forecast_parser(subparsers)
    _build_inspect_parser(subparsers)
    _build_plot_parser(subparsers)
    return parser


def forecast_main(args: argparse.Namespace) -> int:
    import haiku as hk
    import jax
    import numpy as np
    import xarray
    from graphcast import autoregressive
    from graphcast import casting
    from graphcast import checkpoint
    from graphcast import data_utils
    from graphcast import graphcast
    from graphcast import normalization

    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    stats_dir = Path(args.stats_dir)
    output_path = Path(args.output)

    with checkpoint_path.open("rb") as handle:
        ckpt = checkpoint.load(handle, graphcast.CheckPoint)

    example_batch = xarray.open_dataset(dataset_path, decode_timedelta=True).load()
    diffs_stddev_by_level = xarray.load_dataset(stats_dir / "diffs_stddev_by_level.nc").compute()
    mean_by_level = xarray.load_dataset(stats_dir / "mean_by_level.nc").compute()
    stddev_by_level = xarray.load_dataset(stats_dir / "stddev_by_level.nc").compute()

    target_lead_times = slice("6h", f"{args.steps * 6}h")
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        input_variables=ckpt.task_config.input_variables,
        target_variables=ckpt.task_config.target_variables,
        forcing_variables=ckpt.task_config.forcing_variables,
        pressure_levels=ckpt.task_config.pressure_levels,
        input_duration=ckpt.task_config.input_duration,
        target_lead_times=target_lead_times,
    )

    def construct_wrapped_graphcast():
        predictor = graphcast.GraphCast(ckpt.model_config, ckpt.task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    run_forward_jitted = functools.partial(
        run_forward.apply,
        ckpt.params,
        {},
        jax.random.PRNGKey(0),
    )

    run_forward_jitted = jax.jit(run_forward_jitted)

    predictions, _ = run_forward_jitted(
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_netcdf(output_path)
    print(output_path)
    return 0


def inspect_main(args: argparse.Namespace) -> int:
    import xarray

    dataset_path = Path(args.dataset)
    ds = xarray.open_dataset(dataset_path, decode_timedelta=True)

    print(f"path: {dataset_path}")
    print(f"dims: {dict(ds.sizes)}")
    print("coords:")
    for name, coord in ds.coords.items():
        values = coord.values
        if values.size:
            print(f"  {name}: {values[0]} .. {values[-1]}")
        else:
            print(f"  {name}: <empty>")

    print("variables:")
    for name, var in ds.data_vars.items():
        dims = ",".join(var.dims)
        print(f"  {name}: dims={dims} dtype={var.dtype}")
    return 0


def _subset_region(data_array, args):
    if args.lat_min is not None or args.lat_max is not None:
        lat_min = -90.0 if args.lat_min is None else args.lat_min
        lat_max = 90.0 if args.lat_max is None else args.lat_max
        data_array = data_array.sel(lat=slice(lat_min, lat_max))
    if args.lon_min is not None or args.lon_max is not None:
        lon_min = 0.0 if args.lon_min is None else args.lon_min
        lon_max = 360.0 if args.lon_max is None else args.lon_max
        data_array = data_array.sel(lon=slice(lon_min, lon_max))
    return data_array


def plot_main(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt
    import xarray

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    ds = xarray.open_dataset(dataset_path, decode_timedelta=True)

    if args.variable not in ds:
        raise SystemExit(f"Variable not found: {args.variable}")

    data = ds[args.variable]
    if "time" in data.dims:
        data = data.isel(time=args.time_index)
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if "level" in data.dims:
        if args.level is None:
            raise SystemExit("This variable has a `level` dimension. Pass `--level`.")
        data = data.sel(level=args.level)

    data = _subset_region(data, args)

    plt.figure(figsize=(10, 5))
    data.plot()
    plt.title(args.variable)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
