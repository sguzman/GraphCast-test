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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m graphcast_lab.runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_forecast_parser(subparsers)
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
