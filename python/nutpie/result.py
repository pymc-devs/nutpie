"""Shared ArviZ/xarray conversion, independent of the sampler's storage format."""

from importlib.metadata import version

import arviz
import xarray as xr


def _dict_to_arviz(
    data_posterior,
    stats_posterior,
    data_tune,
    stats_tune,
    dims,
    reparameterized_names=(),
    keep_unconstrained_draw=False,
    **kwargs,
):
    uc_data_posterior = {
        name: data_posterior.pop(name)
        for name in reparameterized_names
        if name in data_posterior
    }
    uc_data_tune = {
        name: data_tune.pop(name) for name in reparameterized_names if name in data_tune
    }

    arviz_version = version("arviz")
    use_datatree = tuple(map(int, arviz_version.split(".")[:2])) >= (1, 0)
    if use_datatree:
        idata = arviz.from_dict(
            {
                "posterior": data_posterior,
                "sample_stats": stats_posterior,
                "warmup_posterior": data_tune,
                "warmup_sample_stats": stats_tune,
            },
            dims=dims,
            **kwargs,
        )
    else:
        idata = arviz.from_dict(
            posterior=data_posterior,
            sample_stats=stats_posterior,
            warmup_posterior=data_tune,
            warmup_sample_stats=stats_tune,  # ty:ignore[invalid-argument-type]
            dims=dims,
            **kwargs,
        )

    if keep_unconstrained_draw and uc_data_posterior:
        coords = kwargs.get("coords")
        uc_dims = {name: dims.get(name, []) for name in uc_data_posterior}
        groups = {
            "unconstrained_posterior": arviz.dict_to_dataset(
                uc_data_posterior, coords=coords, dims=uc_dims
            )
        }
        if uc_data_tune:
            groups["warmup_unconstrained_posterior"] = arviz.dict_to_dataset(
                uc_data_tune, coords=coords, dims=uc_dims
            )
        if use_datatree:
            idata = idata.assign(**{k: xr.DataTree(v) for k, v in groups.items()})
        else:
            idata.add_groups(groups)

    return idata
