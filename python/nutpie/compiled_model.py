from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CompiledModel:
    dims: dict[str, tuple[str, ...]] | None
    reparameterized_names: list[str] | None = field(default=None, kw_only=True)

    @property
    def n_dim(self) -> int:
        raise NotImplementedError()

    @property
    def shapes(self) -> dict[str, tuple[int, ...]] | None:
        raise NotImplementedError()

    @property
    def coords(self):
        raise NotImplementedError()

    def _make_sampler(self, *args, **kwargs):
        raise NotImplementedError()

    def _make_model(self, *args, **kwargs):
        raise NotImplementedError()

    def benchmark_logp(self, point, num_evals, cores):
        """Time how long the logp gradient evaluation takes.

        # Parameters
        """
        model = self._make_model(point)
        times = []
        if isinstance(cores, int):
            cores = [cores]
        for num_cores in cores:
            if num_cores == 0:
                continue
            flat = model.benchmark_logp(point, num_cores, num_evals)
            data = pd.DataFrame(flat)
            data.index = pd.MultiIndex.from_product(
                [range(num_cores), [num_cores]], names=["thread", "concurrent_cores"]
            )
            data = data.rename_axis(columns="evaluation")
            times.append(data)
        return pd.concat(times)
