import datetime
import hashlib
import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nutpie import _lib
from nutpie.sample import CompiledModel, _wrap_init_point_fn

logger = logging.getLogger("nutpie")


@dataclass(frozen=True)
class CompiledStanModel(CompiledModel):
    _coords: dict[str, Any] | None
    code: str
    data: dict[str, NDArray] | None
    library: Any
    model: Any
    model_name: str | None = None
    _transform_adapt_args: dict | None = None
    # User init function `fn(model, rng, chain_id)`, see `with_init_point_fn`
    _init_point_fn: Callable | None = None

    def with_data(self, *, seed=None, **updates):
        if self.data is None:
            data = {}
        else:
            data = self.data.copy()

        data.update(updates)

        data_json = _dump_stan_json(data)

        outer_kwargs = self._transform_adapt_args
        if outer_kwargs is None:
            outer_kwargs = {}

        def make_adapter(*args, **kwargs):
            from nutpie.transform_adapter import make_transform_adapter

            return make_transform_adapter(**outer_kwargs)(*args, **kwargs, logp_fn=None)

        coords = self._coords
        if coords is None:
            coords = {}
        coords = coords.copy()

        dims = self.dims
        if dims is None:
            dims = {}
        dims = dims.copy()
        dim_sizes = {name: len(dim) for name, dim in coords.items()}

        model = _lib.StanModel(
            self.library, dim_sizes, dims, coords, seed, data_json, make_adapter
        )

        return replace(self, _coords=coords, data=data, model=model)

    def with_coords(self, **coords):
        if self.coords is None:
            coords_new = {}
        else:
            coords_new = self.coords.copy()
        coords_new.update(coords)
        return replace(self, _coords=coords_new)

    def with_dims(self, **dims):
        if self.dims is None:
            dims_new = {}
        else:
            dims_new = self.dims.copy()
        dims_new.update(dims)
        return replace(self, dims=dims_new)

    def with_transform_adapt(self, **kwargs):
        return replace(self, _transform_adapt_args=kwargs).with_data()

    def with_init_point_fn(self, init_point_fn):
        """Use a custom function to generate the initial point of each chain.

        Parameters
        ----------
        init_point_fn : Callable[[CompiledStanModel, np.random.Generator, int], np.ndarray | dict]
            Called as ``init_point_fn(model, rng, chain_id)``. Must return
            either a flat point on the unconstrained space with shape
            ``(n_dim,)``, or a dict with constrained values for some or all
            parameters. Parameters missing from the dict are drawn uniformly
            from ``[-2, 2]`` on the unconstrained space.
        """
        return replace(self, _init_point_fn=init_point_fn)

    def _init_values_to_json(self, values, seed, chain_id):
        model = self._make_model()
        n_params = model.param_num()
        shapes = {
            name: tuple(var.shape)
            for name, var in model.variables().items()
            if var.end_idx <= n_params
        }
        unknown = [name for name in values if name not in shapes]
        if unknown:
            raise KeyError(
                f"Unknown parameters in initial point: {unknown}. "
                f"Expected a subset of {list(shapes)}."
            )
        for name, value in values.items():
            if np.shape(value) != shapes[name]:
                raise ValueError(
                    f"Initial value for {name} has shape {np.shape(value)}, "
                    f"expected {shapes[name]}"
                )
        return _dump_stan_json(values)

    def unconstrain(self, **values) -> NDArray:
        """Map constrained parameter values to the unconstrained space.

        Values for all parameters of the model must be given as keyword
        arguments, with the shapes as declared in the Stan program.
        """
        model = self._make_model()
        return model.param_unconstrain_json(_dump_stan_json(values))

    def constrain(
        self,
        point: NDArray,
        *,
        include_tp: bool = False,
        include_gq: bool = False,
        seed: int | None = None,
    ) -> NDArray:
        """Map a point on the unconstrained space to the constrained
        parameter values.

        Returns the flat vector of constrained values in the order used by
        Stan: variables in declaration order, each in column-major order.

        Parameters
        ----------
        point:
            Flat point on the unconstrained space with shape ``(n_dim,)``.
        include_tp:
            Also return the transformed parameters.
        include_gq:
            Also return the generated quantities. ``seed`` controls
            the random number generator used for those.
        """
        model = self._make_model()
        return model.param_constrain(
            np.ascontiguousarray(point, dtype=np.float64),
            include_tp,
            include_gq,
            seed,
        )

    def _make_model(self):
        if self.model is None:
            return self.with_data().model
        return self.model

    def _make_sampler(
        self,
        settings,
        cores,
        progress_type,
        extra_callback,
        extra_callback_rate,
        store,
    ):
        compiled = self if self.model is not None else self.with_data()
        model = compiled.model
        if compiled._init_point_fn is not None:
            model = model.with_init_point_func(
                _wrap_init_point_fn(
                    compiled._init_point_fn, compiled, compiled._init_values_to_json
                )
            )
        return _lib.PySampler.from_stan(
            settings,
            cores,
            model,
            progress_type,
            extra_callback,
            extra_callback_rate,
            store,
        )

    @property
    def n_dim(self):
        if self.model is None:
            return self.with_data().n_dim
        return self.model.ndim()

    @property
    def shapes(self):
        if self.model is None:
            return self.with_data().shapes
        return {name: var.shape for name, var in self.model.variables().items()}

    @property
    def coords(self):
        if self.model is None:
            return self.with_data().coords
        return self._coords


def _dump_stan_json(values: dict[str, Any]) -> str:
    if find_spec("stanio") is None:
        raise ImportError(
            "stanio is not installed in the current environment. "
            "Please install it with something like "
            "'pip install stanio' or 'pip install nutpie[stan]'."
        )

    import stanio

    return stanio.dump_stan_json(values)


def _stan_cache_key(
    code: str,
    extra_compile_args: list[str] | None,
    extra_stanc_args: list[str] | None,
) -> str:
    """Return a SHA-256 hex digest identifying a unique compilation job."""
    import bridgestan

    fingerprint = json.dumps(
        {
            "code": code,
            "extra_compile_args": sorted(extra_compile_args or []),
            "extra_stanc_args": sorted(extra_stanc_args or []),
            "bridgestan_version": bridgestan.__version__,
        },
        sort_keys=True,
    )
    return hashlib.sha256(fingerprint.encode()).hexdigest()


def _stan_cache_dir() -> Path:
    """Return (and create) the directory where compiled Stan models are cached."""
    import platformdirs

    cache_dir = Path(platformdirs.user_cache_dir("nutpie")) / "stan"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def prune_stan_cache(
    max_entries: int = 16,
    min_age: datetime.timedelta = datetime.timedelta(weeks=2),
) -> None:
    """Remove old entries from the Stan compilation cache.

    Entries are only considered for removal if they are older than *min_age*.
    Among those, the oldest ones are removed until at most *max_entries*
    entries remain.

    Parameters
    ----------
    max_entries:
        Maximum number of cache entries to keep.  Defaults to 16.
    min_age:
        Entries younger than this are never removed, regardless of how many
        entries exist.  Defaults to 2 weeks.
    """
    cache_dir = _stan_cache_dir()
    now = datetime.datetime.now(tz=datetime.UTC)

    # Collect all valid (marker exists) entries with their mtime.
    entries = []
    for entry_dir in cache_dir.iterdir():
        if not entry_dir.is_dir():
            continue
        marker = entry_dir / "ok"
        if not marker.exists():
            continue
        mtime = datetime.datetime.fromtimestamp(marker.stat().st_mtime, tz=datetime.UTC)
        entries.append((mtime, entry_dir))

    if len(entries) <= max_entries:
        return

    # Only entries older than min_age are candidates for eviction.
    candidates = sorted(
        [(mtime, d) for mtime, d in entries if (now - mtime) >= min_age]
    )

    n_to_remove = len(entries) - max_entries
    for _, entry_dir in candidates[:n_to_remove]:
        shutil.rmtree(entry_dir, ignore_errors=True)


def _compile_stan_model(
    model_name: str,
    code: str,
    build_dir: Path,
    make_args: list[str],
    stanc_args: list[str],
) -> Path:
    """Write *code* into *build_dir*, compile it, and return the path to the shared library."""
    import bridgestan

    model_path = (
        build_dir.joinpath("name")
        .with_name(model_name)  # This verifies that it is a valid filename
        .with_suffix(".stan")
    )
    model_path.write_text(code)
    so_path = bridgestan.compile_model(
        model_path, make_args=make_args, stanc_args=stanc_args
    )
    bridgestan.compile.windows_dll_path_setup()
    return so_path


def compile_stan_model(
    *,
    code: str | None = None,
    filename: str | None = None,
    extra_compile_args: list[str] | None = None,
    extra_stanc_args: list[str] | None = None,
    dims: dict[str, tuple[str, ...]] | None = None,
    coords: dict[str, Any] | None = None,
    model_name: str | None = None,
    cleanup: bool = True,
    cache: bool = False,
    prune_cache: bool = True,
) -> CompiledStanModel:
    """Compile a Stan model and return a :class:`CompiledStanModel`.

    Parameters
    ----------
    code:
        Stan model source code as a string.
    filename:
        Path to a ``.stan`` file.  Mutually exclusive with *code*.
    extra_compile_args:
        Extra arguments forwarded to the C++ compiler via BridgeStan's
        ``make_args``.
    extra_stanc_args:
        Extra arguments forwarded to the Stan compiler (``stanc``).
    dims:
        Variable dimension names, e.g. ``{"alpha": ["county"]}``.
    coords:
        Coordinate labels for each dimension, e.g.
        ``{"county": ["Hennepin", "Ramsey", ...]}``.
    model_name:
        Base name used for the ``.stan`` file.  Defaults to ``"model"``.
    cleanup:
        Remove the temporary build directory after compilation.  Has no
        effect when *cache* is ``True`` (the build directory is the cache
        entry and is never deleted).
    cache:
        When ``True``, compile the model into a persistent directory under
        the user cache directory (``~/.cache/nutpie/stan`` on Linux/macOS,
        ``%LOCALAPPDATA%\\nutpie\\stan`` on Windows) and reuse it on
        subsequent calls with identical arguments and the same BridgeStan
        version.  A marker file ``ok`` is written only after a successful
        build, so interrupted or failed compilations are never reused.
        Defaults to ``False``.
    prune_cache:
        When ``True`` (the default), call :func:`prune_stan_cache` after
        each new compilation to evict old cache entries.  Has no effect
        when *cache* is ``False``.
    """
    if find_spec("bridgestan") is None:
        raise ImportError(
            "BridgeStan is not installed in the current environment. "
            "Please install it with something like "
            "'pip install bridgestan' or 'pip install nutpie[stan]'."
        )

    import bridgestan

    if dims is None:
        dims = {}
    if coords is None:
        coords = {}

    if code is not None and filename is not None:
        raise ValueError("Specify exactly one of `code` and `filename`")
    if code is None:
        if filename is None:
            raise ValueError("Either code or filename have to be specified")
        with Path(filename).open() as file:
            code = file.read()

    if model_name is None:
        model_name = "model"

    make_args = ["STAN_THREADS=true"]
    if extra_compile_args:
        make_args.extend(extra_compile_args)
    stanc_args = []
    if extra_stanc_args:
        stanc_args.extend(extra_stanc_args)

    if cache:
        digest = _stan_cache_key(code, extra_compile_args, extra_stanc_args)
        entry_dir = _stan_cache_dir() / digest
        marker = entry_dir / "ok"

        so_path_file = entry_dir / "so_path.txt"

        if marker.exists():
            # Cache hit: touch the marker to record recent use, then load.
            marker.touch()
            so_path = Path(so_path_file.read_text())
            if not so_path.exists():
                raise FileNotFoundError(
                    f"Cached Stan library not found: {so_path}. "
                    "The cache entry may be corrupt; delete it and recompile."
                )
            bridgestan.compile.windows_dll_path_setup()
            library = _lib.StanLibrary(str(so_path))
        else:
            # Cache miss: compile directly into the cache entry directory so
            # that all relative loading paths inside the .so remain valid.
            entry_dir.mkdir(parents=True, exist_ok=True)
            so_path = _compile_stan_model(
                model_name, code, entry_dir, make_args, stanc_args
            )
            # Write the .so path before the marker so the marker is only
            # ever present once so_path.txt is fully written.
            so_path_file.write_text(str(so_path))
            marker.write_text("")
            library = _lib.StanLibrary(str(so_path))
            if prune_cache:
                prune_stan_cache()
    else:
        basedir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        try:
            so_path = _compile_stan_model(
                model_name, code, Path(basedir.name), make_args, stanc_args
            )
            library = _lib.StanLibrary(str(so_path))
        finally:
            try:
                if cleanup:
                    basedir.cleanup()
            except Exception as e:
                logger.warning("Could not remove stan model cache", exc_info=e)

    return CompiledStanModel(
        code=code,
        library=library,
        dims=dims,
        _coords=coords,
        model_name=model_name,
        model=None,
        data=None,
    )
