import dataclasses
import io
from typing import Any

from nutpie import _lib
from nutpie.sampling import CompiledModel


def compile_pytensor_module(module, n_dim):
    import torch

    x = torch.zeros(n_dim)
    exported = torch.onnx.dynamo_export(module, x)

    exported_bytes = io.BytesIO()
    exported.save(exported_bytes)
    exported_bytes = exported_bytes.getvalue()

    compiled = CompiledOnnx(
        _n_dim=n_dim,
        providers=None,
        logp_module_bytes=exported_bytes,
        dims={"unconstrained_draw": ("unconstrained_parameter",)},
    )

    return compiled.with_providers(["cpu"])


@dataclasses.dataclass(frozen=True)
class CompiledOnnx(CompiledModel):
    logp_module_bytes: Any
    providers: Any
    _n_dim: int

    @property
    def shapes(self):
        return {"unconstrained_draw": (self.n_dim,)}

    @property
    def coords(self):
        return {}

    @property
    def n_dim(self):
        return self._n_dim

    def _make_model(self, init_mean):
        return _lib.OnnxModel(self.n_dim, self.logp_module_bytes, self.providers)

    def _make_sampler(self, settings, init_mean, cores, template, rate, callback=None):
        model = self._make_model(init_mean)
        return _lib.PySampler.from_onnx(
            settings, cores, model, template, rate, callback
        )

    def with_providers(self, provider_names):
        providers = _lib.OnnxProviders()
        for name in provider_names:
            if name == "cuda":
                providers.add_cuda()
            elif name == "tensorrt":
                providers.add_tensorrt()
            elif name == "cpu":
                providers.add_cpu()
            else:
                raise ValueError(f"Unknown provider {name}")
        return dataclasses.replace(self, providers=providers)
