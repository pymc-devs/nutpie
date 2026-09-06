"""Freeze a PyMC model and expose its Numba logp/gradient via a C callback.

Import this example directly; importing nutpie still requires its native extension.
Set PYTENSOR_FLAGS=cxx=,blas__ldflags=,numba__cache=False before importing PyTensor
in the browser. Keep the returned object alive for the lifetime of sampling.
"""

from dataclasses import dataclass

import numba
import numpy as np
import pytensor
import pytensor.tensor as pt
from pymc.pytensorf import join_nonshared_inputs
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import graph_inputs


@dataclass
class BrowserModel:
    callback: object
    function: object
    initial: np.ndarray
    gradient: np.ndarray
    layout: list

    def config(self):
        """Pointers refer to this Python runtime's WASM memory/function table."""
        return {
            "callback_pointer": int(self.callback.address),
            "x_pointer": int(self.initial.ctypes.data),
            "g_pointer": int(self.gradient.ctypes.data),
            "initial": self.initial.tolist(),
            "layout": self.layout,
        }


def compile_browser_model(model):
    """Compile continuous value variables; snapshot all current shared data.

    Returns unconstrained coordinates in model.value_vars order. Data changes
    require recompilation. Discrete variables, JAX, and Python fallback Ops
    are not supported. This experimental example does not expand deterministics
    or convert samples back to constrained variables/InferenceData.
    """
    if model.discrete_value_vars:
        raise ValueError("The browser adapter requires continuous value variables")
    point = model.initial_point()
    layout = [
        {
            "name": v.name,
            "shape": list(point[v.name].shape),
            "size": int(point[v.name].size),
        }
        for v in model.value_vars
    ]
    if not layout:
        raise ValueError("The model needs at least one free variable")
    initial = np.concatenate([point[v.name].ravel() for v in model.value_vars])
    initial = np.ascontiguousarray(initial, dtype=np.float64)
    [logp], q = join_nonshared_inputs(point, [model.logp()], model.value_vars)
    outputs = [logp, pt.grad(logp, q)]
    constants = {
        v: pt.constant(v.get_value())
        for v in graph_inputs(outputs)
        if isinstance(v, SharedVariable)
    }
    outputs = clone_replace(outputs, replace=constants)
    function = pytensor.function([q], outputs, mode="NUMBA")
    function(initial)
    inner = function.vm.jit_fn
    n = len(initial)
    pointer = numba.types.CPointer(numba.types.float64)

    @numba.cfunc(numba.types.float64(pointer, pointer), cache=False)
    def callback(xp, gp):
        x = numba.carray(xp, (n,))
        g = numba.carray(gp, (n,))
        lp, gradient = inner(x)
        for j in range(n):
            g[j] = gradient[j]
        return lp.item()

    return BrowserModel(callback, function, initial, np.zeros(n), layout)
