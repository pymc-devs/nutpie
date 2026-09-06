"""Run directly in an environment with PyMC/Numba; no nutpie import needed."""

import ctypes
import unittest

import numpy as np
import pymc as pm
from compile_model import compile_browser_model


class CompilerTests(unittest.TestCase):
    def test_transformed_model_and_frozen_data(self):
        with pm.Model() as model:
            observed = pm.Data("observed", np.array([0.3, 0.8, -0.1]))
            loc = pm.Normal("loc", initval=0.2)
            scale = pm.HalfNormal("scale", initval=1.1)
            pm.Normal("y", loc, scale, observed=observed)
        compiled = compile_browser_model(model)
        logp = model.compile_logp(mode="NUMBA")
        grad = model.compile_dlogp(mode="NUMBA")
        point = model.initial_point()
        pointer = ctypes.POINTER(ctypes.c_double)
        for shift in [0, -0.03, 0.05]:
            x = compiled.initial + shift
            value = compiled.callback.ctypes(
                x.ctypes.data_as(pointer), compiled.gradient.ctypes.data_as(pointer)
            )
            shifted = {k: v + shift for k, v in point.items()}
            np.testing.assert_allclose(value, logp(shifted), atol=1e-9)
            np.testing.assert_allclose(compiled.gradient, grad(shifted), atol=1e-9)
        observed.set_value(np.array([10.0, 20.0, 30.0]))
        frozen = compiled.callback.ctypes(
            x.ctypes.data_as(pointer), compiled.gradient.ctypes.data_as(pointer)
        )
        self.assertEqual(frozen, value)
        self.assertNotAlmostEqual(frozen, logp(shifted))

    def test_discrete_rejected(self):
        with pm.Model() as model:
            pm.Bernoulli("x", 0.5)
        with self.assertRaisesRegex(ValueError, "continuous"):
            compile_browser_model(model)


if __name__ == "__main__":
    unittest.main()
