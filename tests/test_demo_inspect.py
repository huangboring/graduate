import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from observer_selection.demo_inspect import TensorTrace


class TraceTests(unittest.TestCase):
    def test_inputs_are_snapshots_and_first_call_is_retained(self):
        trace=TensorTrace();x=torch.tensor([[1.,2.]])
        trace.capture('module.input',({'x':x,'mask':torch.tensor([True,False])},))
        x.add_(10);trace.capture('module.input',({'x':x},))
        np.testing.assert_array_equal(trace.arrays['module.input.0.x'],[[1,2]])
        with tempfile.TemporaryDirectory() as tmp:
            report=trace.save(Path(tmp))
            self.assertEqual(report['module.input.0.x']['shape'],[1,2])
            with np.load(Path(tmp)/'module_arrays.npz') as z:
                self.assertEqual(z['module.input.0.mask'].dtype,np.bool_)
