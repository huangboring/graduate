from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from observer_selection.official_demo import strict_weights
from observer_selection.official_demo_visuals import render_demo


class OfficialDemoTests(unittest.TestCase):
    def test_checkpoint_requires_all_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pth';net=torch.nn.Linear(2,3)
            torch.save({'params':{'module.'+k:v for k,v in net.state_dict().items()}},path)
            self.assertEqual(strict_weights(net,path),2)
            torch.save({'params':{'weight':net.weight}},path)
            with self.assertRaises(RuntimeError):strict_weights(net,path)

    def test_visual_export_fixture(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);source=out/'observer.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),20,(32,32))
            self.assertTrue(writer.isOpened())
            for _ in range(4):writer.write(np.zeros((32,32,3),np.uint8))
            writer.release()
            reference=np.zeros((3,22,3));reference[:,:,2]=np.linspace(-1,1,22)
            prediction=reference.copy();prediction[:,:,0]+=.01
            render_demo(out,prediction,reference,reference,source,'EXPLICIT GENERATED TEST FIXTURE')
            values=np.loadtxt(out/'per_frame_errors.csv',delimiter=',',skiprows=1)
            np.testing.assert_allclose(values[:,2],10)
            np.testing.assert_array_equal(values[:,0],[2,3,4])
            for name in ('demo.mp4','viewer.html','pose_comparison.png','error_curve.png','error_curve.pdf'):
                self.assertGreater((out/name).stat().st_size,100)
            self.assertIn('not exact paper reproduction',(out/'viewer.html').read_text())
