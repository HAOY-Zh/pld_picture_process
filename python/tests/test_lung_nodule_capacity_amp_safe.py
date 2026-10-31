import unittest
import torch
from lung_nodule_capacity_amp_safe import bounded_scaler_type,SCALE


class ScaleTests(unittest.TestCase):
    def test_resume_lowers_scale_and_disables_growth(self):
        old=torch.amp.GradScaler('cpu',init_scale=1024.)
        cls=bounded_scaler_type(torch)
        scaler=cls('cpu',init_scale=1024.)
        scaler.load_state_dict(old.state_dict())
        self.assertEqual(scaler.get_scale(),SCALE)
        self.assertEqual(scaler.get_growth_interval(),2**31-1)
        self.assertEqual(scaler.state_dict()['_growth_tracker'],0)
        # A finite optimizer step stays at32 instead of reintroducing overflow.
        p=torch.nn.Parameter(torch.tensor(2.))
        optimizer=torch.optim.SGD([p],lr=.1)
        scaler.scale(p.square()).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertAlmostEqual(float(p.detach()),1.6,places=6)
        self.assertEqual(scaler.get_scale(),SCALE)


if __name__=='__main__':
    unittest.main()
