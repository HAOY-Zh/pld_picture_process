"""Run only on PYNQ with the matching new Overlay; not executed in delivery."""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from ct_registration import warp_fixed,quantize_matrix,rigid_matrix
from registration_board import RegistrationBoard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bit',required=True)
    parser.add_argument('--size',type=int,default=128)
    args=parser.parse_args()
    shape=(args.size,args.size)
    rng=np.random.default_rng(20261015)
    fixed=rng.integers(0,256,shape,dtype=np.uint8)
    moving=rng.integers(0,256,shape,dtype=np.uint8)
    cases=[('identity',fixed,moving,rigid_matrix(shape)),
           ('fractional',fixed,moving,rigid_matrix(shape,0,.5,.25)),
           ('negative',fixed,moving,rigid_matrix(shape,0,-2,-1)),
           ('rotation',fixed,moving,rigid_matrix(shape,3,1,-1)),
           ('no_overlap',fixed,moving,rigid_matrix(shape,0,args.size+1,0)),
           ('black',np.zeros(shape,np.uint8),np.zeros(shape,np.uint8),rigid_matrix(shape)),
           ('white',np.full(shape,255,np.uint8),np.full(shape,255,np.uint8),rigid_matrix(shape))]
    with RegistrationBoard(args.bit,shape) as board:
        for name,reference,source,matrix in cases:
            expected,valid=warp_fixed(source,quantize_matrix(matrix))
            result=board.evaluate(reference,source,matrix)
            np.testing.assert_array_equal(result['warped'],expected,err_msg=name)
            np.testing.assert_array_equal(result['valid'],valid,err_msg=name)
            diff=reference.astype(np.int64)-expected.astype(np.int64)
            if result['ssd']!=int((diff[valid]**2).sum()):
                raise AssertionError(f'{name}: SSD mismatch')
            print(f'{name}: PASS; valid={result["count"]}; SSD={result["ssd"]}')
    print('All listed board cases passed; not a clinical or real-pair validation.')


if __name__=='__main__':
    main()
