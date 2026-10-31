"""Explicit static per-lane result writes; no cross-process variable-index RAM."""
import argparse
import json
from pathlib import Path
from nodule_overlap_feed_build import make
from nodule_overlap_build import simulate,replace
from nodule_plan import sha


def prepare(root):
    make(root);root=Path(root).resolve();f=root/'rtl/lung_mac_tile8.v';t=f.read_text()
    t=replace(t,'''                    result_acc[{r3_slot,3'b000}+o]<=final_sum[31:0];
                    result_error[{r3_slot,3'b000}+o]<=result_group_error[r3_slot] ||
                        (|result_partial_errors[r3_slot][8*o+:8]) || (final_sum[35:31]!={5{final_sum[31]}});''', '''                    // Constant addresses per generated lane avoid an inferred
                    // multi-writer memory. Exactly the same two-slot arithmetic.
                    if(r3_slot)begin
                        result_acc[8+o]<=final_sum[31:0];
                        result_error[8+o]<=result_group_error[1] ||
                            (|result_partial_errors[1][8*o+:8]) || (final_sum[35:31]!={5{final_sum[31]}});
                    end else begin
                        result_acc[o]<=final_sum[31:0];
                        result_error[o]<=result_group_error[0] ||
                            (|result_partial_errors[0][8*o+:8]) || (final_sum[35:31]!={5{final_sum[31]}});
                    end''')
    f.write_text(t)
    p=root/'preparation.json';d=json.loads(p.read_text());d['status']='prepared_static_lane_pixel_overlap_and_hit_feed'
    d['protected_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    d['generated_sha256']={str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='preparation.json'}
    p.write_text(json.dumps(d,indent=2));print(d['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--simulate',type=int,choices=[128,448]);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args()
    if a.simulate:simulate(a.root,a.simulate,a.vivado_bin)
    else:prepare(a.root)
