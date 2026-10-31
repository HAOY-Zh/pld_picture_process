"""Equivalent padded-window edge flags remove chained coordinate arithmetic."""
import argparse
import json
from pathlib import Path
from nodule_overlap_synthsafe_build import prepare as base_prepare
from nodule_overlap_build import simulate,replace
from nodule_plan import sha


def prepare(root):
    base_prepare(root);root=Path(root).resolve();f=root/'rtl/lung_feature_window.v';t=f.read_text()
    t=replace(t,'    reg[8:0]x,y;reg[1:0]kx,ky;reg[13:0]row_stride;', '''    reg[8:0]x,y;reg[1:0]kx,ky;reg[13:0]row_stride;
    reg [8:0] last_x,last_y;
    wire x_last=x==last_x,y_last=y==last_y;
    wire left_edge=x==0,right_edge=x_last,top_edge=y==0,bottom_edge=y_last;''')
    t=replace(t,'''    wire signed[10:0]sy=$signed({2'b0,y})+$signed({9'd0,ky})-$signed({10'd0,padding});
    wire signed[10:0]sx=$signed({2'b0,x})+$signed({9'd0,kx})-$signed({10'd0,padding});
    wire outside=sy<0||sx<0||sy>=$signed({1'b0,height})||sx>=$signed({1'b0,width});''', '''    // Valid unpadded windows never leave the frame. For padded 3x3, only
    // first/last rows or columns can be outside; no add/subtract/compare chain.
    wire outside=kernel3&&padding&&((left_edge&&kx==0)||(right_edge&&kx==2)||
        (top_edge&&ky==0)||(bottom_edge&&ky==2));''')
    t=replace(t,'wire frame_last=x==out_width-1\'b1&&y==out_height-1\'b1;',
        'wire frame_last=x_last&&y_last;')
    first=t.index('    wire [3:0] next_cb=')
    last=t.index('    wire consume_window=',first)
    t=t[:first]+'''    wire [33:0] next_address=(next_channel||next_column)?address+1'b1:
        next_row?kernel_row_address+row_stride:
        !x_last ? pixel_address+blocks : next_row_address;
    // Derive the next tap's padding directly from the stepping case. At the
    // next pixel kx=ky=0; its x is zero on row wrap and y is zero on row zero.
    wire next_outside=kernel3&&padding&&(
        next_channel?outside:
        next_column?((right_edge&&kx==1)||(top_edge&&ky==0)||(bottom_edge&&ky==2)):
        next_row?(left_edge||(bottom_edge&&ky==1)):
        (x_last||top_edge));
'''+t[last:]
    t=replace(t,"else if(x!=out_width-1'b1)begin",'else if(!x_last)begin')
    t=replace(t,'            x<=0;y<=0;kx<=0;ky<=0;row_stride<=0;',
        '            x<=0;y<=0;kx<=0;ky<=0;row_stride<=0;last_x<=0;last_y<=0;')
    t=replace(t,'                    out_width<=c_kernel3&&!c_padding?c_width-2:c_width;', '''                    out_width<=c_kernel3&&!c_padding?c_width-2:c_width;
                    last_x<=c_kernel3&&!c_padding?c_width-3:c_width-1;
                    last_y<=c_kernel3&&!c_padding?c_height-3:c_height-1;''')
    f.write_text(t);p=root/'preparation.json';d=json.loads(p.read_text());d['status']='prepared_pixel_overlap_hit_feed_registered_boundaries'
    d['protected_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    d['generated_sha256']={str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='preparation.json'}
    p.write_text(json.dumps(d,indent=2));print(d['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--simulate',type=int,choices=[128,448]);p.add_argument('--vivado-bin',default='E:/vivado2/Vivado/2022.2/bin');a=p.parse_args()
    if a.simulate:simulate(a.root,a.simulate,a.vivado_bin)
    else:prepare(a.root)
