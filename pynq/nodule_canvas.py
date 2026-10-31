"""Shape-only expansion of the frozen INT8 graph; exact valid-core extraction.

No parameter, requantizer, tile schedule, decoder or threshold is changed.
Only supported aligned rectangles are accepted. Unused tile borders are not
claimed equivalent: the decoder consumes the core plus one heatmap neighbour.
"""
import copy
import numpy as np
from nodule_plan import validate


def expanded_plan(model, parameters, height, width):
    validate(model, parameters)
    if any(type(v) is not int or v < 128 or v > 1024 or v % 8 for v in (height,width)):
        raise ValueError('Canvas dimensions must be multiples of 8 in 128..1024')
    result=copy.deepcopy(model);result['input_shape']=[45,height//2,width//2]
    shapes={'input':(result['input_shape'],result['input_zero'])}
    for node in result['nodes']:
        ca,h,w=shapes[node['sources'][0]][0]
        if node['kind']=='pool':h//=2;w//=2
        elif node['kind']=='fused_concat':h*=2;w*=2
        node['shape']=[node['shape'][0],h,w]
        if node['kind']=='conv':
            node['layer']['input_shape']=list(shapes[node['sources'][0]][0])
            node['layer']['output_shape']=list(node['shape'])
        shapes[node['name']]=(node['shape'],node['zero'])
    return result,shapes


def lower_canvas(codes):
    a=np.asarray(codes)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[0]!=5 or any(v%8 for v in a.shape[1:]):
        raise ValueError('Expected five-channel aligned UINT8 canvas')
    h,w=a.shape[1:];p=np.pad(a,((0,0),(1,1),(1,1)))
    return np.ascontiguousarray(np.stack([p[c,ky:ky+h:2,kx:kx+w:2]
        for c in range(5) for ky in range(3) for kx in range(3)]))


def extract_tile_heads(head,jobs,y0,x0,config):
    """Recover only the decoder-required 34x34 cells, retaining tile ordering.

    All other cells are constant zero logits. They cannot affect any retained
    local maximum: a 3x3 maximum at core [16:48] sees only [15:49].
    """
    a=np.asarray(head);p=config.patch//2;halo=config.halo//2
    if config.patch!=128 or config.halo!=32 or a.ndim!=3 or a.shape[0]!=5:
        raise ValueError('Only frozen 128/32 decoder contract supported')
    lo=halo-1;hi=p-halo+1;out=np.zeros((len(jobs),5,p,p),a.dtype)
    for i,j in enumerate(jobs):
        dy=int(j[1])-y0;dx=int(j[2])-x0
        if dy%8 or dx%8 or min(dy,dx)<0 or dy//2+p>a.shape[1] or dx//2+p>a.shape[2]:
            raise ValueError('Tile phase/outside canvas')
        out[i,:,lo:hi,lo:hi]=a[:,dy//2+lo:dy//2+hi,dx//2+lo:dx//2+hi]
    return out


def valid_dependency_proof(model):
    """Exact graph dependency sets, including padded intermediate constants.

    Track whether any node depends on an artificial tile boundary rather than
    the corresponding translated canvas computation. Every padded conv marks
    its edge invalid, including the first stem; pool and skip paths propagate
    invalidity. Proves all possible input/weight values, not just test images.
    """
    from scipy.ndimage import minimum_filter
    values={'input':np.ones((128,128),bool)};evidence={}
    for node in model['nodes']:
        a=values[node['sources'][0]]
        if node['kind']=='conv':
            if node['name']=='stem0':
                out=minimum_filter(a,size=3,mode='constant',cval=0)[::2,::2]
            elif node['layer']['kernel']==3:
                out=minimum_filter(a,size=3,mode='constant',cval=0)
            else:out=a.copy()
        elif node['kind']=='pool':
            h,w=a.shape;out=a.reshape(h//2,2,w//2,2).all((1,3))
        else:
            out=a.repeat(2,0).repeat(2,1)&values[node['sources'][1]]
        values[node['name']]=out
        ys,xs=np.where(out)
        evidence[node['name']]=dict(shape=list(out.shape),valid_cells=int(out.sum()),
            valid_bounds_yx=[int(ys.min()),int(ys.max()+1),int(xs.min()),int(xs.max()+1)] if len(ys) else None)
    valid=values['head'];required=valid[15:49,15:49]
    if required.shape!=(34,34) or not required.all():raise AssertionError('Halo cannot preserve decoder neighbourhood')
    return dict(status='all_inputs_valid_core_and_maximum_neighbours_proven',
        required_head_interval=[15,49],head_core_interval=[16,48],translation_input_multiple=8,nodes=evidence)
