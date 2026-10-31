"""Same exact native CPU kernel/parameters, only spatial descriptors expanded."""
import ctypes
import numpy as np
from nodule_cpu import NativeNodule,encode_model
from nodule_canvas import expanded_plan


class CanvasCPU(NativeNodule):
    def __init__(self,library,model,parameters,height,width,threads=2):
        plan,shapes=expanded_plan(model,parameters,height,width)
        d,weights,bias,mult,shift=encode_model(model,parameters)
        for i,node in enumerate(plan['nodes']):
            d[i,4:6]=shapes[node['sources'][0]][0][1:]
            d[i,7:9]=node['shape'][1:]
        super().__init__(library,model,parameters,threads)
        self.close();self.model=plan
        self.handle=self.library.lung_cpu_create(d,len(d),*plan['input_shape'],weights,len(weights),bias,mult,shift,len(bias),threads)
        if not self.handle:raise RuntimeError(self.library.lung_cpu_error().decode())
        self.output=np.empty(plan['nodes'][-1]['shape'],np.uint8)
