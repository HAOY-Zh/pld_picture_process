"""RST1 exact scalar HU statistics, shared by PC validation and the board owner."""
from math import sqrt

STATISTICS_ABI='roi_hu_statistics_rst1_v1'
COMPUTED_ON='FPGA_STATISTICS_RST1'
IDENTITY=0x52535431
GPIO_NAMES=('roi_stats_sum','roi_stats_sumsq','roi_stats_meta','roi_stats_identity')


def statistics(count,total,squares,low,high):
    if any(type(v) is not int for v in (count,total,squares)) or not 0<=count<=512*512:
        raise ValueError('Invalid integer ROI statistics/count')
    if not -32768*count<=total<=32767*count or not 0<=squares<=(1<<30)*count:
        raise ValueError('HU sum/sumsq exceeds signed int16 bounds')
    numerator=count*squares-total*total
    if numerator<0:raise ValueError('Inconsistent variance numerator')
    if not count:
        if total or squares or low is not None or high is not None:raise ValueError('Nonzero empty ROI statistics')
    elif (type(low)is not int or type(high)is not int or not -32768<=low<=high<=32767
          or not count*low<=total<=count*high):raise ValueError('Inconsistent HU min/max')
    return dict(count=count,sum=total,sumsq=squares,min=low,max=high,mean=total/count if count else None,
        std_population=sqrt(numerator)/count if count else None,computed_on=COMPUTED_ON,
        dispersion_computed_on='CPU_FROM_FPGA_MOMENTS')


def validate_statistics(value,count):
    if not isinstance(value,dict) or value.get('computed_on')!=COMPUTED_ON:
        raise ValueError('Expected hardware RST1 statistics; no CPU fallback')
    expected=statistics(value.get('count'),value.get('sum'),value.get('sumsq'),value.get('min'),value.get('max'))
    if expected['count']!=count or any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('RST1 scalar statistics differ or are incomplete')
    return expected


def read_statistics(read_word,count,status):
    def word(name):
        value=read_word(name)
        if type(value)is not int or not 0<=value<2**32:raise ValueError('Invalid RST1 GPIO word: '+name)
        return value
    if word('identity')!=IDENTITY:raise ValueError('RST1 hardware identity mismatch')
    if status!=0x01000002 or word('valid')!=1:raise ValueError('ROI statistics not completed/valid')
    total=word('sum_lo')|(word('sum_hi')<<32)
    if total&(1<<63):total-=1<<64
    squares=word('sumsq_lo')|(word('sumsq_hi')<<32)
    extrema=word('minmax')
    signed16=lambda n:n-65536 if n&32768 else n
    low,high=signed16(extrema&65535),signed16(extrema>>16)
    if not count:
        if extrema:raise ValueError('Nonzero empty min/max')
        low=high=None
    result=statistics(count,total,squares,low,high)
    if word('valid')!=1:raise ValueError('Statistics changed during read')
    return result
