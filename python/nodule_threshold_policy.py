"""One fixed low-false-positive operating point; never fit on the current CT."""
import math

BASE_THRESHOLD=0.6418673229668569  # Historical reference only; not selectable.
LOW_FP_THRESHOLD=0.8738163050576156
DEFAULT_THRESHOLD=LOW_FP_THRESHOLD
PRESETS=(DEFAULT_THRESHOLD,)


def checked_threshold(value):
    if type(value)not in (int,float)or not math.isfinite(value)or value!=DEFAULT_THRESHOLD:
        raise ValueError('肺结节检测已固定为低误报档 0.8738，请刷新网页')
    return DEFAULT_THRESHOLD


def select_candidates(rows,threshold=DEFAULT_THRESHOLD):
    threshold=checked_threshold(threshold)
    selected=[c for c in rows if c['score']>=threshold]
    selected.sort(key=lambda c:(-c['score'],*c['center_index_zyx']))
    return selected
