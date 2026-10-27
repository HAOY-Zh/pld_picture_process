"""Selectable development operating points, never fit on the current CT."""
import math

BASE_THRESHOLD=0.6418673229668569
DEFAULT_THRESHOLD=0.7655561641768475
LOW_FP_THRESHOLD=0.8738163050576156
PRESETS=(BASE_THRESHOLD,DEFAULT_THRESHOLD,LOW_FP_THRESHOLD)


def checked_threshold(value):
    if type(value)not in (int,float)or not math.isfinite(value)or value not in PRESETS:
        raise ValueError('请选择已记录的候选筛选档位')
    return float(value)


def select_candidates(rows,threshold):
    threshold=checked_threshold(threshold)
    # No score modification, Top-K cap, geometry change or ID renumbering.
    selected=[c for c in rows if c['score']>=threshold]
    selected.sort(key=lambda c:(-c['score'],*c['center_index_zyx']))
    return selected
