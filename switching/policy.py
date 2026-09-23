"""Selective switching from layer- and head-averaged source attention."""

import numpy as np
from .schedule import Schedule


def decide(attention, schedule, thresholds=(0.50, 0.22)):
    attention = np.asarray(attention, dtype=np.float64)
    if attention.ndim != 2 or not np.isfinite(attention).all():
        raise ValueError('Expected a finite two-dimensional attention map')
    cv = float(attention.std() / (attention.mean() + 1e-8))
    pi = float(np.percentile(attention, 75))
    start, end = max(schedule.start, 10), schedule.end
    switch = cv <= thresholds[0] and pi <= thresholds[1] and start < end
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(attention, 1, mode='edge'), (3, 3))
    local_cv = windows.std((-2, -1)) / (windows.mean((-2, -1)) + 1e-8)
    info = dict(step=9, cv=cv, pi=pi, switched=switch,
                interval=(start, end) if switch else None,
                attention=attention, local_cv=local_cv, thresholds=thresholds)
    chosen = (Schedule('editing_t2i_editing', schedule.steps, start, end) if switch
              else Schedule('pure_editing', schedule.steps))
    return chosen, info
