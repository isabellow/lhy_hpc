import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def log_tick_formatter(val, pos=None):
    return f"$10^{{{val:g}}}$"