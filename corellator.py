#!/usr/bin/env python
###########################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################

import sqlite3
import json
import sys
import itertools
import argparse
from dataclasses import dataclass

@dataclass
class Interval:
    kernel: str
    start: int
    end: int
    
    def __gt__(self, other):
        return self.start > other.start

@dataclass
class Node:
    earlier: 'Node'
    later: 'Node'
    interval: Interval
    def __init__(self, interval):
        self.earlier = None
        self.later = None
        self.interval = interval

    def __len__(self):
        return ((0 if self.earlier is None else len(self.earlier)) + 
                (0 if self.later is None else len(self.later)) + 
                (0 if self.interval is None else 1))
    
def insert_interval(interval: Interval, btree: Node, start=True):
    if btree.interval is None:
        btree.interval = interval
    else:
        if start:
            ts1 = interval.start
            ts2 = btree.interval.start
        else:
            ts1 = interval.end
            ts2 = btree.interval.end

        if ts1 > ts2:
            if btree.later is None:
                btree.later = Node(interval)
            else:
                insert_interval(interval, btree.later, start)
        else:
            if btree.earlier is None:
                btree.earlier = Node(interval)
            else:
                insert_interval(interval, btree.earlier, start)

def insert_intervals(intervals, btree, start=True):
    if len(intervals) == 0:
        return
    elif len(intervals) == 1:
        insert_interval(intervals[0], btree, start)
        return
    
    middle = len(intervals) // 2
    insert_interval(intervals[middle], btree, start)
    insert_intervals(intervals[:middle], btree, start)
    insert_intervals(intervals[middle + 1:], btree, start)

def find_interval(ts: int, btree: Node):
    if btree is None:
        return None
    if ts > btree.interval.start:
        if ts < btree.interval.end:
            return btree.interval
        else:
            return find_interval(ts, btree.later)
    else:
        return find_interval(ts, btree.earlier)

def load_json(path, plot, stretch_plot, skew):
    print("Loading file", path)
    with open(path, 'r') as fp:
        data = json.load(fp)

    reg_values = {}
    stretch_values = {}
    interval_list = []

    print('Finding data points')

    for row in data['traceEvents']:
        if 'args' in row:
            if plot in row['args']:
                v = row['args'][plot]
                reg_values[row['ts'] * 1000 + skew] = v
            elif stretch_plot in row['args']:
                stretch_values[row['ts'] * 1000 + skew] = row['args'][stretch_plot]
            elif 'desc' in row['args'] and row['args']['desc'] == "KernelExecution":
                name = row['name']
                start = int(row['ts']) * 1000
                dur = int(row['dur']) * 1000
                interval_list.append(Interval(name, start, start + dur))
    
    return reg_values, interval_list, stretch_values


def load_rpd(path, plot, stretch_plot, freq, skew):
    print("Loading file", path)
    con = sqlite3.connect(path)
    cur = con.cursor()

    reg_values = {}
    stretch_values = {}
    freq_values = {}
    interval_list = []

    print('Finding data points')
    kerns = cur.execute("SELECT start, end, kernelName FROM kernel").fetchall()
    for start, end, name in kerns:
        interval_list.append(Interval(name, start, end))

    plots = cur.execute(f"SELECT start, value FROM rocpd_monitor WHERE monitorType = '{plot}'").fetchall()
    for time, value in plots:
        value = int(value)
        reg_values[time + skew] = value
    
    plots = cur.execute(f"SELECT start, value FROM rocpd_monitor WHERE monitorType = '{stretch_plot}'").fetchall()
    for time, value in plots:
        value = float(value)
        stretch_values[time + skew] = value

    plots = cur.execute(f"SELECT start, value FROM rocpd_monitor WHERE monitorType = '{freq}'").fetchall()
    for time, value in plots:
        value = float(value)
        freq_values[time + skew] = value
    print("freq", len(plots))

    return reg_values, interval_list, stretch_values, freq_values

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--plots', default='/etc/corellator.json', help="The plot name that counts events")
    parser.add_argument('input', help="A perfetto trace file")
    parser.add_argument('--skew', default=0, type=int, help="A time offset applied to events (ns)")
    parser.add_argument('-n', default=10, type=int, help="Show the top N results")

def main(argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(argv[1:])

    with open(args.plots, 'r') as fp:
        plots = json.load(fp)
    event_plot = plots['counter']
    stretch_plot = plots['stretch']

    if args.input.endswith('json'):
        loader = load_json
    elif args.input.endswith('rpd'):
        loader = load_rpd
    else:
        print("Unknown input format")
        return 1
    reg_values, interval_list, stretch_values, freq_values = loader(args.input, event_plot, stretch_plot, plots['freq'], args.skew)
    
    reg_total = sum(reg_values.values())

    kernel_data = {}
    kernel_time = {}
    kernel_stretch = {}
    freq_data = {}

    for interval in interval_list:
        name = interval.kernel
        if not name in kernel_data:
            kernel_data[name] = []
            freq_data[name] = []
        kernel_time[name] = kernel_time.get(name, 0.0) + (interval.end - interval.start)

    print('Num data points', event_plot, len(reg_values))
    print('Num data points', stretch_plot, len(stretch_values))
    print('Num kernels:', len(kernel_data))
    print('Num intervals:', len(interval_list))
    print('Total', event_plot, reg_total)

    if len(reg_values) == 0 or len(kernel_data) == 0:
        print('There is a problem with the trace, it is missing critical data.')
        return 1

    interval_btree = Node(None)
    insert_intervals(interval_list, interval_btree, True)

    print('Btree assembled')

    no_kern = 0
    no_kern_sum = 0

    for ts in reg_values:
        interval = find_interval(ts, interval_btree)
        if interval is None:
            no_kern += 1
            no_kern_sum += reg_values[ts]
        else:
            kernel_data[interval.kernel].append(reg_values[ts])

    stretch_times = sorted(stretch_values.keys())
    for i in range(1, len(stretch_times)):
        growth = stretch_values[stretch_times[i]] - stretch_values[stretch_times[i-1]]
        if growth > 0:
            interval = find_interval(stretch_times[i], interval_btree)
            if interval is not None:
                kernel_stretch[interval.kernel] = kernel_stretch.get(interval.kernel, 0.0) + growth

    print('Data points without kernel:', no_kern, "({} {:.3f}%)".format(no_kern_sum, no_kern_sum / reg_total * 100))
    print('Data points with kernel:', len(reg_values) - no_kern, "({} {:.3f}%)".format(reg_total - no_kern_sum, (reg_total - no_kern_sum) / reg_total * 100))

    kernel_sums = {}
    for kern in kernel_data:
        kernel_sums[kern] = sum(kernel_data[kern])

    sorted_sums = sorted(kernel_sums.items(), key=lambda x: x[1], reverse=True)

    top_n = min(args.n, len(sorted_sums))
    print('Top', top_n, 'offenders (total events)')
    for kernel, ksum in sorted_sums[:top_n]:
        print(kernel, ksum, '{:.3f}%'.format(ksum/reg_total*100.0))

    sorted_rate = sorted([(x[0], x[1] / kernel_time[x[0]] if kernel_time[x[0]] > 0 else 0.0)
                         for x in kernel_sums.items()], key=lambda x: x[1], reverse=True)

    print('Top', top_n, 'offenders (events/us)')
    for kernel, rate in sorted_rate[:top_n]:
        print(kernel, '{:.3f}'.format(rate * 1000.0))

    sorted_growth = sorted(kernel_stretch.items(), key=lambda x: x[1], reverse=True)
    print('Top', top_n, 'stretch growth')
    for kernel, growth in sorted_growth[:top_n]:
        print(kernel, '{:.3f}'.format(growth))

    for ts in freq_values:
        interval = find_interval(ts, interval_btree)
        if interval is None:
            pass
        else:
            freq_data[interval.kernel].append(freq_values[ts])

    freq_avg = {}
    for kern in freq_data:
        data = freq_data[kern]
        if len(data) > 0:
            avg = sum(data)/len(data)
            freq_avg[kern] = avg


    print('Bottom', top_n, 'average frequency')
    sorted_freq = sorted(freq_avg.items(), key=lambda x: x[1])
    for kernel, freq in sorted_freq[:top_n]:
        print(kernel, '{:.0f}'.format(freq/1000.0))

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
