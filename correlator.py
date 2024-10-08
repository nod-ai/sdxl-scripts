#!/usr/bin/env python
###########################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################

import sqlite3
import json
import sys
import itertools
import argparse
from dataclasses import dataclass, field

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

def mean(items, _):
    return sum(items)/len(items)

@dataclass
class Metric:
    name: str
    trace_name: str
    top: bool
    dtype: type
    cumulative: bool = False
    values: dict = field(default_factory=dict)
    kern_values: dict = field(default_factory=dict)
    no_kern: int = 0
    total = 0
    no_kern_sum = 0

    def load_intervals(self, intervals):
        for ts in self.values:
            interval = find_interval(ts, intervals)
            if interval is None:
                self.no_kern += 1
                if self.cumulative:
                    self.no_kern_sum += self.values[ts]
            else:
                if not interval.kernel in self.kern_values:
                    self.kern_values[interval.kernel] = []
                self.kern_values[interval.kernel].append(self.values[ts])

    def summary(self):
        result = f'{self.name} ({self.trace_name}): {len(self.values)} Samples, '
        if self.cumulative:
            result += f' Total: {self.total} Samples w/o kernel: {self.no_kern}'
        else:
            result += f' Average: {self.total/len(self.values)}'
        return result

    def report(self, note, count, rank, percent=False):
        value_ranks = {}
        for kern in self.kern_values:
            value_ranks[kern] = rank(self.kern_values[kern], kern)

        sorted_rank = sorted(value_ranks.items(), key=lambda x: x[1], reverse=self.top)
        
        count = min(count, len(sorted_rank))

        print('Top' if self.top else 'Bottom', count, f'kernels ({self.name})', note)
        for kernel, rank in sorted_rank[:count]:
            if percent:
                print(kernel, rank, '{:.3f}%'.format(rank/self.total*100.0))
            else:
                print(kernel, rank)

        return sorted_rank

def load_rpd(path, metrics, skew):
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

    for metric in metrics:
        plots = cur.execute(f"SELECT start, value FROM rocpd_monitor WHERE monitorType = '{metric.trace_name}'").fetchall()
        for time, value in plots:
            value = metric.dtype(value)
            metric.values[time + skew] = value

    return interval_list

def prepare_metrics(metrics, intervals):
    for metric in metrics:
        metric.total = sum(metric.values.values())
        metric.load_intervals(intervals)

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--plots', default='/etc/correlator.json', help="The plot name that counts events")
    parser.add_argument('input', help="A perfetto trace file")
    parser.add_argument('--skew', default=0, type=int, help="A time offset applied to events (ns)")
    parser.add_argument('-n', default=10, type=int, help="Show the top N results")

def main(argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(argv[1:])

    with open(args.plots, 'r') as fp:
        plots = json.load(fp)

    pcc_events = Metric("PCC Events", plots['counter'], True, int, True)
    clock_stretch = Metric("Clock Stretch", plots['stretch'], True, float)
    gfx_frequency = Metric("GFX Frequency", plots['freq'], False, float)

    metrics = [pcc_events, clock_stretch, gfx_frequency]

    interval_list = load_rpd(args.input, metrics, args.skew)

    print('Assemble btree')
    interval_btree = Node(None)
    insert_intervals(interval_list, interval_btree, True)

    print('Prepare Metrics')
    prepare_metrics(metrics, interval_btree)

    kernel_time = {}

    for interval in interval_list:
        name = interval.kernel
        kernel_time[name] = kernel_time.get(name, 0.0) + (interval.end - interval.start)

    print('Num kernels:', len(kernel_time))
    print('Num intervals:', len(interval_list))

    for metric in metrics:
        print(metric.summary())

    if any([len(x.values) == 0 for x in metrics]):
        print('There is a problem with the trace, it is missing critical data.')
        return 1

    def event_rate(values, kern):
        return 1000.0 * sum(values) / kernel_time[kern] if kernel_time[kern] > 0 else 0.0

    pcc_events.report('Count', args.n, lambda x, _: sum(x), True)
    pcc_events.report('Rate (per us)', args.n, event_rate)
    clock_stretch.report('Average', args.n, mean)
    gfx_frequency.report('Average', args.n, mean)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
