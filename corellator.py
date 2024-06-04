#!/usr/bin/env python
###########################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################

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

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('plot', help="The plot name that counts events")
    parser.add_argument('input', help="A perfetto trace file")
    parser.add_argument('--skew', default=0, type=int, help="A time offset applied to events")
    parser.add_argument('-n', default=10, type=int, help="Show the top N results")

def main(argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(argv[1:])

    print("Loading file", args.input)
    with open(args.input, 'r') as fp:
        data = json.load(fp)

    reg_values = {}
    reg_total = 0

    kernel_intervals = {}
    kernel_data = {}
    kernel_time = {}

    interval_btree = Node(None)
    interval_list = []

    print('Finding data points')

    for row in data['traceEvents']:
        if 'args' in row:
            if args.plot in row['args']:
                v = row['args'][args.plot]
                reg_values[row['ts'] + args.skew] = v
                reg_total += v
            elif 'desc' in row['args'] and row['args']['desc'] == "KernelExecution":
                name = row['name']
                start = int(row['ts'])
                dur = int(row['dur'])
                interval_list.append(Interval(name, start, start + dur))
                kernel_data[name] = []
                kernel_time[name] = kernel_time.get(name, 0.0) + dur

    print('Num data points:', len(reg_values))
    print('Num kernels:', len(kernel_data))
    print('Num intervals:', len(interval_list))
    print('Total', args.plot, reg_total)

    if len(reg_values) == 0 or len(kernel_data) == 0:
        print('There is a problem with the trace, it is missing critical data.')
        return 1

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

    print('Data points without kernel:', no_kern, "({} {:.3f}%)".format(no_kern_sum, no_kern_sum / reg_total * 100))

    kernel_sums = {}
    for kern in kernel_data:
        sum = 0
        for val in kernel_data[kern]:
            sum += val
        kernel_sums[kern] = sum

    sorted_sums = sorted(kernel_sums.items(), key=lambda x: x[1], reverse=True)

    top_n = min(args.n, len(sorted_sums))
    print('Top', top_n, 'offenders (total events)')
    for kernel, sum in sorted_sums[:top_n]:
        print(kernel, sum, '{:.3f}%'.format(sum/reg_total*100.0))

    sorted_rate = sorted([(x[0], x[1] / kernel_time[x[0]] if kernel_time[x[0]] > 0 else 0.0)
                         for x in kernel_sums.items()], key=lambda x: x[1], reverse=True)

    print('Top', top_n, 'offenders (events/us)')
    for kernel, rate in sorted_rate[:top_n]:
        print(kernel, '{:.3f}'.format(rate))

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
