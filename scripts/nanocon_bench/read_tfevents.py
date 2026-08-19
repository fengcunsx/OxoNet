"""把 pytorch-lightning 的 tfevents 里的标量导出成 CSV/表格（不用起 tensorboard 服务）。

NanoCon 训练日志的用途：给审稿人证明"训练已收敛"，比"我们跑满 N 轮"更有力。

    /home/bio/anaconda3/bin/python read_tfevents.py <events 文件> [--csv out.csv]
"""
import argparse
import collections

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('events')
    p.add_argument('--csv', default='')
    p.add_argument('--per-epoch', action='store_true', help='按 epoch 而不是 step 汇总')
    args = p.parse_args()

    ea = EventAccumulator(args.events, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags()['scalars']
    print('标量 tag ({} 个): {}'.format(len(tags), ', '.join(tags)))

    # step -> {tag: value}
    rows = collections.defaultdict(dict)
    for t in tags:
        for e in ea.Scalars(t):
            rows[e.step][t] = e.value

    show = [t for t in tags if t.startswith('avg_val') or t == 'epoch'] or tags
    print('\n{:>8s} '.format('step') + ' '.join('{:>18s}'.format(t[-18:]) for t in show))
    for s in sorted(rows):
        r = rows[s]
        if not any(t in r for t in show):
            continue
        print('{:>8d} '.format(s) + ' '.join(
            '{:>18.5f}'.format(r[t]) if t in r else ' ' * 18 for t in show))

    if args.csv:
        import csv as _csv
        with open(args.csv, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['step'] + tags)
            for s in sorted(rows):
                w.writerow([s] + [rows[s].get(t, '') for t in tags])
        print('\n写出 ' + args.csv)


if __name__ == '__main__':
    main()
