"""GPU health probe -- catches the Xid 62 (PMU halt) degraded state.

After an Xid 62 the GPU stays stuck at minimum clocks and every clock/power
query returns "System is not in ready state"; compute still runs but at ~1/30
of the real speed, so a long job silently turns into a 10-hour job. This probe
gives a hard OK / DEGRADED verdict in ~5 seconds.

    python gpu_health.py            # verdict + exit code (0 = OK, 1 = degraded)
    python gpu_health.py --watch 60 # poll forever, log a line per check
"""
import argparse
import subprocess
import sys
import time

# a healthy 4060 Laptop does ~200-250 GB/s; after a PMU halt it does ~6
MIN_BANDWIDTH_GBS = 100.0


def read_clocks():
    """(sm_clock_str, power_str) -- 'not in ready state' means telemetry is dead."""
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=clocks.sm,power.draw,temperature.gpu',
         '--format=csv,noheader'],
        capture_output=True, text=True).stdout.strip()
    return out


def measure_bandwidth():
    import torch
    x = torch.empty(64 * 1024 * 1024, device='cuda')   # 256 MB fp32
    for _ in range(3):
        x.mul_(1.0001)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        x.mul_(1.0001)
    torch.cuda.synchronize()
    dt = (time.time() - t) / 20
    return 2 * x.numel() * 4 / dt / 1e9


def check():
    clocks = read_clocks()
    telemetry_dead = 'not in ready state' in clocks
    gbs = measure_bandwidth()
    ok = (gbs >= MIN_BANDWIDTH_GBS) and not telemetry_dead
    return ok, gbs, clocks


def recent_xid(minutes=30):
    """Any Xid in the kernel log recently? (empty string if none / no access)"""
    out = subprocess.run(['journalctl', '-k', '--since', '-{}m'.format(minutes),
                          '--no-pager'], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if 'Xid' in l]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', type=int, default=0, help='seconds between checks; 0 = one shot')
    ap.add_argument('--xid', action='store_true', help='also report recent Xid lines')
    args = ap.parse_args()

    while True:
        ok, gbs, clocks = check()
        print('[{}] {}  bandwidth={:.0f} GB/s  |  {}'.format(
            time.strftime('%H:%M:%S'), 'OK      ' if ok else 'DEGRADED', gbs, clocks), flush=True)
        if args.xid:
            for l in recent_xid():
                print('   ', l, flush=True)
        if not args.watch:
            sys.exit(0 if ok else 1)
        time.sleep(args.watch)
