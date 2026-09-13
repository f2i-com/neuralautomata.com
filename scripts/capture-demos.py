"""Capture actual local runs for the README and landing page.

Requires Python Playwright, installed Chrome, ffmpeg on PATH, the bundled research models and CUDA PyTorch. No synthetic model frames.
Run from repository root: python scripts/capture-demos.py
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'web/recordings'


def capture(page, name, frames, seconds=10, fps=6):
    folder = frames / name
    folder.mkdir()
    count = round(seconds * fps)
    start = time.perf_counter()
    for i in range(count):
        remaining = start + i / fps - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        page.screenshot(path=str(folder / f'{i:04}.png'))
    elapsed = time.perf_counter() - start
    rate = count / elapsed
    ffmpeg = shutil.which('ffmpeg')
    palette = folder / 'palette.png'
    common = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(rate), '-i', str(folder / '%04d.png')]
    subprocess.run(common + ['-vf', 'scale=1120:-1:flags=lanczos,palettegen=stats_mode=diff', '-frames:v', '1', '-update', '1', str(palette)], check=True)
    subprocess.run(common + ['-i', str(palette), '-lavfi', 'scale=1120:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle', '-loop', '0', str(OUT / f'{name}.gif')], check=True)
    # A separate static poster supports readers who prefer reduced motion.
    shutil.copyfile(folder / f'{count - 1:04}.png', OUT / f'{name}.png')
    print(f'{name}: {count} real frames over {elapsed:.1f}s, {(OUT / (name + ".gif")).stat().st_size / 2**20:.2f} MiB', flush=True)
    return dict(frames=count, capturedSeconds=round(elapsed, 2), playbackFps=round(rate, 3))


def main():
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg must be on PATH')
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    frames = ROOT / 'target/demo-capture' / stamp
    frames.mkdir(parents=True)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    owned = False
    with (frames / 'server.log').open('w') as log:
        server = subprocess.Popen(['node', str(ROOT / 'server/serve.cjs')], cwd=ROOT,
                                  env=dict(os.environ, PORT=str(port)), stdout=log, stderr=log)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError('Capture server failed to start')
                if f'native GPU lab: {base}/' in (frames / 'server.log').read_text():
                    owned = True
                    break
                time.sleep(.1)
            if not owned:
                raise RuntimeError('Capture server did not become ready')
            with sync_playwright() as p:
                browser = p.chromium.launch(channel=os.environ.get('PLAYWRIGHT_CHANNEL', 'chrome'), headless=True)
                page = browser.new_page(viewport=dict(width=1400, height=960), device_scale_factor=1)
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                clips = {}
                page.set_viewport_size(dict(width=1400, height=1200))
                page.goto(base + '/')
                page.wait_for_function("!document.querySelector('#run').disabled", timeout=75000)
                status = page.request.get(base + '/api/nca/status').json()
                devices = [d['id'] for d in status['devices'] if d['usable']]
                if not devices:
                    raise RuntimeError('Need at least one verified CUDA device for NCA recordings')
                headers = {'content-type': 'application/json', 'x-nca-token': status['token']}
                for d in devices[:2]:
                    page.locator(f'#devices input[value="{d}"]').check()
                run = page.request.post(base + '/api/nca/run', headers=headers, data=dict(mode='memory', devices=devices[:2], frameMs=500))
                assert run.status == 202
                page.wait_for_function("document.querySelector('#frame-status').textContent.includes('write 1')", timeout=60000)
                clips['nca-memory'] = capture(page, 'nca-memory', frames, seconds=10)
                page.wait_for_function("!document.querySelector('#run').disabled", timeout=60000)
                memory_results = page.request.get(base + '/api/nca/status').json()['jobs']
                assert all(j['state'] == 'complete' and j['result']['sharedWeightsUnchanged'] for j in memory_results)

                page.locator('#mode').select_option('language')
                page.locator('#bytes').fill('100')
                page.locator('#run').click()
                page.wait_for_function("document.querySelector('#frame-status').textContent.includes('generation')", timeout=60000)
                clips['nca-language'] = capture(page, 'nca-language', frames, seconds=12)
                page.wait_for_function("!document.querySelector('#run').disabled", timeout=60000)
                assert all(j['state'] == 'complete' for j in page.request.get(base + '/api/nca/status').json()['jobs'])
                if errors:
                    raise RuntimeError(str(errors))
                tracked_sources = ['server/native_lab.py', 'web/nca-renderer.mjs']
                provenance = dict(capturedAtUtc=stamp, browser=browser.version,
                    torch=status['torch'], cuda=status['cuda'],
                    devices=[{key: value for key, value in device.items() if key != 'uuid'} for device in status['devices']], clips=clips,
                    sourceSha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in tracked_sources},
                    note='Actual local browser captures. NCA runs in native CPython/PyTorch CUDA; WebGL2 renders model state. GIFs are demonstrations, not benchmarks.')
                (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
                browser.close()
        finally:
            if owned and server.poll() is None:
                try:
                    status = json.load(urllib.request.urlopen(base + '/api/nca/status', timeout=5))
                    req = urllib.request.Request(base + '/api/nca/stop', data=b'{}', headers={'Content-Type': 'application/json', 'X-Nca-Token': status['token']})
                    urllib.request.urlopen(req, timeout=3).close()
                except OSError:
                    pass
            server.terminate()
            server.wait(timeout=10)


if __name__ == '__main__':
    main()
