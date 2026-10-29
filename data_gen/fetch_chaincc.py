"""Download CC BY 4.0 team CSVs using bounded HTTP range requests."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
from core import LEAGUES

ROOT = Path(__file__).resolve().parents[1]
LEAGUE_ALIASES = {'WLDs': 'WCS', 'Worlds': 'WCS', 'LTA North': 'LTA N', 'LTA South': 'LTA S'}


def get_range(url, start, end):
    cache = ROOT/'data'/'raw'/'chunks'/Path(url).name/f'{start}-{end}.part'
    if cache.exists() and cache.stat().st_size == end-start+1:
        return cache.read_bytes(), (cache.with_suffix('.range').read_text())
    last_error = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
                content_range = r.headers.get('Content-Range', '')
                if r.status != 206 or not content_range.startswith(f'bytes {start}-'):
                    raise ValueError(f'Server did not honour range: {r.status} {content_range}')
                if len(data) != end-start+1:
                    raise ValueError(f'Truncated chunk {start}-{end}: {len(data)}')
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
                cache.with_suffix('.range').write_text(content_range)
                return data, content_range
        except Exception as error:
            last_error = error
    raise last_error


def main():
    (ROOT/'data'/'raw').mkdir(parents=True, exist_ok=True)
    metadata, selected = [], []
    for year in [2024, 2025, 2026]:
        url = f'https://chaincc.lol/data/chaincc-teams-{year}.csv.gz'
        path = ROOT/'data'/'raw'/f'chaincc-teams-{year}.csv.gz'
        if path.exists():
            compressed = path.read_bytes()
        else:
            head, header = get_range(url, 0, 16383)
            total = int(re.search(r'/(\d+)$', header).group(1))
            pieces = {0: head}
            jobs = [(start, min(start+16384-1, total-1)) for start in range(len(head), total, 16384)]
            print(f'Downloading {year}: {total} bytes in {len(jobs)+1} chunks', flush=True)
            with ThreadPoolExecutor(max_workers=12) as pool:
                pending = {pool.submit(get_range, url, start, end): start for start, end in jobs}
                for i, future in enumerate(as_completed(pending), 1):
                    pieces[pending[future]] = future.result()[0]
                    if i % 8 == 0:
                        print(f'{year}: {i}/{len(jobs)} chunks', flush=True)
            compressed = b''.join(pieces[start] for start in sorted(pieces))
            gzip.decompress(compressed)  # Validate gzip CRC before persisting.
            path.write_bytes(compressed)
        raw = gzip.decompress(compressed)
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        renames = {'game_id': 'gameid', 'game_number': 'game', 'team_name': 'teamname', 'team_id': 'teamid'}
        for source in rows:
            league = LEAGUE_ALIASES.get(source['league'], source['league'])
            if league not in LEAGUES:
                continue
            row = {renames.get(k, k): v for k, v in source.items()}
            row['league'] = league
            row['result'] = str(int(row['result'].lower() == 'true'))
            row['playoffs'] = str(int(row['playoffs'].lower() == 'true'))
            row['roster'] = ''
            selected.append(row)
        meta = {'year': year, 'url': url, 'sha256_gzip': hashlib.sha256(compressed).hexdigest(),
                'sha256_uncompressed': hashlib.sha256(raw).hexdigest(), 'rows': len(rows),
                'dates': [min(r['date'] for r in rows), max(r['date'] for r in rows)],
                'selected_rows': sum(r['year'] == str(year) for r in selected),
                'bo_format_counts': dict(__import__('collections').Counter(r['bo_format'] for r in rows)),
                'league_row_counts': dict(__import__('collections').Counter(r['league'] for r in rows)),
                'attribution': 'ChainCC. League of Legends esports match dataset. https://chaincc.lol/free/data ; CC BY 4.0; accessed 2026-10-28'}
        metadata.append(meta)
        print(json.dumps(meta), flush=True)
    fields = sorted(set().union(*(r.keys() for r in selected)))
    with (ROOT/'data'/'teams.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(selected)
    (ROOT/'data'/'sources.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved {len(selected)} team rows', flush=True)


if __name__ == '__main__':
    main()
