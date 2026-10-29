"""Download and cache player snapshots; retain pre-game identities and historical stats."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
from data_gen.fetch_chaincc import ROOT, get_range


def download(year):
    url = f'https://chaincc.lol/data/chaincc-players-{year}.csv.gz'
    path = ROOT/'data'/'raw'/Path(url).name
    if path.exists():
        compressed = path.read_bytes()
    else:
        first, header = get_range(url, 0, 16383)
        total = int(re.search(r'/(\d+)$', header).group(1))
        chunks = {0: first}
        jobs = [(s, min(s+65535, total-1)) for s in range(len(first), total, 65536)]
        print(f'{year}: {total} bytes, {len(jobs)} ranges', flush=True)
        def chunk(start, end):
            try:
                return get_range(url, start, end)[0]
            except (OSError, TimeoutError):
                return b''.join(get_range(url, s, min(s+16383, end))[0] for s in range(start, end+1, 16384))
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(chunk, s, e): s for s, e in jobs}
            for i, f in enumerate(as_completed(futures), 1):
                chunks[futures[f]] = f.result()
                if i % 16 == 0:
                    print(f'{year}: {i}/{len(jobs)} ranges', flush=True)
        compressed = b''.join(chunks[s] for s in sorted(chunks))
        gzip.decompress(compressed)
        path.write_bytes(compressed)
    raw = gzip.decompress(compressed)
    meta = {'year': year, 'url': url, 'sha256_gzip': hashlib.sha256(compressed).hexdigest(),
            'sha256_uncompressed': hashlib.sha256(raw).hexdigest(),
            'attribution': 'ChainCC. League of Legends esports match dataset. CC BY 4.0. https://chaincc.lol/free/data ; accessed 2026-10-28'}
    return raw, meta


def main():
    # All leagues retained: a promoted player may have a history in an academy league.
    fields = ['game_id', 'date', 'league', 'patch', 'team_name', 'side', 'position', 'playername',
              'result', 'gamelength', 'kills', 'deaths', 'assists', 'goldat15', 'golddiffat15',
              'xpdiffat15', 'csdiffat15', 'dpm', 'cspm', 'wpm']
    metadata, count = [], 0
    with gzip.open(ROOT/'data'/'players.csv.gz', 'wt', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for year in [2024, 2025, 2026]:
            raw, meta = download(year)
            rows = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
            n = 0
            for row in rows:
                writer.writerow({key: row.get(key, '') for key in fields})
                count += 1
                n += 1
            meta['rows'] = n
            metadata.append(meta)
            print(f'{year}: saved {n} player rows', flush=True)
    (ROOT/'data'/'player_sources.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {count} player rows', flush=True)


if __name__ == '__main__':
    main()
