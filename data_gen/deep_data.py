"""Prepare domestic pretraining / Worlds adaptation data with frozen histories."""
import csv
import gzip
import io
import json
from collections import Counter
from core import ROOT
from data_gen.fetch_chaincc import LEAGUE_ALIASES
from data_gen.features_v2 import build
import numpy as np

OUT = ROOT/'artifacts_deep'
DOMESTIC = {'LCK', 'LPL', 'LEC', 'LCS', 'LTA N', 'LTA S', 'LCP', 'PCS', 'VCS',
            'CBLOL', 'LLA', 'LJL', 'LCO', 'TCL', 'LFL', 'PRM', 'LVP SL', 'NLC', 'AL',
            'LIT', 'EBL', 'LPLOL', 'UL', 'GLL', 'ESLOL', 'HLL', 'ROL', 'NACL', 'LCKC',
            'LDL', 'CBLOLA', 'LFL2', 'PRMP', 'NEXO', 'LAS', 'LRN', 'LRS', 'HM', 'HC',
            'CD', 'RL', 'LES', 'NL', 'PCL', 'EPL', 'TSC'}
WORLD_WINDOWS = {
    '2024': ('2024-09-25', '2024-11-02', 'https://lolesports.com/en-US/news/worlds-2024'),
    '2025': ('2025-10-14', '2025-11-09', 'https://lolesports.com/en-US/news/worlds-2025-primer'),
    '2026': ('2026-10-15', '2026-11-14', 'https://lolesports.com/en-GB/news/worlds-2026-primer'),
}


def write_all_teams():
    """Refresh the combined team table without touching any model artifacts."""
    rows = []
    for year in [2024, 2025, 2026]:
        path = ROOT/'data'/'raw'/f'chaincc-teams-{year}.csv.gz'
        raw = gzip.decompress(path.read_bytes()).decode('utf-8-sig')
        for source in csv.DictReader(io.StringIO(raw)):
            renames = {'game_id': 'gameid', 'game_number': 'game', 'team_name': 'teamname', 'team_id': 'teamid'}
            row = {renames.get(k, k): v for k, v in source.items()}
            row['league'] = LEAGUE_ALIASES.get(row['league'], row['league'])
            row['result'] = str(int(row['result'].lower() == 'true'))
            row['playoffs'] = str(int(row['playoffs'].lower() == 'true'))
            row['roster'] = ''
            rows.append(row)
    path = ROOT/'data'/'teams_all.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)
    return path, len(rows)


def main():
    OUT.mkdir(exist_ok=True)
    path, row_count = write_all_teams()
    samples, metadata = build(team_path=path, include_first_game=True, sequence_length=12, save_snapshot=OUT/'snapshot.json')
    targets, domains = [], []
    for sample in samples:
        year = sample['date'][:4]
        first, last, _ = WORLD_WINDOWS[year]
        worlds = sample['league'] == 'WCS' and first <= sample['date'] <= last
        domain = 'worlds' if worlds else 'domestic' if sample['league'] in DOMESTIC else 'other'
        sample['domain'] = domain
        sample['worlds_year'] = year if worlds else ''
        targets.append({k: v for k, v in sample.items() if k not in {'d', 'c', 'k_logits', 'seq_a', 'seq_b', 'mask_a', 'mask_b'}})
        domains.append(domain)
    np.savez_compressed(OUT/'data.npz',
                        d=np.stack([s['d'] for s in samples]).astype('float32'),
                        c=np.stack([s['c'] for s in samples]).astype('float32'),
                        seq_a=np.stack([s['seq_a'] for s in samples]), seq_b=np.stack([s['seq_b'] for s in samples]),
                        mask_a=np.stack([s['mask_a'] for s in samples]), mask_b=np.stack([s['mask_b'] for s in samples]),
                        y=np.array([s['y'] for s in samples], dtype='float32'),
                        first_map_y=np.array([s['first_map_y'] for s in samples], dtype='float32'),
                        map_target=np.array([s['map_fraction'] for s in samples], dtype='float32'),
                        game_count=np.array([s['game_count'] for s in samples], dtype='float32'),
                        best_of=np.array([s['best_of'] for s in samples], dtype='int64'),
                        offset=np.array([np.log(10)*s['d'][2] for s in samples], dtype='float32'))
    (OUT/'samples.json').write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata.update({'domain_counts': dict(Counter(domains)), 'worlds_windows': WORLD_WINDOWS,
                     'worlds_counts': dict(Counter(s['worlds_year'] for s in targets if s['domain'] == 'worlds')),
                     'sequence_length': 12, 'sequence_features': 14,
                     'excluded_worlds_regional_qualifiers': sum(s['league'] == 'WCS' and s['domain'] != 'worlds' for s in targets),
                     'source_team_rows': row_count,
                     'one_game_records': 'First-map supervision only, not automatically labelled BO1.'})
    (OUT/'data_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ['source_team_rows', 'domain_counts', 'worlds_counts', 'excluded_worlds_regional_qualifiers']}), flush=True)


if __name__ == '__main__':
    main()
