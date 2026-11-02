"""Record final inference artifacts without changing original training provenance."""
import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = tomllib.loads((ROOT/'config.toml').read_text(encoding='utf-8'))['model']
folder = ROOT/model.replace('roster_transformer', 'artifacts_roster', 1)
paths = [ROOT/'config.toml', ROOT/'main.py', ROOT/'core.py', ROOT/'requirements.txt']
for name in ['models', 'data_gen', 'pipelines']:
    paths.extend(sorted((ROOT/name).glob('*.py')))
paths.extend(sorted(path for path in folder.rglob('*') if path.is_file()))
paths.extend([ROOT/'data/swiss_2026_results.json', ROOT/'data/swiss_2026_complete.json', ROOT/'data/worlds_2026_knockouts.json'])
manifest = {'date': '2026-11-02', 'model': model,
            'files': {path.relative_to(ROOT).as_posix(): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                                       'bytes': path.stat().st_size} for path in paths}}
(ROOT/'docs/release_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
print(f'Recorded {len(paths)} release files.')
