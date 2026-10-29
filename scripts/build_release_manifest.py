"""Record final inference artifacts without changing original training provenance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'artifacts_roster_2026_swiss_weighted'
paths = [ROOT/'config.toml', ROOT/'main.py', ROOT/'core.py', ROOT/'requirements.txt']
for name in ['models', 'data_gen', 'pipelines']:
    paths.extend(sorted((ROOT/name).glob('*.py')))
paths.extend(sorted(path for path in folder.rglob('*') if path.is_file()))
manifest = {'date': '2026-10-29', 'model': 'roster_transformer_2026_swiss_weighted',
            'files': {path.relative_to(ROOT).as_posix(): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                                       'bytes': path.stat().st_size} for path in paths}}
(ROOT/'docs/release_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
print(f'Recorded {len(paths)} release files.')
