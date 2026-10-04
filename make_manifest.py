"""Produce a reviewable publication inventory without credentials or host paths."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
entries = []
for path in sorted(root.rglob('*')):
    if not path.is_file():
        continue
    rel = path.relative_to(root)
    if any(part in ('.git', '__pycache__', 'runs') or part.startswith('.venv') for part in rel.parts):
        continue
    if path.name == 'MANIFEST.json' or path.suffix in ('.pyc', '.tmp') or path.name.endswith('.tmp.pt'):
        continue
    if path.suffix == '.log' and rel.parts[0] != 'results':
        continue
    data = path.read_bytes()
    if path.suffix in ('.py', '.md', '.json', '.txt', '.log', '.jsonl'):
        text = data.decode('utf-8', errors='replace')
        if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----', text):
            raise RuntimeError('Credential-like content in '+rel.as_posix())
    entries.append({'path': rel.as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
manifest = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'origin': 'https://github.com/Kitjesen/worm-sim',
            'origin_base_commit': '04c8177',
            'source_note': 'Curated working-tree snapshot, including previously uncommitted experiments; not an unchanged copy of the origin commit.',
            'scope': 'Current standalone GPU core/training, selected relevant evidence and demos; not the complete historical project archive.',
            'file_count': len(entries), 'total_bytes': sum(e['bytes'] for e in entries), 'files': entries}
(root/'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({'files': len(entries), 'bytes': manifest['total_bytes']}))
