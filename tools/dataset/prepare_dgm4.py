"""Build a reproducible REAL / AI_EDITED subset from DGM4 JSON and ZIPs."""
import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Folder containing JSON and ZIP files')
    parser.add_argument('--output', type=Path, required=True, help='New, empty output folder')
    parser.add_argument('--counts', type=int, nargs=3, default=[4000, 500, 500], metavar=('TRAIN', 'VAL', 'TEST'), help='Images per class in each split')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.counts) < 1:
        parser.error('Counts must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty; existing files will not be overwritten')
    rng = random.Random(args.seed)
    splits = ['train', 'val', 'test']
    rows = {s: json.loads((args.source / (s + '.json')).read_text(encoding='utf-8')) for s in splits}
    # Preserve official splits; exclude ambiguous source IDs/paths instead of moving them.
    owners = defaultdict(set)
    for split, items in rows.items():
        for row in items:
            owners['id:' + str(row['id'])].add(split)
            owners['path:' + row['image']].add(split)
    skipped = Counter()
    manifest = []
    hashes = set()
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / 'INCOMPLETE.txt'
    marker.write_text('Build running or interrupted. Use only after report.json confirms completion.', encoding='utf-8')
    with ExitStack() as stack:
        archives = {}
        index = {}
        for name in ['bbc', 'usa_today', 'simswap', 'StyleCLIP']:
            archives[name] = stack.enter_context(zipfile.ZipFile(args.source / (name + '.zip')))
            index[name] = {}
            for member in archives[name].namelist():
                parts = member.replace('\\', '/').split('/')
                if name in parts and member.lower().endswith(('.jpg', '.jpeg', '.png')):
                    key = '/'.join(parts[parts.index(name):])
                    if key in index[name]:
                        raise ValueError('Ambiguous ZIP member: ' + key)
                    index[name][key] = member
        for split, count in zip(splits, args.counts):
            pools = defaultdict(list)
            seen = set()
            for row in rows[split]:
                path = row['image']
                if path in seen:
                    continue
                seen.add(path)
                if len(owners['id:' + str(row['id'])]) > 1 or len(owners['path:' + path]) > 1:
                    skipped['cross_split_source_or_path'] += 1
                    continue
                parts = path.replace('\\', '/').split('/')
                if 'origin' in parts and row['fake_cls'] == 'orig':
                    key = '/'.join(parts[parts.index('origin') + 1:])
                    group = 'real'
                elif 'manipulation' in parts and row.get('fake_image_box') and any(t in row['fake_cls'].split('&') for t in ['face_swap', 'face_attribute']):
                    key = '/'.join(parts[parts.index('manipulation') + 1:])
                    group = key.split('/')[0]
                else:
                    continue
                archive = key.split('/')[0]
                if archive in index and key in index[archive] and group in ['real', 'simswap', 'StyleCLIP']:
                    pools[group].append((row, archive, index[archive][key]))
            for group, target in [('real', count), ('simswap', count // 2), ('StyleCLIP', count - count // 2)]:
                rng.shuffle(pools[group])
                label = 'real' if group == 'real' else 'ai_edited'
                folder = args.output / split / label
                folder.mkdir(parents=True, exist_ok=True)
                accepted = 0
                for row, archive, member in pools[group]:
                    if accepted == target:
                        break
                    try:
                        data = archives[archive].read(member)  # ZIP CRC checked on read.
                        with Image.open(io.BytesIO(data)) as im:
                            im = ImageOps.exif_transpose(im).convert('RGB')
                            im.load()
                            digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                            width, height = im.size
                    except (OSError, ValueError, zipfile.BadZipFile) as exc:
                        skipped['unreadable_image'] += 1
                        print('Skip unreadable:', member, str(exc), flush=True)
                        continue
                    if digest in hashes:
                        skipped['duplicate_pixels'] += 1
                        continue
                    hashes.add(digest)
                    dest = folder / (archive + '_' + hashlib.sha256(row['image'].encode()).hexdigest()[:20] + Path(member).suffix.lower())
                    dest.write_bytes(data)
                    manifest.append(dict(image=dest.relative_to(args.output).as_posix(), label=label.upper(), split=split, source_id=str(row['id']), method=archive if label == 'ai_edited' else 'original', source_image=row['image'], fake_cls=row['fake_cls'], fake_image_box=json.dumps(row.get('fake_image_box', [])), pixel_sha256=digest, width=width, height=height))
                    accepted += 1
                print(f'{split}/{group}: {accepted}/{target}', flush=True)
                if accepted != target:
                    raise RuntimeError(f'Insufficient valid images for {split}/{group}. Output is incomplete; use a new output folder on retry.')
    with (args.output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    report = dict(complete=True, seed=args.seed, total=len(manifest), counts=dict(Counter(r['split'] + '/' + r['label'] for r in manifest)), skipped=dict(skipped), scope='DGM4 face manipulation: SimSwap and StyleCLIP', deduplication='Exact decoded RGB pixels; source IDs and paths cannot span splits. Near-duplicates are not detected.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    marker.unlink()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
