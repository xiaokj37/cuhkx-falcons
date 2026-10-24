"""Download only the three public model assets, never challenge data or credentials."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    'video': ('videomaev2_vitb_k710_distilled_original.pth',
              'https://huggingface.co/OpenGVLab/VideoMAE2/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/distill/vit_b_k710_dl_from_giant.pth',
              '8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec'),
    'imu': ('limu_bert_x_original.pt',
            'https://github.com/WANDS-HKUST/LIMU-BERT_Experience/raw/bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb/weights/limu_bert_x.pt',
            '65697e7017020f36429a221382253f1fc370fdfd493cfb71667f95846c18516c'),
    'detector': ('yolo11n.pt', 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
                 '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(name, directory):
    filename, url, expected = ASSETS[name]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if path.exists():
        if sha256(path) != expected:
            raise ValueError('Existing model asset has the wrong SHA-256; preserved')
        print('Verified existing asset:', filename, flush=True)
        return
    partial = path.with_name(path.name + '.part')
    if partial.exists():
        raise ValueError('An interrupted .part file exists; inspect it before retrying')
    request = urllib.request.Request(url, headers={'User-Agent': 'cuhkx-falcons/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as handle:
        while block := response.read(8 * 1024 * 1024):
            handle.write(block)
    if sha256(partial) != expected:
        raise ValueError('Downloaded asset failed its pinned hash; .part preserved')
    partial.replace(path)
    print('Verified:', filename, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=list(ASSETS), default=list(ASSETS))
    parser.add_argument('--directory', type=Path, default=ROOT / 'pretrained')
    args = parser.parse_args()
    for name in args.only:
        download(name, args.directory)


if __name__ == '__main__':
    main()
