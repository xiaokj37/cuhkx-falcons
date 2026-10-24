"""Static code-only release checks; no GPU, dataset, or external service is used."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.npy', '.npz', '.csv', '.docx', '.pdf', '.heic', '.jpg', '.png'}
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'access_token': re.compile(r'\b(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'private_host': re.compile(r'(?<![\w.=])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})(?![\w.])'),
    'private_home': re.compile(r'/(?:Users|root)/(?:workspace|xiaokj)/'),
    'signed_material': re.compile(r'(?i)honor[_]declaration'),
}


def release_files():
    result = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], capture_output=True)
    if result.returncode == 0 and result.stdout:
        return [ROOT / p.decode() for p in result.stdout.split(b'\0') if p]
    ignored = {'.git', '__pycache__', '.pytest_cache', '.venv', '.conda', 'runs', 'outputs', 'data', 'cache', 'pretrained'}
    return [p for p in ROOT.rglob('*') if p.is_file() and not any(x in ignored for x in p.relative_to(ROOT).parts)]


def main():
    issues = []
    files = release_files()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink() or path.suffix.lower() in FORBIDDEN_SUFFIXES or path.stat().st_size > 2_000_000:
            issues.append((relative, 'Unexpected binary/data/large file or symlink'))
        text = path.read_text(encoding='utf-8')
        if path.suffix == '.py':
            ast.parse(text, filename=relative)
        for name, regex in PATTERNS.items():
            if regex.search(text):
                issues.append((relative, name))
    mapping = json.loads((ROOT / 'docs/SOURCE_MAP.json').read_text())
    for name, spec in mapping['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != spec['release_sha256']:
            issues.append((name, 'Release source hash differs'))
    print(json.dumps(dict(files=len(files), python_files=sum(p.suffix == '.py' for p in files), issues=issues), indent=2))
    raise SystemExit(bool(issues))


if __name__ == '__main__':
    main()
