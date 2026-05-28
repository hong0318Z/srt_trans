import re
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class SubtitleBlock:
    index: int
    start: str
    end: str
    lines: List[str]

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)


def parse_srt(filepath: str) -> List[SubtitleBlock]:
    path = Path(filepath)
    content = None
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'cp932', 'big5', 'latin-1'):
        try:
            content = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if content is None:
        content = path.read_text(encoding='utf-8', errors='replace')

    blocks = []
    for block in re.split(r'\n\s*\n', content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(
            r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
            lines[1].strip(),
        )
        if not m:
            continue
        start = m.group(1).replace('.', ',')
        end   = m.group(2).replace('.', ',')
        text_lines = [l for l in lines[2:] if l.strip()]
        if text_lines:
            blocks.append(SubtitleBlock(index=idx, start=start, end=end, lines=text_lines))
    return blocks


def get_output_paths(input_path: str, detected_lang_code: str) -> tuple[str, str]:
    """Returns (source_srt_path, korean_srt_path).
    If input already has a known language suffix, keeps it; otherwise appends detected code."""
    p    = Path(input_path)
    stem = p.stem
    dir_ = p.parent

    known = {'.zh', '.en', '.ja', '.ko', '.cn', '.chs', '.cht'}
    # check if stem ends with a known lang tag
    base = stem
    for tag in known:
        if stem.lower().endswith(tag):
            base = stem[:len(stem)-len(tag)]
            break
    else:
        # no known tag → name source file with detected code
        src_name = f"{stem}.{detected_lang_code}.srt"
        return str(dir_ / src_name), str(dir_ / f"{stem}.ko.srt")

    return str(dir_ / f"{base}{tag}.srt"), str(dir_ / f"{base}.ko.srt")


def write_ko_srt(blocks: List[SubtitleBlock], translated: List[str], filepath: str):
    """Write Korean SRT. Blocks with empty translation (OCR labels) are skipped."""
    lines = []
    seq   = 1
    for block, trans in zip(blocks, translated):
        if not trans:
            continue
        lines.extend([str(seq), f"{block.start} --> {block.end}", trans, ''])
        seq += 1
    Path(filepath).write_text('\n'.join(lines), encoding='utf-8')
