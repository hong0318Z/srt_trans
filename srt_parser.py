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


def get_output_paths(input_path: str, detected_lang_code: str = 'zh') -> tuple[str, str]:
    """Returns (origin_path, output_srt_path).
    origin_path: where to move the original file  (dir/origin/original_filename.srt)
    output_srt_path: output path for the translation  (dir/base.srt, no lang tag)
    """
    p    = Path(input_path)
    dir_ = p.parent

    known = {'.zh', '.en', '.ja', '.ko', '.cn', '.chs', '.cht'}
    stem = p.stem
    base = stem
    for tag in known:
        if stem.lower().endswith(tag):
            base = stem[:len(stem) - len(tag)]
            break

    origin_path = str(dir_ / 'origin' / p.name)
    output_path = str(dir_ / f"{base}.srt")
    return origin_path, output_path


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


def write_cleaned_srt(blocks: List[SubtitleBlock], filepath: str):
    """Write SRT after cleanup (overwrite). Re-numbers sequentially."""
    lines = []
    for i, block in enumerate(blocks, 1):
        lines.extend([str(i), f"{block.start} --> {block.end}", block.text, ''])
    Path(filepath).write_text('\n'.join(lines), encoding='utf-8')
