import sys
import copy
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QTableWidget, QTableWidgetItem, QCheckBox, QProgressBar,
    QScrollArea, QSizePolicy, QMessageBox, QFileDialog,
    QHeaderView, QFrame, QSpinBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer
from PyQt6.QtGui import QFont

from llm_api import LLMClient
from srt_parser import parse_srt, write_ko_srt, write_cleaned_srt, get_output_paths
from translator import SRTTranslator, BATCH_SIZE as DEFAULT_BATCH_SIZE

# ── Page indices ──────────────────────────────────────────────────────────────
P_SETUP    = 0
P_LOADING  = 1
P_ANALYSIS = 2
P_SAMPLE   = 3
P_TRANS    = 4
P_REVIEW   = 5
P_DONE     = 6
P_CLEANUP  = 7
P_MISSING  = 8

# ── Stylesheet ────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget { background: #1e1e2e; color: #cdd6f4;
    font-family: "Segoe UI", "Malgun Gothic", "Noto Sans KR", sans-serif;
    font-size: 13px; }
QLabel { color: #cdd6f4; }
QLabel#title   { font-size: 22px; font-weight: bold; color: #89b4fa; }
QLabel#subtitle{ font-size: 13px; color: #a6adc8; }
QLabel#section { font-size: 14px; font-weight: bold; color: #89b4fa; padding-top:10px; }
QLabel#info    { color: #a6e3a1; }
QLabel#warn    { color: #fab387; }
QPushButton {
    background: #89b4fa; color: #1e1e2e; border: none;
    padding: 8px 20px; border-radius: 6px; font-weight: bold; }
QPushButton:hover   { background: #b4d0f7; }
QPushButton:disabled{ background: #45475a; color: #6c7086; }
QPushButton#secondary { background: #45475a; color: #cdd6f4; font-weight: normal; }
QPushButton#secondary:hover { background: #585b70; }
QPushButton#danger  { background: #f38ba8; }
QPushButton#danger:hover { background: #f5a0b5; }
QPushButton#green   { background: #a6e3a1; color: #1e1e2e; }
QPushButton#green:hover { background: #c3f0c0; }
QLineEdit, QTextEdit {
    background: #313244; border: 1px solid #45475a;
    border-radius: 5px; padding: 6px 10px; color: #cdd6f4; }
QLineEdit:focus, QTextEdit:focus { border-color: #89b4fa; }
QTableWidget {
    background: #313244; border: 1px solid #45475a;
    gridline-color: #45475a; border-radius: 4px;
    alternate-background-color: #2a2a3e; }
QTableWidget::item { padding: 4px 8px; }
QTableWidget::item:selected { background: #45475a; }
QHeaderView::section {
    background: #45475a; color: #cdd6f4; padding: 6px 8px;
    border: none; border-right: 1px solid #585b70; font-weight: bold; }
QProgressBar {
    border: 1px solid #45475a; border-radius: 5px;
    background: #313244; text-align: center; color: #cdd6f4; }
QProgressBar::chunk { background: #89b4fa; border-radius: 4px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #313244; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #585b70; border-radius: 4px; }
QCheckBox { color: #cdd6f4; spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #585b70;
    border-radius: 3px; background: #313244; }
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QFrame#card { background: #313244; border: 1px solid #45475a;
    border-radius: 8px; padding: 12px; }
QSpinBox { background: #313244; border: 1px solid #45475a;
    border-radius: 5px; padding: 4px 8px; color: #cdd6f4; }
"""

# ── Workers ───────────────────────────────────────────────────────────────────

class TestWorker(QThread):
    done  = pyqtSignal(str)
    error = pyqtSignal(str)
    def __init__(self, client):
        super().__init__(); self.client = client
    def run(self):
        try:    self.done.emit(self.client.test_connection())
        except Exception as e: self.error.emit(str(e))


class AnalysisWorker(QThread):
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, translator, blocks):
        super().__init__()
        self.translator = translator; self.blocks = blocks
    def run(self):
        try:    self.done.emit(self.translator.analyze(self.blocks))
        except Exception as e: self.error.emit(str(e))


class CleanupWorker(QThread):
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, translator, blocks):
        super().__init__()
        self.translator = translator; self.blocks = blocks
    def run(self):
        try:    self.done.emit(self.translator.cleanup_analyze(self.blocks))
        except Exception as e: self.error.emit(str(e))


class PlanWorker(QThread):
    done  = pyqtSignal(str)
    error = pyqtSignal(str)
    def __init__(self, translator, instructions):
        super().__init__()
        self.translator = translator; self.instructions = instructions
    def run(self):
        try:    self.done.emit(self.translator.get_translation_plan(self.instructions))
        except Exception as e: self.error.emit(str(e))


class SampleWorker(QThread):
    done  = pyqtSignal(str)
    error = pyqtSignal(str)
    def __init__(self, translator, blocks):
        super().__init__()
        self.translator = translator; self.blocks = blocks
    def run(self):
        try:    self.done.emit(self.translator.translate_sample(self.blocks))
        except Exception as e: self.error.emit(str(e))


class TranslationWorker(QThread):
    progress     = pyqtSignal(int, int)
    batch_done   = pyqtSignal(int, list, int, int, int, int)
    partial_save = pyqtSignal(list)
    done         = pyqtSignal(list)
    error        = pyqtSignal(str)

    def __init__(self, translator, blocks, batch_size: int = DEFAULT_BATCH_SIZE):
        super().__init__()
        self.translator = translator
        self.blocks     = blocks
        self.batch_size = batch_size
        self._cancel    = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            total    = len(self.blocks)
            results  = [''] * total
            prev_ctx = []
            batch_n  = 0
            cum_in   = 0
            cum_out  = 0

            for start in range(0, total, self.batch_size):
                if self._cancel:
                    break
                batch = self.blocks[start:start + self.batch_size]

                # translate_batch handles label-stripping internally;
                # returns '' for label-only blocks
                trans_map, b_in, b_out = self.translator.translate_batch(
                    batch, prev_ctx)
                cum_in  += b_in
                cum_out += b_out

                preview = []
                for i, block in enumerate(batch):
                    trans = trans_map.get(str(i + 1), '')
                    results[start + i] = trans
                    if trans:
                        prev_ctx.append((block.text, trans))
                        if len(preview) < 3:
                            preview.append((block.text, trans))

                batch_n += 1
                self.progress.emit(min(start + self.batch_size, total), total)
                self.batch_done.emit(batch_n, preview, b_in, b_out, cum_in, cum_out)
                self.partial_save.emit(list(results))

            self.done.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _h_sep():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #45475a;")
    return line

def _label(text, obj_name=''):
    lbl = QLabel(text)
    if obj_name:
        lbl.setObjectName(obj_name)
    return lbl

def _btn(text, obj_name=''):
    b = QPushButton(text)
    if obj_name:
        b.setObjectName(obj_name)
    return b

def _scroll_wrap(widget):
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(widget)
    return sa


# ── Pages ─────────────────────────────────────────────────────────────────────

class SetupPage(QWidget):
    def __init__(self, main_win):
        super().__init__()
        self.main = main_win
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 40, 60, 40)
        root.setSpacing(16)

        root.addWidget(_label('SRT 한국어 번역기', 'title'))
        root.addWidget(_label('중국어 · 영어 · 일본어 자막을 한국어로 번역합니다', 'subtitle'))
        root.addWidget(_h_sep())
        root.addSpacing(10)

        # Token
        root.addWidget(_label('GitHub Copilot 토큰'))
        tok_row = QHBoxLayout()
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText('ghp_xxxxxxxxxxxxxxxxxxxx')
        self.test_btn = _btn('연결 테스트', 'secondary')
        tok_row.addWidget(self.token_edit)
        tok_row.addWidget(self.test_btn)
        root.addLayout(tok_row)
        self.test_status = _label('')
        root.addWidget(self.test_status)

        root.addSpacing(6)

        # File
        root.addWidget(_label('SRT 파일'))
        file_row = QHBoxLayout()
        self.file_label = _label('파일을 선택하세요...')
        self.file_label.setStyleSheet(
            'color:#a6adc8; background:#313244; border:1px solid #45475a;'
            'border-radius:5px; padding:6px 10px;')
        self.file_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pick_btn = _btn('파일 선택', 'secondary')
        file_row.addWidget(self.file_label)
        file_row.addWidget(pick_btn)
        root.addLayout(file_row)

        root.addSpacing(6)

        # Batch size
        root.addWidget(_label('배치 크기 (자막 수 / 회)'))
        batch_row = QHBoxLayout()
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(10, 300)
        self.batch_spin.setSingleStep(10)
        self.batch_spin.setValue(
            int(self.main.settings.value('batch_size', DEFAULT_BATCH_SIZE)))
        self.batch_spin.setFixedWidth(90)
        batch_hint = _label('  숫자가 클수록 컨텍스트가 넓어지지만 오류 가능성↑ (권장: 50~150)')
        batch_hint.setStyleSheet('color:#a6adc8; font-size:12px;')
        batch_row.addWidget(self.batch_spin)
        batch_row.addWidget(batch_hint)
        batch_row.addStretch()
        root.addLayout(batch_row)

        root.addStretch()

        # Bottom buttons
        btn_row = QHBoxLayout()
        self.cleanup_btn = _btn('자막 정리 (노이즈 제거)', 'secondary')
        self.cleanup_btn.setEnabled(False)
        self.missing_btn = _btn('미번역 목록', 'secondary')
        self.start_btn   = _btn('  번역 분석 시작  →')
        self.start_btn.setEnabled(False)
        for b in (self.cleanup_btn, self.missing_btn, self.start_btn):
            b.setFixedHeight(44)
        btn_row.addWidget(self.cleanup_btn)
        btn_row.addWidget(self.missing_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.start_btn)
        root.addLayout(btn_row)

        # Signals
        pick_btn.clicked.connect(self._pick_file)
        self.test_btn.clicked.connect(self._test)
        self.start_btn.clicked.connect(self.main.start_analysis)
        self.cleanup_btn.clicked.connect(self.main.start_cleanup)
        self.missing_btn.clicked.connect(
            lambda: self.main.stack.setCurrentIndex(P_MISSING))
        self.token_edit.textChanged.connect(self._check_ready)
        self.batch_spin.valueChanged.connect(
            lambda v: self.main.settings.setValue('batch_size', v))

        saved = self.main.settings.value('token', '')
        if saved:
            self.token_edit.setText(saved)

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '자막 파일 선택', '', 'SRT 파일 (*.srt);;모든 파일 (*)')
        if path:
            self.main.srt_path = path
            self.file_label.setText(path)
            self._check_ready()

    def _check_ready(self):
        ok = bool(self.token_edit.text().strip() and self.main.srt_path)
        self.start_btn.setEnabled(ok)
        self.cleanup_btn.setEnabled(ok)
        # 미번역 목록은 파일 없이도 사용 가능 (디렉토리만 지정)
        self.missing_btn.setEnabled(True)

    def _test(self):
        token = self.token_edit.text().strip()
        if not token:
            self.test_status.setText('토큰을 입력하세요.')
            return
        self.test_btn.setEnabled(False)
        self.test_status.setText('연결 확인 중...')
        client   = LLMClient(token)
        self._tw = TestWorker(client)
        self._tw.done.connect(lambda r: self._test_done(r, token))
        self._tw.error.connect(self._test_err)
        self._tw.start()

    def _test_done(self, result, token):
        self.test_status.setStyleSheet('color:#a6e3a1;')
        self.test_status.setText(f'✓ 연결 성공: {result[:60]}')
        self.test_btn.setEnabled(True)
        self.main.settings.setValue('token', token)

    def _test_err(self, err):
        self.test_status.setStyleSheet('color:#f38ba8;')
        self.test_status.setText(f'✗ 오류: {err[:80]}')
        self.test_btn.setEnabled(True)

    @property
    def token(self): return self.token_edit.text().strip()

    @property
    def batch_size(self): return self.batch_spin.value()


class LoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label = _label('', 'title')
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.label)
        self._dots  = 0
        self._base  = ''
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self, msg='LLM이 분석 중입니다'):
        self._base = msg; self._dots = 0
        self.label.setText(msg)
        self._timer.start(500)

    def stop(self): self._timer.stop()

    def _tick(self):
        self._dots = (self._dots + 1) % 4
        self.label.setText(self._base + '.' * self._dots)


class AnalysisPage(QWidget):
    request_sample = pyqtSignal()
    request_start  = pyqtSignal()

    def __init__(self, main_win):
        super().__init__()
        self.main         = main_win
        self.ocr_checks: list = []
        self._plan_worker = None
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        self._lay = QVBoxLayout(content)
        self._lay.setContentsMargins(40, 24, 40, 24)
        self._lay.setSpacing(12)
        outer.addWidget(_scroll_wrap(content))

        bar = QWidget()
        bar.setStyleSheet('background:#313244; border-top:1px solid #45475a;')
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(40, 12, 40, 12)
        self.sample_btn = _btn('샘플 번역 보기', 'secondary')
        self.trans_btn  = _btn('  번역 시작  →')
        self.trans_btn.setFixedHeight(40)
        bar_lay.addStretch()
        bar_lay.addWidget(self.sample_btn)
        bar_lay.addWidget(self.trans_btn)
        outer.addWidget(bar)

        self.sample_btn.clicked.connect(self.request_sample)
        self.trans_btn.clicked.connect(self.request_start)

    def populate(self, result: dict, total_blocks: int):
        lay = self._lay
        while lay.count():
            item = lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.ocr_checks.clear()

        # 작품 정보
        lay.addWidget(_label('분석 결과', 'section'))
        card = QFrame(); card.setObjectName('card')
        cl   = QVBoxLayout(card)
        lang    = result.get('language_name', '알 수 없음')
        tok_in  = result.get('_tok_in', 0)
        tok_out = result.get('_tok_out', 0)
        tok_str = f'  |  분석 토큰 IN {tok_in:,} / OUT {tok_out:,}' if tok_in else ''
        cl.addWidget(_label(f"언어: <b>{lang}</b>  |  총 자막: <b>{total_blocks}개</b>{tok_str}"))
        summary = result.get('summary', '')
        if summary:
            lbl = QLabel(summary); lbl.setWordWrap(True)
            lbl.setStyleSheet('color:#cdd6f4; padding-top:4px;')
            cl.addWidget(lbl)
        lay.addWidget(card)

        # 고유명사
        lay.addWidget(_label('고유명사 / 이름', 'section'))
        nouns = result.get('proper_nouns', [])
        if nouns:
            note = _label('  ※ "내 번역" 칸이 비어 있으면 LLM 추정 번역을 사용합니다.')
            note.setStyleSheet('color:#a6adc8; font-size:12px;')
            lay.addWidget(note)
            self.noun_table = QTableWidget(len(nouns), 4)
            self.noun_table.setHorizontalHeaderLabels(['원문', '맥락/역할', 'LLM 추정', '내 번역'])
            self.noun_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.noun_table.verticalHeader().setVisible(False)
            self.noun_table.setMinimumHeight(min(len(nouns) * 36 + 36, 260))
            self.noun_table.setAlternatingRowColors(True)
            for r, noun in enumerate(nouns):
                self.noun_table.setRowHeight(r, 36)
                for c, val in enumerate([noun.get('original',''), noun.get('context',''), noun.get('suggested','')]):
                    item = QTableWidgetItem(val)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.noun_table.setItem(r, c, item)
                le = QLineEdit()
                le.setPlaceholderText('직접 입력...')
                le.setStyleSheet(
                    'background:#1e1e2e; border:1px solid #89b4fa;'
                    'border-radius:3px; padding:2px 8px; color:#cdd6f4; margin:3px;')
                self.noun_table.setCellWidget(r, 3, le)
            lay.addWidget(self.noun_table)
        else:
            self.noun_table = QTableWidget(0, 4)
            lay.addWidget(_label('감지된 고유명사 없음', 'warn'))

        # OCR 레이블
        lay.addWidget(_label('OCR 레이블 후보', 'section'))
        ocr_items = result.get('ocr_labels', [])
        if ocr_items:
            note2 = _label('  ※ 체크된 항목은 번역 시 텍스트에서 제거됩니다.')
            note2.setStyleSheet('color:#a6adc8; font-size:12px;')
            lay.addWidget(note2)
            for item in ocr_items:
                cb = QCheckBox(f"{item.get('text','')}   —   {item.get('reason','')}")
                cb.setChecked(True)
                cb._label_text = item.get('text', '')
                lay.addWidget(cb)
                self.ocr_checks.append(cb)
        else:
            lay.addWidget(_label('감지된 OCR 레이블 없음'))

        # 추가 지시사항
        lay.addWidget(_label('추가 번역 지시사항 (선택)', 'section'))
        self.instr_edit = QTextEdit()
        self.instr_edit.setPlaceholderText('예) 주인공 남자의 말투를 거칠고 명령적으로 번역해주세요.')
        self.instr_edit.setFixedHeight(80)
        lay.addWidget(self.instr_edit)
        plan_row = QHBoxLayout()
        self.plan_btn = _btn('LLM에게 계획 확인', 'secondary')
        plan_row.addWidget(self.plan_btn)
        plan_row.addStretch()
        lay.addLayout(plan_row)
        self.plan_label = QLabel('')
        self.plan_label.setWordWrap(True)
        self.plan_label.setStyleSheet(
            'background:#313244; border:1px solid #45475a; border-radius:5px;'
            'padding:8px; color:#a6e3a1;')
        self.plan_label.hide()
        lay.addWidget(self.plan_label)
        lay.addStretch()
        self.plan_btn.clicked.connect(self._ask_plan)

    def _ask_plan(self):
        instr = self.instr_edit.toPlainText().strip()
        if not instr:
            self.plan_label.setText('지시사항을 입력해주세요.'); self.plan_label.show(); return
        self.plan_btn.setEnabled(False)
        self.plan_label.setText('계획 생성 중...'); self.plan_label.show()
        self._plan_worker = PlanWorker(self.main.translator, instr)
        self._plan_worker.done.connect(self._plan_done)
        self._plan_worker.error.connect(lambda e: self.plan_label.setText(f'오류: {e}'))
        self._plan_worker.start()

    def _plan_done(self, text):
        self.plan_label.setText(text); self.plan_btn.setEnabled(True)

    def collect_nouns(self) -> dict:
        result = {}
        for r in range(self.noun_table.rowCount()):
            orig      = (self.noun_table.item(r, 0) or QTableWidgetItem()).text()
            suggested = (self.noun_table.item(r, 2) or QTableWidgetItem()).text()
            widget    = self.noun_table.cellWidget(r, 3)
            user_val  = widget.text().strip() if isinstance(widget, QLineEdit) else ''
            final     = user_val or suggested
            if orig and final:
                result[orig] = final
        return result

    def collect_excluded(self) -> list:
        return [cb._label_text for cb in self.ocr_checks if cb.isChecked()]

    def collect_instructions(self) -> str:
        return self.instr_edit.toPlainText().strip()


class SamplePage(QWidget):
    go_back  = pyqtSignal()
    go_start = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 24, 40, 24)
        lay.addWidget(_label('번역 샘플 미리보기', 'title'))
        lay.addWidget(_label('(처음 5개 자막 블록 기준)', 'subtitle'))
        lay.addWidget(_h_sep())
        self.loading_lbl = _label('샘플 번역 생성 중...')
        self.loading_lbl.setStyleSheet('color:#a6adc8;')
        lay.addWidget(self.loading_lbl)
        self.text_edit = QTextEdit(); self.text_edit.setReadOnly(True); self.text_edit.hide()
        lay.addWidget(self.text_edit)
        lay.addStretch()
        btn_row = QHBoxLayout()
        back_btn  = _btn('← 수정하기', 'secondary')
        start_btn = _btn('이대로 번역 시작  →')
        start_btn.setFixedHeight(40)
        btn_row.addWidget(back_btn); btn_row.addStretch(); btn_row.addWidget(start_btn)
        lay.addLayout(btn_row)
        back_btn.clicked.connect(self.go_back)
        start_btn.clicked.connect(self.go_start)

    def set_loading(self, loading: bool):
        self.loading_lbl.setVisible(loading); self.text_edit.setVisible(not loading)

    def set_text(self, text: str):
        self.text_edit.setPlainText(text); self.set_loading(False)


class TranslatingPage(QWidget):
    cancel_sig = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._batch_size = DEFAULT_BATCH_SIZE
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 24, 40, 24)
        lay.setSpacing(10)
        lay.addWidget(_label('번역 중...', 'title'))
        self.status_lbl = _label('준비 중...')
        self.status_lbl.setStyleSheet('color:#a6adc8;')
        lay.addWidget(self.status_lbl)
        self.bar = QProgressBar(); self.bar.setMinimumHeight(20)
        lay.addWidget(self.bar)

        tok_card = QFrame(); tok_card.setObjectName('card')
        tl = QHBoxLayout(tok_card); tl.setContentsMargins(12, 8, 12, 8)
        self.tok_batch_lbl = QLabel('현재 배치  IN — / OUT —')
        self.tok_cum_lbl   = QLabel('누적 합계  IN — / OUT —')
        self.tok_batch_lbl.setStyleSheet('color:#89dceb; font-size:12px; font-family:monospace;')
        self.tok_cum_lbl.setStyleSheet('color:#a6e3a1; font-size:12px; font-family:monospace;')
        tl.addWidget(self.tok_batch_lbl); tl.addSpacing(30); tl.addWidget(self.tok_cum_lbl); tl.addStretch()
        lay.addWidget(tok_card)

        lay.addWidget(_label('최근 완료 배치 미리보기:', 'section'))
        self.preview = QTextEdit(); self.preview.setReadOnly(True)
        lay.addWidget(self.preview)
        lay.addStretch()
        cancel_btn = _btn('취소', 'danger')
        lay.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)
        cancel_btn.clicked.connect(self.cancel_sig)

    def reset(self, total: int, batch_size: int = DEFAULT_BATCH_SIZE,
              label: str = '번역 시작 중...'):
        self._batch_size = batch_size
        self.bar.setMaximum(total); self.bar.setValue(0)
        self.preview.clear(); self.status_lbl.setText(label)
        self.tok_batch_lbl.setText('현재 배치  IN — / OUT —')
        self.tok_cum_lbl.setText('누적 합계  IN — / OUT —')

    def update_progress(self, done: int, total: int):
        self.bar.setValue(done)
        bs = self._batch_size
        bd = (done + bs - 1) // bs
        bt = (total + bs - 1) // bs
        self.status_lbl.setText(f'배치 {bd} / {bt}  ({done} / {total} 자막)')

    def append_preview(self, batch_n, preview, b_in, b_out, cum_in, cum_out):
        self.tok_batch_lbl.setText(f'현재 배치  IN {b_in:,} / OUT {b_out:,} tokens')
        self.tok_cum_lbl.setText(f'누적 합계  IN {cum_in:,} / OUT {cum_out:,} tokens')
        if not preview: return
        self.preview.append(f'\n── 배치 {batch_n} 완료  (IN {b_in:,} / OUT {b_out:,}) ──')
        for orig, trans in preview:
            self.preview.append(f'원: {orig}'); self.preview.append(f'번: {trans}')


class ReviewPage(QWidget):
    """번역 완료 후 교정 테이블."""
    save_sig = pyqtSignal()

    def __init__(self, main_win):
        super().__init__()
        self.main          = main_win
        self._block_indices: list = []   # table_row → original blocks list index
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Header
        hdr = QWidget()
        hl  = QVBoxLayout(hdr)
        hl.setContentsMargins(40, 16, 40, 12)
        hl.addWidget(_label('번역 검토 / 교정', 'title'))
        hint = _label('번역 열을 더블클릭하면 직접 수정할 수 있습니다. 수정 후 저장하세요.')
        hint.setStyleSheet('color:#a6adc8; font-size:12px;')
        hl.addWidget(hint)
        lay.addWidget(hdr)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['#', '시작 시간', '원문', '번역 (수정 가능)'])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 165)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(38)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table)

        # Bottom bar
        bar = QWidget()
        bar.setStyleSheet('background:#313244; border-top:1px solid #45475a;')
        bl  = QHBoxLayout(bar)
        bl.setContentsMargins(40, 12, 40, 12)
        self.count_lbl = _label('')
        self.count_lbl.setStyleSheet('color:#a6adc8; font-size:12px;')
        save_btn = _btn('  저장하기  →', 'green')
        save_btn.setFixedHeight(40)
        bl.addWidget(self.count_lbl)
        bl.addStretch()
        bl.addWidget(save_btn)
        lay.addWidget(bar)

        save_btn.clicked.connect(self.save_sig)

    def populate(self, blocks, translated):
        """blocks와 translated 리스트로 테이블 채우기. 빈 번역은 제외."""
        self._block_indices.clear()
        rows = [(i, b, t) for i, (b, t) in enumerate(zip(blocks, translated)) if t]
        self.table.setRowCount(len(rows))
        for row, (orig_idx, block, trans) in enumerate(rows):
            self._block_indices.append(orig_idx)
            vals = [str(orig_idx + 1), block.start, block.text.replace('\n', ' / '), trans]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col < 3:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setForeground(item.foreground())  # keep default
                self.table.setItem(row, col, item)
        excluded = len(translated) - len(rows)
        self.count_lbl.setText(
            f'총 {len(rows)}개 자막  '
            + (f'(OCR 레이블 {excluded}개 제외)' if excluded else ''))

    def collect(self, original_translated: list) -> list:
        """테이블에서 수정된 번역을 읽어 원본 길이의 리스트로 반환."""
        result = list(original_translated)
        for row, orig_idx in enumerate(self._block_indices):
            item = self.table.item(row, 3)
            if item:
                result[orig_idx] = item.text()
        return result


class DonePage(QWidget):
    restart = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(60, 60, 60, 60)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_lbl = _label('완료!', 'title')
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title_lbl)
        self.info_lbl = QLabel('')
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet('color:#a6e3a1; font-size:14px; margin:16px 0;')
        self.info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.info_lbl)
        self.tok_lbl = QLabel('')
        self.tok_lbl.setStyleSheet('color:#89dceb; font-size:12px; font-family:monospace; margin:4px 0;')
        self.tok_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.tok_lbl)
        self.path_lbl = QLabel('')
        self.path_lbl.setWordWrap(True)
        self.path_lbl.setStyleSheet('color:#a6adc8; font-size:12px;')
        self.path_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.path_lbl)
        lay.addSpacing(30)
        again_btn = _btn('다른 파일 번역하기')
        lay.addWidget(again_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        again_btn.clicked.connect(self.restart)

    def set_info(self, path: str, count: int, excluded: int = 0,
                 tok_in: int = 0, tok_out: int = 0, mode: str = 'translate'):
        if mode == 'cleanup':
            self.title_lbl.setText('정리 완료!')
            self.info_lbl.setText(f'{count}개 자막 블록 삭제 완료')
        else:
            self.title_lbl.setText('번역 완료!')
            self.info_lbl.setText(
                f'총 {count}개 자막 번역 완료'
                + (f'  (OCR 레이블 {excluded}개 제외)' if excluded else ''))
        if tok_in or tok_out:
            self.tok_lbl.setText(
                f'사용 토큰  입력: {tok_in:,} / 출력: {tok_out:,}  '
                f'(합계: {tok_in + tok_out:,})')
        else:
            self.tok_lbl.setText('')
        self.path_lbl.setText(f'저장 위치:\n{path}')


class CleanupReviewPage(QWidget):
    """자막 정리: LLM이 감지한 노이즈 패턴 확인 후 삭제."""
    apply_sig  = pyqtSignal()
    cancel_sig = pyqtSignal()

    def __init__(self, main_win):
        super().__init__()
        self.main = main_win
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable content
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(40, 24, 40, 24)
        cl.setSpacing(12)

        cl.addWidget(_label('자막 정리 검토', 'title'))
        note = _label('체크된 패턴을 자막 파일에서 삭제합니다. 확인 후 저장하세요.')
        note.setStyleSheet('color:#a6adc8;')
        cl.addWidget(note)

        # Summary card
        self.summary_card = QFrame(); self.summary_card.setObjectName('card')
        scl = QVBoxLayout(self.summary_card)
        self.summary_lbl = QLabel(''); self.summary_lbl.setWordWrap(True)
        self.tok_lbl = QLabel('')
        self.tok_lbl.setStyleSheet('color:#89dceb; font-size:12px; font-family:monospace;')
        scl.addWidget(self.summary_lbl); scl.addWidget(self.tok_lbl)
        cl.addWidget(self.summary_card)

        # Select all / none row
        sel_row = QHBoxLayout()
        sel_all  = _btn('전체 선택', 'secondary')
        sel_none = _btn('전체 해제', 'secondary')
        sel_all.setFixedHeight(30); sel_none.setFixedHeight(30)
        sel_all.clicked.connect(lambda: self._set_all(True))
        sel_none.clicked.connect(lambda: self._set_all(False))
        sel_row.addWidget(sel_all); sel_row.addWidget(sel_none); sel_row.addStretch()
        cl.addLayout(sel_row)

        # Patterns table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['삭제', '텍스트', '이유', '횟수'])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(3, 60)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.setAlternatingRowColors(True)
        cl.addWidget(self.table)
        cl.addStretch()
        outer.addWidget(_scroll_wrap(content))

        # Bottom bar
        bar = QWidget()
        bar.setStyleSheet('background:#313244; border-top:1px solid #45475a;')
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(40, 12, 40, 12)
        cancel_btn = _btn('취소', 'secondary')
        apply_btn  = _btn('  선택 항목 삭제 후 저장  →', 'danger')
        apply_btn.setFixedHeight(40)
        bl.addWidget(cancel_btn); bl.addStretch(); bl.addWidget(apply_btn)
        outer.addWidget(bar)

        cancel_btn.clicked.connect(self.cancel_sig)
        apply_btn.clicked.connect(self.apply_sig)

    def populate(self, result: dict):
        patterns = result.get('patterns', [])
        summary  = result.get('summary', '')
        tok_in   = result.get('_tok_in', 0)
        tok_out  = result.get('_tok_out', 0)

        self.summary_lbl.setText(summary or '분석 완료')
        if tok_in:
            self.tok_lbl.setText(f'분석 토큰  IN {tok_in:,} / OUT {tok_out:,}')

        self.table.setRowCount(len(patterns))
        for r, p in enumerate(patterns):
            # checkbox column
            cb_widget = QWidget()
            cb_lay = QHBoxLayout(cb_widget)
            cb_lay.setContentsMargins(8, 0, 8, 0)
            cb_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cb = QCheckBox(); cb.setChecked(True)
            cb_lay.addWidget(cb)
            self.table.setCellWidget(r, 0, cb_widget)
            # text, reason, count
            for c, val in enumerate([p.get('text',''), p.get('reason',''), str(p.get('count',''))]):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c + 1, item)

    def _set_all(self, checked: bool):
        for r in range(self.table.rowCount()):
            w = self.table.cellWidget(r, 0)
            if w:
                for child in w.findChildren(QCheckBox):
                    child.setChecked(checked)

    def collect_selected(self) -> list:
        selected = []
        for r in range(self.table.rowCount()):
            w = self.table.cellWidget(r, 0)
            if w:
                checks = w.findChildren(QCheckBox)
                if checks and checks[0].isChecked():
                    item = self.table.item(r, 1)
                    if item:
                        selected.append(item.text())
        return selected


class MissingPage(QWidget):
    """두 디렉토리를 지정해 .ko.srt 가 없는 파일 목록을 표시."""

    # 알려진 언어 접미사 (확장자 제외)
    _LANG_TAGS = {'.ko', '.zh', '.en', '.ja', '.cn', '.chs', '.cht',
                  '.sc', '.tc', '.chi', '.jpn', '.eng'}

    def __init__(self, main_win):
        super().__init__()
        self.main = main_win
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 24, 40, 24)
        lay.setSpacing(14)

        lay.addWidget(_label('미번역 목록', 'title'))
        sub = _label('.ko.srt 파일이 없는 자막을 찾습니다.')
        sub.setStyleSheet('color:#a6adc8;')
        lay.addWidget(sub)
        lay.addWidget(_h_sep())

        # Dir 1 – source
        lay.addWidget(_label('원본 SRT 디렉토리 (필수)'))
        d1_row = QHBoxLayout()
        self.dir1_lbl = _label('선택 안 됨')
        self.dir1_lbl.setStyleSheet(
            'color:#a6adc8; background:#313244; border:1px solid #45475a;'
            'border-radius:5px; padding:6px 10px;')
        self.dir1_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        d1_btn = _btn('선택', 'secondary')
        d1_row.addWidget(self.dir1_lbl); d1_row.addWidget(d1_btn)
        lay.addLayout(d1_row)

        # Dir 2 – translation (optional)
        lay.addWidget(_label('번역본 디렉토리 (비워두면 원본과 동일)'))
        d2_row = QHBoxLayout()
        self.dir2_lbl = _label('선택 안 됨 (원본과 같은 디렉토리)')
        self.dir2_lbl.setStyleSheet(
            'color:#a6adc8; background:#313244; border:1px solid #45475a;'
            'border-radius:5px; padding:6px 10px;')
        self.dir2_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        d2_btn   = _btn('선택', 'secondary')
        d2_clear = _btn('초기화', 'secondary')
        d2_row.addWidget(self.dir2_lbl)
        d2_row.addWidget(d2_btn)
        d2_row.addWidget(d2_clear)
        lay.addLayout(d2_row)

        # Options
        opt_row = QHBoxLayout()
        self.recursive_cb = QCheckBox('하위 폴더 포함')
        self.recursive_cb.setChecked(True)
        opt_row.addWidget(self.recursive_cb); opt_row.addStretch()
        lay.addLayout(opt_row)

        # Search button
        search_row = QHBoxLayout()
        self.search_btn = _btn('검색')
        self.search_btn.setFixedHeight(40)
        search_row.addStretch(); search_row.addWidget(self.search_btn)
        lay.addLayout(search_row)

        # Result
        self.result_lbl = _label('')
        self.result_lbl.setStyleSheet('color:#a6adc8; font-size:12px;')
        lay.addWidget(self.result_lbl)

        self.list_edit = QTextEdit()
        self.list_edit.setReadOnly(True)
        self.list_edit.setPlaceholderText('검색 결과가 여기에 표시됩니다.')
        lay.addWidget(self.list_edit)

        # Back
        back_btn = _btn('← 돌아가기', 'secondary')
        lay.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # State
        self._dir1 = ''
        self._dir2 = ''

        # Signals
        d1_btn.clicked.connect(self._pick_dir1)
        d2_btn.clicked.connect(self._pick_dir2)
        d2_clear.clicked.connect(self._clear_dir2)
        self.search_btn.clicked.connect(self._search)
        back_btn.clicked.connect(
            lambda: self.main.stack.setCurrentIndex(P_SETUP))

    def _pick_dir1(self):
        d = QFileDialog.getExistingDirectory(self, '원본 디렉토리 선택')
        if d:
            self._dir1 = d
            self.dir1_lbl.setText(d)

    def _pick_dir2(self):
        d = QFileDialog.getExistingDirectory(self, '번역본 디렉토리 선택')
        if d:
            self._dir2 = d
            self.dir2_lbl.setText(d)

    def _clear_dir2(self):
        self._dir2 = ''
        self.dir2_lbl.setText('선택 안 됨 (원본과 같은 디렉토리)')

    _VIDEO_EXTS = {'.mkv', '.mp4', '.avi', '.mov', '.wmv',
                   '.ts', '.m2ts', '.flv', '.webm', '.mpg', '.mpeg'}

    def _search(self):
        if not self._dir1:
            QMessageBox.warning(self, '알림', '원본 디렉토리를 선택하세요.')
            return

        p1 = Path(self._dir1)
        p2 = Path(self._dir2) if self._dir2 else None
        recursive = self.recursive_cb.isChecked()
        pattern   = '**/*' if recursive else '*'

        # ── 1) SRT 파일 수집 ──────────────────────────────────────────────────
        all_srt = sorted(
            f for f in p1.glob('**/*.srt' if recursive else '*.srt')
        )

        # .ko.srt 가 없는 원본 SRT
        missing_ko = []
        for srt in all_srt:
            if srt.stem.lower().endswith('.ko'):
                continue
            base      = self._base_name(srt.stem)
            rel       = srt.relative_to(p1)
            same_dir  = srt.parent / f"{base}.ko.srt"
            if p2:
                trans_rel  = p2 / rel.parent / f"{base}.ko.srt"
                trans_flat = p2 / f"{base}.ko.srt"
                found = same_dir.exists() or trans_rel.exists() or trans_flat.exists()
            else:
                found = same_dir.exists()
            if not found:
                missing_ko.append(str(srt))

        # ── 2) 자막 자체가 없는 동영상 파일 ─────────────────────────────────
        # Dir 1 전체 파일에서 비디오 확장자 탐색
        all_videos = sorted(
            f for f in p1.glob(pattern)
            if f.is_file() and f.suffix.lower() in self._VIDEO_EXTS
        )
        no_sub = []
        for vid in all_videos:
            base     = self._base_name(vid.stem)
            vid_dir  = vid.parent

            # 같은 폴더 안에 어떤 .srt 든 존재하면 OK
            def srt_exists_local():
                for tag in ('', '.zh', '.en', '.ja', '.cn', '.chs', '.cht',
                            '.ko', '.sc', '.tc', '.chi', '.jpn', '.eng'):
                    if (vid_dir / f"{base}{tag}.srt").exists():
                        return True
                return False

            def srt_exists_dir2():
                if not p2:
                    return False
                rel_parent = vid.relative_to(p1).parent
                for tag in ('', '.zh', '.en', '.ja', '.cn', '.chs', '.cht',
                            '.ko', '.sc', '.tc', '.chi', '.jpn', '.eng'):
                    if (p2 / rel_parent / f"{base}{tag}.srt").exists():
                        return True
                    if (p2 / f"{base}{tag}.srt").exists():
                        return True
                return False

            if not srt_exists_local() and not srt_exists_dir2():
                no_sub.append(str(vid))

        # ── 결과 표시 ─────────────────────────────────────────────────────────
        self.list_edit.clear()
        parts = []
        if no_sub:
            parts.append(
                f'■ 자막 없음 ({len(no_sub)}개) — 동영상에 .srt 파일 없음\n'
                + '\n'.join(no_sub)
            )
        if missing_ko:
            parts.append(
                f'■ 미번역 ({len(missing_ko)}개) — .ko.srt 없음\n'
                + '\n'.join(missing_ko)
            )

        srt_src_count = sum(1 for s in all_srt
                            if not s.stem.lower().endswith('.ko'))
        self.result_lbl.setText(
            f'자막 없음: {len(no_sub)}개  /  미번역: {len(missing_ko)}개'
            f'  /  동영상 스캔: {len(all_videos)}개  /  SRT 스캔: {srt_src_count}개'
        )
        if parts:
            self.list_edit.setPlainText('\n\n'.join(parts))
        else:
            self.list_edit.setPlainText('(모든 동영상에 자막·번역본이 존재합니다)')

    @staticmethod
    def _base_name(stem: str) -> str:
        """stem 에서 언어 태그 제거 후 기본 이름 반환."""
        known = {'.ko', '.zh', '.en', '.ja', '.cn', '.chs', '.cht',
                 '.sc', '.tc', '.chi', '.jpn', '.eng'}
        low = stem.lower()
        for tag in known:
            if low.endswith(tag):
                return stem[: len(stem) - len(tag)]
        return stem


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings      = QSettings('srt_trans', 'app')
        self.srt_path      = ''
        self.srt_blocks    = []
        self.translator    = None
        self._worker       = None
        self._trans_results: list = []
        self._ko_path      = ''
        self._cum_tok_in   = 0
        self._cum_tok_out  = 0
        self._missing_original_indices: list = []

        self.setWindowTitle('SRT 한국어 번역기')
        self.setMinimumSize(960, 660)
        self.resize(1050, 720)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.setup_page    = SetupPage(self)
        self.loading_page  = LoadingPage()
        self.analysis_page = AnalysisPage(self)
        self.sample_page   = SamplePage()
        self.trans_page    = TranslatingPage()
        self.review_page   = ReviewPage(self)
        self.done_page     = DonePage()
        self.cleanup_page  = CleanupReviewPage(self)
        self.missing_page  = MissingPage(self)

        for p in [self.setup_page, self.loading_page, self.analysis_page,
                  self.sample_page, self.trans_page, self.review_page,
                  self.done_page, self.cleanup_page, self.missing_page]:
            self.stack.addWidget(p)
        # indices: 0=setup 1=loading 2=analysis 3=sample 4=trans 5=review 6=done 7=cleanup 8=missing

        self.analysis_page.request_sample.connect(self._show_sample)
        self.analysis_page.request_start.connect(self._begin_translation)
        self.sample_page.go_back.connect(lambda: self.stack.setCurrentIndex(P_ANALYSIS))
        self.sample_page.go_start.connect(self._begin_translation)
        self.trans_page.cancel_sig.connect(self._cancel_translation)
        self.review_page.save_sig.connect(self._save_review)
        self.done_page.restart.connect(self._restart)
        self.cleanup_page.apply_sig.connect(self._apply_cleanup)
        self.cleanup_page.cancel_sig.connect(lambda: self.stack.setCurrentIndex(P_SETUP))

    # ── Translation flow ─────────────────────────────────────────────────────

    def start_analysis(self):
        token = self.setup_page.token
        if not token or not self.srt_path:
            return
        try:
            self.srt_blocks = parse_srt(self.srt_path)
        except Exception as e:
            QMessageBox.critical(self, '오류', f'SRT 파싱 실패:\n{e}'); return
        if not self.srt_blocks:
            QMessageBox.warning(self, '경고', '자막 블록을 찾을 수 없습니다.'); return

        self.translator = SRTTranslator(LLMClient(token))
        self.loading_page.start('LLM이 자막을 분석 중입니다')
        self.stack.setCurrentIndex(P_LOADING)

        self._worker = AnalysisWorker(self.translator, self.srt_blocks)
        self._worker.done.connect(self._analysis_done)
        self._worker.error.connect(self._analysis_error)
        self._worker.start()

    def _analysis_done(self, result: dict):
        self.loading_page.stop()
        self.analysis_page.populate(result, len(self.srt_blocks))
        self.stack.setCurrentIndex(P_ANALYSIS)

    def _analysis_error(self, err: str):
        self.loading_page.stop()
        reply = QMessageBox.question(
            self, '분석 실패',
            f'분석 오류:\n{err}\n\n기본 설정으로 계속하시겠습니까?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.analysis_page.populate({}, len(self.srt_blocks))
            self.stack.setCurrentIndex(P_ANALYSIS)
        else:
            self.stack.setCurrentIndex(P_SETUP)

    def _collect_settings(self):
        self.translator.proper_nouns       = self.analysis_page.collect_nouns()
        self.translator.excluded_labels    = self.analysis_page.collect_excluded()
        self.translator.extra_instructions = self.analysis_page.collect_instructions()

    def _show_sample(self):
        self._collect_settings()
        self.sample_page.set_loading(True)
        self.stack.setCurrentIndex(P_SAMPLE)
        self._worker = SampleWorker(self.translator, self.srt_blocks)
        self._worker.done.connect(self.sample_page.set_text)
        self._worker.error.connect(lambda e: self.sample_page.set_text(f'샘플 실패:\n{e}'))
        self._worker.start()

    def _begin_translation(self):
        self._collect_settings()
        total      = len(self.srt_blocks)
        batch_size = self.setup_page.batch_size
        self._cum_tok_in  = 0
        self._cum_tok_out = 0

        lang_code = getattr(self.translator, 'source_lang_code', 'zh')
        _, self._ko_path = get_output_paths(self.srt_path, lang_code)

        self.trans_page.reset(total, batch_size)
        self.stack.setCurrentIndex(P_TRANS)

        self._worker = TranslationWorker(self.translator, self.srt_blocks, batch_size)
        self._worker.progress.connect(self.trans_page.update_progress)
        self._worker.batch_done.connect(self._on_batch_done)
        self._worker.partial_save.connect(self._auto_save)
        self._worker.done.connect(self._translation_done)
        self._worker.error.connect(self._translation_error)
        self._worker.start()

    def _on_batch_done(self, batch_n, preview, b_in, b_out, cum_in, cum_out):
        self._cum_tok_in  = cum_in
        self._cum_tok_out = cum_out
        self.trans_page.append_preview(batch_n, preview, b_in, b_out, cum_in, cum_out)

    def _auto_save(self, partial: list):
        if not self._ko_path or not self.srt_blocks:
            return
        try:
            write_ko_srt(self.srt_blocks, partial, self._ko_path)
        except Exception:
            pass

    def _cancel_translation(self):
        if self._worker: self._worker.cancel()
        self.stack.setCurrentIndex(P_SETUP)

    def _translation_done(self, results: list):
        self._trans_results = results
        # 최종 저장
        lang_code = getattr(self.translator, 'source_lang_code', 'zh')
        ko_path   = self._ko_path or get_output_paths(self.srt_path, lang_code)[1]
        src_out, _ = get_output_paths(self.srt_path, lang_code)
        if Path(src_out).resolve() != Path(self.srt_path).resolve():
            try: shutil.copy2(self.srt_path, src_out)
            except Exception: pass
        try:
            write_ko_srt(self.srt_blocks, results, ko_path)
        except Exception as e:
            QMessageBox.critical(self, '저장 실패', f'파일 저장 실패:\n{e}')
            self.stack.setCurrentIndex(P_SETUP); return

        self._validate_and_proceed()

    def _validate_and_proceed(self):
        """번역 수량 검증 후 누락이 있으면 재번역 여부 물음. 통과하면 교정 테이블로."""
        results = self._trans_results
        blocks  = self.srt_blocks

        missing_indices = [
            i for i, (b, r) in enumerate(zip(blocks, results))
            if self.translator.is_translatable(b) and not r
        ]

        if missing_indices:
            total      = len(blocks)
            translated = sum(1 for r in results if r)
            excluded   = sum(1 for b in blocks
                             if not self.translator.is_translatable(b))
            msg = (
                f"번역 누락 {len(missing_indices)}개가 발견되었습니다.\n\n"
                f"  전체 블록:   {total}개\n"
                f"  번역 완료:   {translated}개\n"
                f"  OCR 제외:    {excluded}개\n"
                f"  누락:        {len(missing_indices)}개\n\n"
                f"배치 크기가 너무 크면 LLM 출력이 잘려 누락이 생길 수 있습니다.\n"
                f"누락된 블록만 재번역하시겠습니까?"
            )
            reply = QMessageBox.question(
                self, '번역 검증', msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._retranslate_missing(missing_indices)
                return

        self.review_page.populate(self.srt_blocks, results)
        self.stack.setCurrentIndex(P_REVIEW)

    def _retranslate_missing(self, missing_indices: list):
        """누락 인덱스 블록만 골라 재번역 워커 시작."""
        self._missing_original_indices = missing_indices
        missing_blocks = [self.srt_blocks[i] for i in missing_indices]
        batch_size     = self.setup_page.batch_size

        self.trans_page.reset(
            len(missing_blocks), batch_size,
            f'누락 {len(missing_blocks)}개 재번역 중...',
        )
        self.stack.setCurrentIndex(P_TRANS)

        self._worker = TranslationWorker(self.translator, missing_blocks, batch_size)
        self._worker.progress.connect(self.trans_page.update_progress)
        self._worker.batch_done.connect(self._on_batch_done)
        self._worker.done.connect(self._retrans_done)
        self._worker.error.connect(self._translation_error)
        self._worker.start()

    def _retrans_done(self, new_results: list):
        """재번역 결과를 기존 결과에 병합 후 재저장 → 재검증."""
        for orig_idx, trans in zip(self._missing_original_indices, new_results):
            if trans:
                self._trans_results[orig_idx] = trans

        lang_code = getattr(self.translator, 'source_lang_code', 'zh')
        ko_path   = self._ko_path or get_output_paths(self.srt_path, lang_code)[1]
        try:
            write_ko_srt(self.srt_blocks, self._trans_results, ko_path)
        except Exception as e:
            QMessageBox.critical(self, '저장 실패', f'파일 저장 실패:\n{e}'); return

        self._validate_and_proceed()

    def _translation_error(self, err: str):
        QMessageBox.critical(self, '번역 오류', f'번역 중 오류:\n{err}')
        self.stack.setCurrentIndex(P_SETUP)

    def _save_review(self):
        """교정 테이블에서 최종 번역 수집 후 파일 저장 → 완료 화면."""
        updated = self.review_page.collect(self._trans_results)
        lang_code = getattr(self.translator, 'source_lang_code', 'zh')
        ko_path   = self._ko_path or get_output_paths(self.srt_path, lang_code)[1]
        try:
            write_ko_srt(self.srt_blocks, updated, ko_path)
        except Exception as e:
            QMessageBox.critical(self, '저장 실패', f'파일 저장 실패:\n{e}'); return
        excluded_count   = sum(1 for r in updated if r == '')
        translated_count = len(updated) - excluded_count
        self.done_page.set_info(
            ko_path, translated_count, excluded_count,
            self._cum_tok_in, self._cum_tok_out, mode='translate')
        self.stack.setCurrentIndex(P_DONE)

    # ── Cleanup flow ─────────────────────────────────────────────────────────

    def start_cleanup(self):
        token = self.setup_page.token
        if not token or not self.srt_path:
            return
        try:
            self.srt_blocks = parse_srt(self.srt_path)
        except Exception as e:
            QMessageBox.critical(self, '오류', f'SRT 파싱 실패:\n{e}'); return
        if not self.srt_blocks:
            QMessageBox.warning(self, '경고', '자막 블록을 찾을 수 없습니다.'); return

        if self.translator is None or True:
            self.translator = SRTTranslator(LLMClient(token))

        self.loading_page.start('LLM이 자막 전체를 분석 중입니다')
        self.stack.setCurrentIndex(P_LOADING)

        self._worker = CleanupWorker(self.translator, self.srt_blocks)
        self._worker.done.connect(self._cleanup_done)
        self._worker.error.connect(self._cleanup_error)
        self._worker.start()

    def _cleanup_done(self, result: dict):
        self.loading_page.stop()
        self.cleanup_page.populate(result)
        self.stack.setCurrentIndex(P_CLEANUP)

    def _cleanup_error(self, err: str):
        self.loading_page.stop()
        QMessageBox.critical(self, '분석 실패', f'분석 중 오류:\n{err}')
        self.stack.setCurrentIndex(P_SETUP)

    def _apply_cleanup(self):
        selected = self.cleanup_page.collect_selected()
        if not selected:
            QMessageBox.information(self, '알림', '삭제할 패턴이 선택되지 않았습니다.')
            return

        pattern_set = [p.strip() for p in selected if p.strip()]

        # 각 블록에서 패턴 제거 또는 블록 통째 삭제
        kept = []
        deleted_count = 0
        for block in self.srt_blocks:
            cleaned = block.text
            for p in pattern_set:
                cleaned = cleaned.replace(p, '')
            lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
            if lines:
                new_block = copy.copy(block)
                new_block.lines = lines
                kept.append(new_block)
            else:
                deleted_count += 1

        try:
            write_cleaned_srt(kept, self.srt_path)
        except Exception as e:
            QMessageBox.critical(self, '저장 실패', f'저장 실패:\n{e}'); return

        self.done_page.set_info(
            self.srt_path, deleted_count, mode='cleanup')
        self.stack.setCurrentIndex(P_DONE)

    # ── Common ────────────────────────────────────────────────────────────────

    def _restart(self):
        self.srt_path         = ''
        self.srt_blocks       = []
        self.translator       = None
        self._trans_results   = []
        self._ko_path         = ''
        self._cum_tok_in      = 0
        self._cum_tok_out     = 0
        self.setup_page.file_label.setText('파일을 선택하세요...')
        self.setup_page.start_btn.setEnabled(False)
        self.setup_page.cleanup_btn.setEnabled(False)
        self.stack.setCurrentIndex(P_SETUP)


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
