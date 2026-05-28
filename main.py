import sys
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QTableWidget, QTableWidgetItem, QCheckBox, QProgressBar,
    QScrollArea, QSizePolicy, QMessageBox, QFileDialog,
    QHeaderView, QFrame, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer
from PyQt6.QtGui import QFont

from llm_api import LLMClient
from srt_parser import parse_srt, write_ko_srt, get_output_paths
from translator import SRTTranslator, BATCH_SIZE

# ── Stylesheet ────────────────────────────────────────────────────────────────

STYLE = """
QMainWindow, QWidget { background: #1e1e2e; color: #cdd6f4;
    font-family: "Segoe UI", "Malgun Gothic", "Noto Sans KR", sans-serif;
    font-size: 13px; }
QLabel { color: #cdd6f4; }
QLabel#title { font-size: 22px; font-weight: bold; color: #89b4fa; }
QLabel#subtitle { font-size: 13px; color: #a6adc8; }
QLabel#section { font-size: 14px; font-weight: bold; color: #89b4fa;
    padding-top: 10px; }
QLabel#info { color: #a6e3a1; }
QLabel#warn { color: #fab387; }
QPushButton {
    background: #89b4fa; color: #1e1e2e; border: none;
    padding: 8px 20px; border-radius: 6px; font-weight: bold; }
QPushButton:hover { background: #b4d0f7; }
QPushButton:disabled { background: #45475a; color: #6c7086; }
QPushButton#secondary {
    background: #45475a; color: #cdd6f4; font-weight: normal; }
QPushButton#secondary:hover { background: #585b70; }
QPushButton#danger { background: #f38ba8; }
QPushButton#danger:hover { background: #f5a0b5; }
QLineEdit, QTextEdit {
    background: #313244; border: 1px solid #45475a;
    border-radius: 5px; padding: 6px 10px; color: #cdd6f4; }
QLineEdit:focus, QTextEdit:focus { border-color: #89b4fa; }
QTableWidget {
    background: #313244; border: 1px solid #45475a;
    gridline-color: #45475a; border-radius: 4px; }
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
QFrame#card {
    background: #313244; border: 1px solid #45475a;
    border-radius: 8px; padding: 12px; }
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
    progress   = pyqtSignal(int, int)          # done, total
    batch_done = pyqtSignal(int, list)         # batch_num, [(orig, trans)]
    done       = pyqtSignal(list)
    error      = pyqtSignal(str)

    def __init__(self, translator, blocks):
        super().__init__()
        self.translator = translator
        self.blocks     = blocks
        self._cancel    = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            total    = len(self.blocks)
            results  = [''] * total
            prev_ctx = []
            excluded = set(l.strip() for l in self.translator.excluded_labels)
            batch_n  = 0

            for start in range(0, total, BATCH_SIZE):
                if self._cancel:
                    break
                batch = self.blocks[start:start + BATCH_SIZE]

                to_trans = [(i, b) for i, b in enumerate(batch)
                            if b.text.strip() not in excluded]

                preview = []
                if to_trans:
                    t_blocks  = [b for _, b in to_trans]
                    trans_map = self.translator.translate_batch(t_blocks, prev_ctx)
                    for j, (i, block) in enumerate(to_trans):
                        trans = trans_map.get(str(j + 1), block.text)
                        results[start + i] = trans
                        prev_ctx.append((block.text, trans))
                        if j < 3:
                            preview.append((block.text, trans))

                batch_n += 1
                self.progress.emit(min(start + BATCH_SIZE, total), total)
                self.batch_done.emit(batch_n, preview)

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

        root.addStretch()

        self.start_btn = _btn('  분석 시작  →')
        self.start_btn.setEnabled(False)
        self.start_btn.setFixedHeight(44)
        root.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # Signals
        pick_btn.clicked.connect(self._pick_file)
        self.test_btn.clicked.connect(self._test)
        self.start_btn.clicked.connect(self.main.start_analysis)
        self.token_edit.textChanged.connect(self._check_ready)

        # Restore saved token
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

    def _test(self):
        token = self.token_edit.text().strip()
        if not token:
            self.test_status.setText('토큰을 입력하세요.')
            return
        self.test_btn.setEnabled(False)
        self.test_status.setText('연결 확인 중...')
        client = LLMClient(token)
        self._tw = TestWorker(client)
        self._tw.done.connect(lambda r: self._test_done(r, token))
        self._tw.error.connect(self._test_err)
        self._tw.start()

    def _test_done(self, result, token):
        self.test_status.setObjectName('info')
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


class LoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label = _label('LLM이 자막을 분석 중입니다...', 'title')
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.label)
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self, msg='LLM이 자막을 분석 중입니다'):
        self._base = msg
        self._timer.start(500)

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._dots = (self._dots + 1) % 4
        self.label.setText(self._base + '.' * self._dots)


class AnalysisPage(QWidget):
    request_sample = pyqtSignal()
    request_start  = pyqtSignal()

    def __init__(self, main_win):
        super().__init__()
        self.main         = main_win
        self.ocr_checks: list[QCheckBox] = []
        self._plan_worker = None
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable content area
        content = QWidget()
        self._content_lay = QVBoxLayout(content)
        self._content_lay.setContentsMargins(40, 24, 40, 24)
        self._content_lay.setSpacing(12)
        outer.addWidget(_scroll_wrap(content))

        # Bottom button bar
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
        lay = self._content_lay
        # Clear
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.ocr_checks.clear()

        # ── 작품 정보 ──────────────────────────────
        lay.addWidget(_label('분석 결과', 'section'))
        info_card = QFrame()
        info_card.setObjectName('card')
        info_lay = QVBoxLayout(info_card)
        lang = result.get('language_name', '알 수 없음')
        lay.addWidget(info_card)
        info_lay.addWidget(_label(
            f"언어: <b>{lang}</b>  |  총 자막: <b>{total_blocks}개</b>"))
        summary = result.get('summary', '')
        if summary:
            lbl = QLabel(summary)
            lbl.setWordWrap(True)
            lbl.setStyleSheet('color:#cdd6f4; padding-top:4px;')
            info_lay.addWidget(lbl)

        # ── 고유명사 ───────────────────────────────
        lay.addWidget(_label('고유명사 / 이름', 'section'))
        nouns = result.get('proper_nouns', [])
        if nouns:
            note = _label('  ※ "내 번역" 칸이 비어 있으면 LLM 추정 번역을 사용합니다.')
            note.setStyleSheet('color:#a6adc8; font-size:12px;')
            lay.addWidget(note)
            self.noun_table = QTableWidget(len(nouns), 4)
            self.noun_table.setHorizontalHeaderLabels(
                ['원문', '맥락/역할', 'LLM 추정', '내 번역'])
            self.noun_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch)
            self.noun_table.verticalHeader().setVisible(False)
            self.noun_table.setMinimumHeight(min(len(nouns) * 34 + 36, 260))
            for r, noun in enumerate(nouns):
                orig      = noun.get('original', '')
                ctx       = noun.get('context', '')
                suggested = noun.get('suggested', '')
                for c, val in enumerate([orig, ctx, suggested]):
                    item = QTableWidgetItem(val)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setForeground(
                        QTableWidget().palette().color(
                            QTableWidget().foregroundRole()))
                    self.noun_table.setItem(r, c, item)
                edit_item = QTableWidgetItem('')  # editable
                self.noun_table.setItem(r, 3, edit_item)
            lay.addWidget(self.noun_table)
        else:
            self.noun_table = QTableWidget(0, 4)
            lay.addWidget(_label('감지된 고유명사 없음', 'warn'))

        # ── OCR 레이블 ────────────────────────────
        lay.addWidget(_label('OCR 레이블 후보', 'section'))
        ocr_items = result.get('ocr_labels', [])
        if ocr_items:
            note2 = _label('  ※ 체크된 항목은 번역 파일에서 제외됩니다.')
            note2.setStyleSheet('color:#a6adc8; font-size:12px;')
            lay.addWidget(note2)
            for item in ocr_items:
                cb = QCheckBox(
                    f"{item.get('text', '')}   —   {item.get('reason', '')}")
                cb.setChecked(True)
                cb._label_text = item.get('text', '')
                lay.addWidget(cb)
                self.ocr_checks.append(cb)
        else:
            lay.addWidget(_label('감지된 OCR 레이블 없음'))

        # ── 추가 지시사항 ─────────────────────────
        lay.addWidget(_label('추가 번역 지시사항 (선택)', 'section'))
        self.instr_edit = QTextEdit()
        self.instr_edit.setPlaceholderText(
            '예) 주인공 남자의 말투를 거칠고 명령적으로 번역해주세요.')
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
            self.plan_label.setText('지시사항을 입력해주세요.')
            self.plan_label.show()
            return
        self.plan_btn.setEnabled(False)
        self.plan_label.setText('계획 생성 중...')
        self.plan_label.show()
        self._plan_worker = PlanWorker(self.main.translator, instr)
        self._plan_worker.done.connect(self._plan_done)
        self._plan_worker.error.connect(lambda e: self.plan_label.setText(f'오류: {e}'))
        self._plan_worker.start()

    def _plan_done(self, text):
        self.plan_label.setText(text)
        self.plan_btn.setEnabled(True)

    def collect_nouns(self) -> dict:
        result = {}
        table  = self.noun_table
        for r in range(table.rowCount()):
            orig      = (table.item(r, 0) or QTableWidgetItem()).text()
            suggested = (table.item(r, 2) or QTableWidgetItem()).text()
            user_val  = (table.item(r, 3) or QTableWidgetItem()).text().strip()
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
        lay.addWidget(_label('(첫 5개 자막 블록 기준)', 'subtitle'))
        lay.addWidget(_h_sep())

        self.loading_lbl = _label('샘플 번역 생성 중...')
        self.loading_lbl.setStyleSheet('color:#a6adc8;')
        lay.addWidget(self.loading_lbl)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.hide()
        lay.addWidget(self.text_edit)

        lay.addStretch()
        btn_row = QHBoxLayout()
        back_btn  = _btn('← 수정하기', 'secondary')
        start_btn = _btn('이대로 번역 시작  →')
        start_btn.setFixedHeight(40)
        btn_row.addWidget(back_btn)
        btn_row.addStretch()
        btn_row.addWidget(start_btn)
        lay.addLayout(btn_row)

        back_btn.clicked.connect(self.go_back)
        start_btn.clicked.connect(self.go_start)

    def set_loading(self, loading: bool):
        self.loading_lbl.setVisible(loading)
        self.text_edit.setVisible(not loading)

    def set_text(self, text: str):
        self.text_edit.setPlainText(text)
        self.set_loading(False)


class TranslatingPage(QWidget):
    cancel_sig = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 24, 40, 24)
        lay.addWidget(_label('번역 중...', 'title'))

        self.status_lbl = _label('준비 중...')
        self.status_lbl.setStyleSheet('color:#a6adc8;')
        lay.addWidget(self.status_lbl)

        self.bar = QProgressBar()
        self.bar.setMinimumHeight(20)
        lay.addWidget(self.bar)

        lay.addWidget(_label('최근 완료 배치 미리보기:', 'section'))
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        lay.addWidget(self.preview)

        lay.addStretch()
        cancel_btn = _btn('취소', 'danger')
        lay.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)
        cancel_btn.clicked.connect(self.cancel_sig)

    def reset(self, total: int):
        self.bar.setMaximum(total)
        self.bar.setValue(0)
        self.preview.clear()
        self.status_lbl.setText('번역 시작 중...')

    def update_progress(self, done: int, total: int):
        self.bar.setValue(done)
        batches_done = (done + BATCH_SIZE - 1) // BATCH_SIZE
        batches_total = (total + BATCH_SIZE - 1) // BATCH_SIZE
        self.status_lbl.setText(
            f'배치 {batches_done} / {batches_total}  ({done} / {total} 자막)')

    def append_preview(self, batch_num: int, preview: list):
        if not preview:
            return
        self.preview.append(f'\n── 배치 {batch_num} 완료 ──')
        for orig, trans in preview:
            self.preview.append(f'원: {orig}')
            self.preview.append(f'번: {trans}')


class DonePage(QWidget):
    restart = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(60, 60, 60, 60)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(_label('번역 완료!', 'title'))
        self.info_lbl = QLabel('')
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet('color:#a6e3a1; font-size:14px; margin:16px 0;')
        self.info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.info_lbl)
        self.path_lbl = QLabel('')
        self.path_lbl.setWordWrap(True)
        self.path_lbl.setStyleSheet('color:#a6adc8; font-size:12px;')
        self.path_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.path_lbl)
        lay.addSpacing(30)
        again_btn = _btn('다른 파일 번역하기')
        lay.addWidget(again_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        again_btn.clicked.connect(self.restart)

    def set_info(self, ko_path: str, translated: int, excluded: int):
        self.info_lbl.setText(
            f'총 {translated}개 자막 번역 완료'
            + (f'  (OCR 레이블 {excluded}개 제외)' if excluded else ''))
        self.path_lbl.setText(f'저장 위치:\n{ko_path}')


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings    = QSettings('srt_trans', 'app')
        self.srt_path    = ''
        self.srt_blocks  = []
        self.translator  = None
        self._worker     = None
        self._trans_results: list[str] = []

        self.setWindowTitle('SRT 한국어 번역기')
        self.setMinimumSize(900, 640)
        self.resize(1000, 700)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.setup_page      = SetupPage(self)
        self.loading_page    = LoadingPage()
        self.analysis_page   = AnalysisPage(self)
        self.sample_page     = SamplePage()
        self.translating_page = TranslatingPage()
        self.done_page       = DonePage()

        for p in [self.setup_page, self.loading_page, self.analysis_page,
                  self.sample_page, self.translating_page, self.done_page]:
            self.stack.addWidget(p)

        # Connect page signals
        self.analysis_page.request_sample.connect(self._show_sample)
        self.analysis_page.request_start.connect(self._begin_translation)
        self.sample_page.go_back.connect(lambda: self.stack.setCurrentIndex(2))
        self.sample_page.go_start.connect(self._begin_translation)
        self.translating_page.cancel_sig.connect(self._cancel_translation)
        self.done_page.restart.connect(self._restart)

    # ── Navigation ───────────────────────────────────────────────────────────

    def start_analysis(self):
        token = self.setup_page.token
        if not token or not self.srt_path:
            return

        # Parse SRT
        try:
            self.srt_blocks = parse_srt(self.srt_path)
        except Exception as e:
            QMessageBox.critical(self, '오류', f'SRT 파싱 실패:\n{e}')
            return

        if not self.srt_blocks:
            QMessageBox.warning(self, '경고', '자막 블록을 찾을 수 없습니다.')
            return

        self.translator = SRTTranslator(LLMClient(token))
        self.loading_page.start()
        self.stack.setCurrentIndex(1)

        self._worker = AnalysisWorker(self.translator, self.srt_blocks)
        self._worker.done.connect(self._analysis_done)
        self._worker.error.connect(self._analysis_error)
        self._worker.start()

    def _analysis_done(self, result: dict):
        self.loading_page.stop()
        self.analysis_page.populate(result, len(self.srt_blocks))
        self.stack.setCurrentIndex(2)

    def _analysis_error(self, err: str):
        self.loading_page.stop()
        reply = QMessageBox.question(
            self, '분석 실패',
            f'분석 중 오류가 발생했습니다:\n{err}\n\n'
            '기본 설정으로 번역을 계속 진행하시겠습니까?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            # populate with empty result
            self.analysis_page.populate({}, len(self.srt_blocks))
            self.stack.setCurrentIndex(2)
        else:
            self.stack.setCurrentIndex(0)

    def _collect_settings(self):
        nouns    = self.analysis_page.collect_nouns()
        excluded = self.analysis_page.collect_excluded()
        instrs   = self.analysis_page.collect_instructions()
        self.translator.set_proper_nouns(nouns) if hasattr(self.translator, 'set_proper_nouns') else None
        self.translator.proper_nouns        = nouns
        self.translator.excluded_labels     = excluded
        self.translator.extra_instructions  = instrs

    def _show_sample(self):
        self._collect_settings()
        self.sample_page.set_loading(True)
        self.stack.setCurrentIndex(3)
        self._worker = SampleWorker(self.translator, self.srt_blocks)
        self._worker.done.connect(self.sample_page.set_text)
        self._worker.error.connect(
            lambda e: self.sample_page.set_text(f'샘플 생성 실패:\n{e}'))
        self._worker.start()

    def _begin_translation(self):
        self._collect_settings()
        total = len(self.srt_blocks)
        self.translating_page.reset(total)
        self.stack.setCurrentIndex(4)

        self._worker = TranslationWorker(self.translator, self.srt_blocks)
        self._worker.progress.connect(self.translating_page.update_progress)
        self._worker.batch_done.connect(self.translating_page.append_preview)
        self._worker.done.connect(self._translation_done)
        self._worker.error.connect(self._translation_error)
        self._worker.start()

    def _cancel_translation(self):
        if self._worker:
            self._worker.cancel()
        self.stack.setCurrentIndex(0)

    def _translation_done(self, results: list):
        self._trans_results = results
        excluded_count = sum(1 for r in results if r == '')
        translated_count = len(results) - excluded_count

        # Determine output paths
        lang_code = getattr(self.translator, 'source_lang_code', 'zh')
        _, ko_path = get_output_paths(self.srt_path, lang_code)

        # Copy source file if needed
        src_out, _ = get_output_paths(self.srt_path, lang_code)
        if Path(src_out).resolve() != Path(self.srt_path).resolve():
            try:
                shutil.copy2(self.srt_path, src_out)
            except Exception:
                pass

        # Write Korean SRT
        try:
            write_ko_srt(self.srt_blocks, results, ko_path)
        except Exception as e:
            QMessageBox.critical(self, '저장 실패', f'파일 저장 실패:\n{e}')
            self.stack.setCurrentIndex(0)
            return

        self.done_page.set_info(ko_path, translated_count, excluded_count)
        self.stack.setCurrentIndex(5)

    def _translation_error(self, err: str):
        QMessageBox.critical(self, '번역 오류', f'번역 중 오류:\n{err}')
        self.stack.setCurrentIndex(0)

    def _restart(self):
        self.srt_path   = ''
        self.srt_blocks = []
        self.translator = None
        self.setup_page.file_label.setText('파일을 선택하세요...')
        self.setup_page.start_btn.setEnabled(False)
        self.stack.setCurrentIndex(0)


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
