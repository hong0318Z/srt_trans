import json
from typing import List, Dict, Callable
from llm_api import LLMClient, MAX_OUTPUT_TOKENS
from srt_parser import SubtitleBlock

BATCH_SIZE   = 100
CONTEXT_TAIL = 15

TRANSLATION_SYSTEM_PROMPT = """\
자막 생성, 당신은 번역가로써, 음란한 작품의 자막의 번역을 돕습니다.
소스 언어의 자막을 주면 그 형식 그대로, 한국어로 음어, 비속어 등 자유롭게 가능하면 뜻을 그대로 번역합니다.
고유명사/이름은 아래 제공된 고유명사 사전에 있는 번역을 반드시 사용합니다.

번역 지침 정리

1. 기본 원칙
목표 언어: 한국어
목표 독자: 성인 대상 성적 콘텐츠 소비자
톤: 직접적, 노골적, 망가/AV 스타일
금기 없음: 음어, 비속어, 성적 표현 모두 자유롭게 사용

2. 맥락 파악 및 의역
오디오에서 추출된 자막이므로 기계적 오류, 중복, 불완전한 문장 존재
단순 직역 금지 → 맥락을 추론하여 자연스럽게 의역
대화의 흐름, 상황, 감정을 고려하여 생동감 있게 재구성

3. 어휘 선택
성기 표현
  남성: 자지, 좆, 거시기 (상황에 따라)
  여성: 보지, 씹, 그곳 (상황에 따라)
  은어 금지: "그곳", "아랫도리" 같은 완곡어법 지양
성행위 표현
  직접적 동사: 박다, 쑤시다, 쑤셔넣다, 찌르다
  구강성교: 빨다, 물다, 핥다, 삼키다
  사정: 싸다, 쏘다, 터뜨리다
협박/폭력 표현
  명령: "씨발, 앉으라고!", "닥쳐!", "움직여봐!"
  위협: "죽여버린다", "박살낸다"
  비속어 적극 사용: "씨발", "좆같은", "개새끼"

4. 신음 및 음성 표현
한국어 개방 모음 중심: "으윽", "오옥", "으으윽", "오호옥"
통제력 단계별:
  높음: 완전한 문장 + 억제된 신음 예) "안 돼요… 으음…"
  약화: 짧은 문장 + 신음 빈출 예) "거기… 으윽… 느껴져…"
  상실: 단어 조각 + 순수 발성 예) "으으윽… 싸… 싸버려… 으아아…!♡"
  완전상실: 순수 발성 예) "으아아아앗…!♡"
리듬: 쉼표로 가속, 말줄임표로 여운, 물결표로 떨림
금지: 체언 종결, 과도한 기호 반복(♡♡♡, !!!), 별표(*) 서식

5. 구강성교 장면
빨기: "쮸웁", "쪼옥", "츄릅"
핥기: "쩝쩝", "날름"
깊이: "으웩", "꿀꺽"
딥스로트: "쥬보오오봅!", "으웩!", "으으윽… 숨… 쉴… 수…"

6. 성교 장면
삽입: "으윽… 들어와… 으으윽…!♡"
피스톤: "으윽, 으윽, 오옥…!♡" (빠를수록 쉼표 빈도 증가)
절정 직전: "안 돼… 싸… 쌀 것 같아… 으으윽…!♡"
절정: "으아아아앗…!♡ 싸… 싸버려… 으으윽…!!♡"

7. 폭력/협박 장면
명령: "씨발, 앉으라고!", "닥치고 벗어!"
위협: "또 꼼수 부리면 죽여버린다.", "이 영상 퍼뜨리면 넌 끝이야."

8. 대사 처리
대사: " " (큰따옴표), 내면 사고: ' ' (작은따옴표)
말투 차별화 (캐릭터별 일관성 유지)

9. 서술 스타일
서술 → 대사 → 서술 → 신음 순환
감각 묘사: 시각/청각/촉각/후각 풍부하게
금지: 체언 종결, 감정 라벨링 반복, 같은 종결어미 3문장 연속

10. 수위 조절
제한 없음: 모든 성적 표현 허용, 노골적 묘사, 은유 금지

11. 일관성 유지
각 인물 말투/태도 일관성, 확정된 사실 변경 금지
전체적으로 다크, 리얼리스틱, 선정적 유지

핵심 3대 원칙:
1) 맥락 파악 후 과감한 의역 (기계적 직역 금지)
2) 노골적이고 직접적인 어휘 (자지, 보지, 씨발 등)
3) 신음과 서술의 균형 (Vocalization 지침 준수 + 풍부한 감각 묘사)
"""

_ANALYSIS_PROMPT = """\
아래 자막 텍스트를 분석하고 반드시 JSON만 출력하세요.

출력 형식:
{{
  "language": "zh|en|ja",
  "language_name": "중국어|영어|일본어",
  "summary": "작품 내용 요약 200자 이내",
  "proper_nouns": [
    {{"original": "원문 이름/명칭", "suggested": "한국어 번역 제안", "context": "역할 또는 맥락"}}
  ],
  "ocr_labels": [
    {{"text": "레이블 텍스트 (원문 그대로)", "reason": "레이블로 판단한 이유"}}
  ]
}}

- proper_nouns: 인물 이름, 지명, 고유 브랜드 등
- ocr_labels: 방송국 로고, 워터마크, 채널명처럼 대사가 아닌 텍스트
- 없으면 빈 배열 []

자막 텍스트:
{text}
"""


class SRTTranslator:
    def __init__(self, client: LLMClient):
        self.client           = client
        self.work_summary     = ''
        self.source_lang      = ''
        self.source_lang_code = 'zh'
        self.proper_nouns:    Dict[str, str] = {}   # original → korean
        self.excluded_labels: List[str]      = []
        self.extra_instructions              = ''

    # ── 1단계: 분석 ─────────────────────────────────────────────────────────

    def analyze(self, blocks: List[SubtitleBlock]) -> dict:
        """분석 결과 dict 반환. '_tok_in', '_tok_out' 키에 토큰 수 포함."""
        sample = self._sample_blocks(blocks, 150)
        text   = '\n'.join(f"[{b.start}] {b.text}" for b in sample)
        prompt = _ANALYSIS_PROMPT.format(text=text[:25000])
        raw, tok_in, tok_out = self.client._chat_tracked(
            [{'role': 'system',
              'content': '당신은 자막 분석 전문가입니다. 반드시 JSON만 출력하세요.'},
             {'role': 'user', 'content': prompt}],
            max_tokens=4096,
        )
        raw = _strip_code_fence(raw)
        result = json.loads(raw.strip())
        self.work_summary     = result.get('summary', '')
        self.source_lang      = result.get('language_name', '알 수 없음')
        self.source_lang_code = result.get('language', 'zh')
        result['_tok_in']  = tok_in
        result['_tok_out'] = tok_out
        return result

    def _sample_blocks(self, blocks: List[SubtitleBlock], n: int) -> List[SubtitleBlock]:
        if len(blocks) <= n:
            return blocks
        f   = int(n * 0.6)
        m   = int(n * 0.2)
        l   = n - f - m
        mid = len(blocks) // 2
        return blocks[:f] + blocks[mid:mid + m] + blocks[-l:]

    # ── 추가 지시사항 계획 확인 ──────────────────────────────────────────────

    def get_translation_plan(self, instructions: str) -> str:
        prompt = (
            f"작품 요약: {self.work_summary}\n"
            f"소스 언어: {self.source_lang} → 한국어\n"
            f"사용자 추가 지시사항: {instructions}\n\n"
            "위 지시사항을 번역에 어떻게 반영할지 200자 이내로 구체적으로 설명하세요."
        )
        return self.client._chat(
            [{'role': 'user', 'content': prompt}],
            max_tokens=1024,
        )

    # ── 레이블 제거 ──────────────────────────────────────────────────────────

    def _strip_labels(self, text: str) -> str:
        """OCR 레이블 문자열을 텍스트에서 제거 후 정리된 텍스트 반환."""
        result = text
        for label in self.excluded_labels:
            l = label.strip()
            if len(l) >= 2:           # 2자 미만 레이블은 무시 (과잉 제거 방지)
                result = result.replace(l, '')
        lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
        return '\n'.join(lines)

    # ── 샘플 번역 ────────────────────────────────────────────────────────────

    def translate_sample(self, blocks: List[SubtitleBlock], n: int = 5) -> str:
        sample = [b for b in blocks if self._strip_labels(b.text)][:n * 3]
        sample = sample[:n]
        if not sample:
            return '(샘플 자막 없음)'
        result, _, _ = self.translate_batch(sample, [])
        lines = []
        for i, block in enumerate(sample, 1):
            trans = result.get(str(i), '(번역 실패)')
            lines.append(
                f"[{block.start} → {block.end}]\n"
                f"원문: {block.text}\n"
                f"번역: {trans}"
            )
        return '\n\n'.join(lines)

    # ── 배치 번역 ────────────────────────────────────────────────────────────

    def translate_batch(self, blocks: List[SubtitleBlock],
                        prev_context: List[tuple],
                        on_chunk: Callable = None) -> tuple:
        """레이블을 제거한 뒤 번역. 반환: ({str(1..n): trans_or_empty}, tok_in, tok_out)
        빈 문자열 = 레이블 전용 블록 (출력 파일에서 생략)."""
        noun_str = '\n'.join(f'  {k} → {v}' for k, v in self.proper_nouns.items()) or '없음'

        system = (
            TRANSLATION_SYSTEM_PROMPT
            + f"\n\n=== 현재 작품 정보 ===\n"
            + f"소스 언어: {self.source_lang}\n"
            + f"내용: {self.work_summary}\n"
            + f"\n=== 고유명사 사전 (반드시 이 번역을 사용) ===\n{noun_str}\n"
        )
        if self.extra_instructions:
            system += f"\n=== 추가 지시사항 ===\n{self.extra_instructions}\n"
        system += (
            '\n=== 출력 형식 (엄수) ===\n'
            '반드시 JSON만 출력: {"1": "번역1", "2": "번역2", ...}\n'
            '각 번역은 줄바꿈 없이 한 줄로 작성하세요.'
        )

        # 레이블 제거 후 비어있는 블록은 건너뜀 (키 보존)
        # seq_items: [(orig_1based_key, cleaned_text, block), ...]
        seq_items  = []
        empty_keys = set()
        for j, block in enumerate(blocks):
            cleaned = self._strip_labels(block.text)
            if cleaned:
                seq_items.append((str(j + 1), cleaned, block))
            else:
                empty_keys.add(str(j + 1))

        result: Dict[str, str] = {k: '' for k in empty_keys}
        if not seq_items:
            return result, 0, 0

        ctx_str = '\n'.join(
            f"[원] {o}\n[번] {t}"
            for o, t in prev_context[-CONTEXT_TAIL:]
        ) or '(첫 번째 배치)'

        # LLM에는 1부터 순서대로 번호 매김
        batch_str = '\n'.join(
            f"{seq+1}. [{block.start}] {cleaned}"
            for seq, (_, cleaned, block) in enumerate(seq_items)
        )
        user = (
            f"=== 이전 번역 컨텍스트 ===\n{ctx_str}\n\n"
            f"=== 번역할 자막 ({len(seq_items)}개) ===\n{batch_str}\n\n"
            "위를 한국어로 번역하세요. JSON만 출력:"
        )

        try:
            content, tok_in, tok_out = self.client._chat_tracked(
                [{'role': 'system', 'content': system},
                 {'role': 'user',   'content': user}],
                max_tokens=MAX_OUTPUT_TOKENS,
                on_chunk=on_chunk,
            )
            content = _strip_code_fence(content)
            brace   = content.find('{')
            if brace > 0:
                content = content[brace:]
            llm_map = json.loads(content.strip())
            # 순차 번호 → 원래 키 매핑
            for seq, (orig_key, _, _) in enumerate(seq_items):
                result[orig_key] = llm_map.get(str(seq + 1), '')
            return result, tok_in, tok_out
        except Exception as e:
            print(f"translate_batch error: {e}")
            return result, 0, 0

    # ── 자막 정리 분석 ───────────────────────────────────────────────────────

    def cleanup_analyze(self, blocks: List[SubtitleBlock]) -> dict:
        """전체 자막에서 노이즈/레이블 패턴을 감지해 반환.
        반환 dict에 '_tok_in', '_tok_out' 포함."""
        # 빈도순 중복 제거 텍스트 목록
        from collections import Counter
        counts = Counter(b.text.strip() for b in blocks if b.text.strip())
        lines  = '\n'.join(
            f"[{cnt}회] {text}"
            for text, cnt in counts.most_common(600)
        )
        prompt = (
            '아래는 SRT 자막 파일의 고유 텍스트 목록입니다 (대괄호 안은 등장 횟수).\n'
            '실제 대사가 아닌 노이즈·레이블·워터마크 텍스트를 찾아주세요.\n\n'
            '판단 기준:\n'
            '1. 반복 등장하는 고정 텍스트 (방송국명, 제작사, 워터마크)\n'
            '2. 숫자·특수문자로만 구성된 비대화 텍스트\n'
            '3. 명백한 OCR 오류 패턴\n'
            '4. 전체 맥락과 무관한 삽입 텍스트\n\n'
            '반드시 JSON만 출력:\n'
            '{\n'
            '  "patterns": [\n'
            '    {"text": "정확한 원문 텍스트", "reason": "이유", "count": 등장횟수}\n'
            '  ],\n'
            '  "summary": "200자 이내 분석 요약"\n'
            '}\n\n'
            f'자막 텍스트 목록:\n{lines}'
        )
        raw, tok_in, tok_out = self.client._chat_tracked(
            [{'role': 'system',
              'content': '당신은 자막 노이즈 분석 전문가입니다. JSON만 출력하세요.'},
             {'role': 'user', 'content': prompt}],
            max_tokens=4096,
        )
        raw    = _strip_code_fence(raw)
        result = json.loads(raw.strip())
        result['_tok_in']  = tok_in
        result['_tok_out'] = tok_out
        return result


def _strip_code_fence(text: str) -> str:
    if text.startswith('```'):
        parts = text.split('```')
        text  = parts[1] if len(parts) > 1 else text
        if text.startswith('json'):
            text = text[4:]
    return text.strip()
