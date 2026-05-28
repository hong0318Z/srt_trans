"""
GitHub Copilot API 래퍼 (httpx 직접 호출)
endpoint = https://api.githubcopilot.com
api_key  = GitHub Personal Access Token (PAT)
"""

import json
import httpx

GITHUB_COPILOT_ENDPOINT = "https://api.githubcopilot.com"
DEFAULT_MODEL            = "claude-sonnet-4.5"
MAX_OUTPUT_TOKENS        = 64000

_COPILOT_HEADERS = {
    "Editor-Version":         "vscode/1.95.0",
    "Editor-Plugin-Version":  "copilot-chat/0.22.0",
    "Copilot-Integration-Id": "vscode-chat",
    "Openai-Organization":    "github-copilot",
}


class LLMClient:
    def __init__(self, token: str,
                 model: str    = DEFAULT_MODEL,
                 endpoint: str = GITHUB_COPILOT_ENDPOINT):
        self.model     = model
        self._endpoint = endpoint.rstrip('/')
        self._headers  = {
            **_COPILOT_HEADERS,
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

    def _chat(self, messages: list, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
        content, _, _ = self._chat_tracked(messages, max_tokens)
        return content

    def _chat_tracked(self, messages: list, max_tokens: int = MAX_OUTPUT_TOKENS,
                      on_chunk: callable = None) -> tuple:
        url     = f"{self._endpoint}/chat/completions"
        payload = {
            "model":      self.model,
            "messages":   messages,
            "max_tokens": max_tokens,
            "stream":     True,
        }
        with httpx.Client(timeout=httpx.Timeout(connect=30, read=120,
                                                write=30, pool=10)) as client:
            chunks  = []
            tok_in  = 0
            tok_out = 0
            with client.stream('POST', url, json=payload,
                               headers=self._headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if 'usage' in obj:
                        u = obj['usage']
                        tok_in  = u.get('prompt_tokens', tok_in)
                        tok_out = u.get('completion_tokens', tok_out)
                    delta = (obj.get('choices') or [{}])[0].get('delta', {})
                    piece = delta.get('content') or ''
                    if piece:
                        chunks.append(piece)
                        tok_out += 1
                        if on_chunk:
                            on_chunk(piece)
            return ''.join(chunks).strip(), tok_in, tok_out

    def test_connection(self) -> str:
        return self._chat(
            [{"role": "user", "content": "2+2는 뭔가요? 한 줄로 간단히 답해주세요."}]
        )
