# -*- coding: utf-8 -*-
"""RAG 服务客户端：双路召回检索 + SSE 流式问答。"""
import requests


class RagClient(object):
    def __init__(self, api_base="http://127.0.0.1:8080",
                 retrieval_base="http://127.0.0.1:8000", timeout=300):
        self.api_base = api_base.rstrip("/")
        self.retrieval_base = retrieval_base.rstrip("/")
        self.timeout = timeout

    def health(self):
        try:
            r = requests.get(self.retrieval_base + "/health", timeout=5)
            if r.status_code == 200:
                return True, r.json()
        except Exception as e:  # noqa: BLE001
            return False, {"error": str(e)}
        return False, {}

    def retrieve(self, query, k=50):
        r = requests.get(self.retrieval_base + "/retrieve",
                         params={"q": query, "k": k, "mode": "hybrid"},
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def ask_stream(self, query, k=5):
        """生成器：yield (event, data)。"""
        with requests.get(self.api_base + "/api/ask",
                          params={"q": query, "k": k},
                          stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            event, data = None, []
            for raw in resp.iter_lines(decode_unicode=False):
                if raw is None:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if line == "":
                    if event:
                        yield event, "\n".join(data)
                    event, data = None, []
                    continue
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data.append(line[len("data:"):].lstrip())
                else:
                    data.append(line)
