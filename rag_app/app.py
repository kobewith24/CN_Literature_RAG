# -*- coding: utf-8 -*-
"""文献检索问答 · 桌面客户端（Tkinter）

输入问题后：
  - 答案区流式显示生成结果，并列出前 5 条引用（点击查看摘要）
  - 「查看召回结果」「查看精排结果」按钮打开完整 Top50 列表

依赖后端：SpringBoot :8080（检索编排 + LangChain4j）与 FastAPI :8000（双路召回）。
"""
import json
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

from rag_client import RagClient

BG = "#eef1f6"
CARD = "#ffffff"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
TEXT = "#111827"
MUTED = "#6b7280"
OK = "#059669"
WARN = "#b45309"
ERRC = "#b91c1c"
CHIP_BG = "#eef2ff"
CHIP_FG = "#3730a3"
ROW_ALT = "#f8fafc"

FONT = "Microsoft YaHei UI"


class ResultsWindow(tk.Toplevel):
    """完整召回 / 精排结果窗口。"""

    def __init__(self, master, hits):
        super().__init__(master)
        self.title("检索结果 · Top50")
        self.geometry("1080x640")
        self.minsize(900, 520)
        self.configure(bg=BG)
        self.hits = hits

        style = ttk.Style(self)
        style.configure("R.Treeview", background=CARD, fieldbackground=CARD,
                        foreground=TEXT, rowheight=28, font=(FONT, 10), borderwidth=0)
        style.configure("R.Treeview.Heading", font=(FONT, 10, "bold"),
                        background="#eef2f7", foreground=TEXT, relief="flat", padding=(6, 5))
        style.map("R.Treeview", background=[("selected", "#dbeafe")])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=14, pady=(14, 6))

        self.trees = {}
        for key, label in (("recall", "召回结果（按向量相似度）"),
                           ("ranked", "精排结果（按重排分）")):
            frame = ttk.Frame(nb)
            nb.add(frame, text="  " + label + "  ")
            self.trees[key] = self._make_tree(frame)

        detail_frame = ttk.Frame(self)
        detail_frame.pack(fill="both", expand=False, padx=14, pady=(0, 12))
        ttk.Label(detail_frame, text="详情（选中条目查看摘要）",
                  foreground=MUTED).pack(anchor="w")
        self.detail = tk.Text(detail_frame, height=7, wrap="word", relief="flat",
                              bg=CARD, fg=TEXT, font=(FONT, 9), padx=10, pady=8)
        self.detail.pack(fill="both", expand=True, pady=(4, 0))
        self.detail.configure(state="disabled")

        self._fill()

    def _make_tree(self, parent):
        tree = ttk.Treeview(parent, columns=("rank", "score", "source", "title"),
                            show="headings", style="R.Treeview", selectmode="browse")
        tree.heading("rank", text="#")
        tree.heading("score", text="分数")
        tree.heading("source", text="来源")
        tree.heading("title", text="标题")
        tree.column("rank", width=56, anchor="center", stretch=False)
        tree.column("score", width=84, anchor="e", stretch=False)
        tree.column("source", width=90, anchor="center", stretch=False)
        tree.column("title", width=760, anchor="w")
        tree.tag_configure("odd", background=ROW_ALT)
        tree.tag_configure("even", background=CARD)
        ysb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        tree.bind("<<TreeviewSelect>>", lambda e, t=tree: self._show_detail(t))
        return tree

    def _fill(self):
        recall = sorted(self.hits, key=lambda h: h.get("score") or 0, reverse=True)
        ranked = sorted(self.hits, key=lambda h: h.get("rerank_score") or 0, reverse=True)
        for key, items, score_key in (("recall", recall, "score"),
                                      ("ranked", ranked, "rerank_score")):
            tree = self.trees[key]
            tree.delete(*tree.get_children())
            tree.iid_map = {}
            for i, h in enumerate(items[:50], 1):
                src = "摘要库" if h.get("source") == "doc" else "题录库"
                iid = "%s-%d" % (key, i)
                tree.insert("", "end", iid=iid,
                            values=(i, "%.4f" % (h.get(score_key) or 0), src,
                                    (h.get("title") or "")[:80]),
                            tags=("odd" if i % 2 else "even",))
                tree.iid_map[iid] = h

    def _show_detail(self, tree):
        sel = tree.selection()
        if not sel:
            return
        h = tree.iid_map.get(sel[0])
        if not h:
            return
        parts = [
            "标题：%s" % (h.get("title") or ""),
            "来源：%s   学科：%s   向量分：%.4f   重排分：%s" % (
                "摘要库" if h.get("source") == "doc" else "题录库",
                h.get("discipline") or "-",
                h.get("score") or 0,
                ("%.4f" % h["rerank_score"]) if h.get("rerank_score") is not None else "-"),
            "",
            (h.get("text") or "（题录条目无摘要）")[:1200],
        ]
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", "\n".join(parts))
        self.detail.configure(state="disabled")


class RagApp(object):
    def __init__(self, root):
        self.root = root
        self.client = RagClient()
        self.queue = queue.Queue()
        self.hits = []
        self.answer = ""
        self.ready = False
        self._build_ui()
        self._check_backend()
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = self.root
        root.title("文献检索问答 · Search_Rank_System")
        root.geometry("1120x780")
        root.minsize(960, 640)
        root.configure(bg=BG)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=TEXT, font=(FONT, 10))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=(FONT, 20, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=(FONT, 10))

        header = ttk.Frame(root, padding=(24, 18, 24, 6))
        header.pack(fill="x")
        left = ttk.Frame(header)
        left.pack(side="left")
        ttk.Label(left, text="文献检索问答", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="BGE-M3 双路召回（题录 30 万 + 摘要 39.6 万）→ 重排 → Qwen3-4B 生成",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 0))
        self.pill = tk.Label(header, text="● 连接中…", bg="#fef3c7", fg=WARN,
                             font=(FONT, 9), padx=12, pady=5)
        self.pill.pack(side="right")

        bar = ttk.Frame(root, padding=(24, 6, 24, 6))
        bar.pack(fill="x")
        card = ttk.Frame(bar, style="Card.TFrame", padding=10)
        card.pack(fill="x")
        self.entry = tk.Entry(card, font=(FONT, 12), relief="flat", bg="#f8fafc", fg=TEXT,
                              insertbackground=TEXT, highlightthickness=1,
                              highlightbackground="#d1d5db", highlightcolor=ACCENT)
        self.entry.pack(side="left", fill="x", expand=True, ipady=9, padx=(2, 10))
        self.entry.bind("<Return>", lambda e: self.ask())
        self.btn = tk.Button(card, text="提 问", command=self.ask, font=(FONT, 11, "bold"),
                             bg=ACCENT, fg="white", activebackground=ACCENT_DARK,
                             activeforeground="white", relief="flat", cursor="hand2",
                             padx=26, pady=7, state="disabled")
        self.btn.pack(side="right")

        body = ttk.Frame(root, padding=(24, 4, 24, 4))
        body.pack(fill="both", expand=True)
        answer_card = tk.Frame(body, bg=CARD, highlightthickness=1, highlightbackground="#e5e7eb")
        answer_card.pack(fill="both", expand=True)
        self.answer_text = tk.Text(answer_card, wrap="word", relief="flat", bg=CARD, fg=TEXT,
                                   font=(FONT, 11), padx=18, pady=14, spacing1=2, spacing3=4)
        self.answer_text.pack(fill="both", expand=True)
        self.answer_text.insert("1.0", "输入问题后按回车或点击「提问」。\n"
                                       "例如：双目立体视觉的相机标定有哪些常用方法？")
        self.answer_text.configure(state="disabled")

        cite = ttk.Frame(root, padding=(24, 2, 24, 2))
        cite.pack(fill="x")
        ttk.Label(cite, text="引用来源（前 5 条）", style="Sub.TLabel").pack(anchor="w")
        self.chips = tk.Frame(cite, bg=BG)
        self.chips.pack(fill="x", pady=(4, 0))

        actions = ttk.Frame(root, padding=(24, 8, 24, 14))
        actions.pack(fill="x")
        self.btn_recall = tk.Button(actions, text="查看召回结果（50）",
                                    command=self.open_results,
                                    font=(FONT, 10), bg="#e5e7eb", fg=TEXT, relief="flat",
                                    cursor="hand2", padx=14, pady=6, state="disabled")
        self.btn_recall.pack(side="left")
        self.btn_rank = tk.Button(actions, text="查看精排结果（50）",
                                  command=self.open_results,
                                  font=(FONT, 10), bg="#e5e7eb", fg=TEXT, relief="flat",
                                  cursor="hand2", padx=14, pady=6, state="disabled")
        self.btn_rank.pack(side="left", padx=8)
        self.footer = ttk.Label(actions, text="就绪", style="Sub.TLabel")
        self.footer.pack(side="right")

    # ------------------------------------------------------------ 后端检查
    def _check_backend(self):
        def work():
            ok, info = self.client.health()
            self.queue.put(("health", ok, info))
        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------------- 提问
    def ask(self):
        if not self.ready:
            return
        q = self.entry.get().strip()
        if not q:
            self._footer("请先输入问题")
            return
        self.entry.delete(0, "end")
        self.hits = []
        self.answer = ""
        self.btn.configure(state="disabled", text="生成中…")
        self.btn_recall.configure(state="disabled")
        self.btn_rank.configure(state="disabled")
        self._pill("● 检索中…", "#dbeafe", ACCENT_DARK)
        self._footer("检索 + 生成中…")
        self._clear_chips()
        self._set_answer("已提交，等待检索结果…")
        threading.Thread(target=self._ask_worker, args=(q,), daemon=True).start()

    def _ask_worker(self, q):
        try:
            data = self.client.retrieve(q, k=50)
            self.queue.put(("hits", data.get("hits") or []))
        except Exception as e:  # noqa: BLE001
            self.queue.put(("retrieve_error", str(e)))
        n = 0
        try:
            for event, payload in self.client.ask_stream(q, k=5):
                if event == "token":
                    n += 1
                self.queue.put((event, payload))
        except Exception as e:  # noqa: BLE001
            self.queue.put(("error", str(e)))
        self.token_count = n

    # --------------------------------------------------------------- 轮询
    def _poll(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == "health":
                    _, ok, info = msg
                    self.ready = bool(ok)
                    if self.ready:
                        self.pill.configure(text="● 就绪", bg="#d1fae5", fg=OK)
                        self.btn.configure(state="normal")
                        self.footer.configure(
                            text="检索服务就绪：题录 %s · 摘要 %s" % (
                                format(info.get("titles", 0), ","),
                                format(info.get("docs", 0), ",")))
                    else:
                        self.pill.configure(text="● 服务未启动", bg="#fee2e2", fg=ERRC)
                        self.footer.configure(text="请先运行 启动RAG.bat")
                elif kind == "hits":
                    self.hits = msg[1]
                    if self.hits:
                        self.btn_recall.configure(state="normal")
                        self.btn_rank.configure(state="normal")
                elif kind == "sources":
                    try:
                        self._render_chips(json.loads(msg[1]))
                    except json.JSONDecodeError:
                        pass
                elif kind == "token":
                    self.answer += msg[1]
                    self._set_answer(self.answer)
                elif kind == "done":
                    self._pill("● 就绪", "#d1fae5", OK)
                    self.btn.configure(state="normal", text="提 问")
                    self._footer("完成 · 答案 %d 字" % len(self.answer))
                elif kind == "error":
                    self._pill("● 出错", "#fee2e2", ERRC)
                    self._footer("生成失败：%s" % msg[1][:120])
                    self.btn.configure(state="normal", text="提 问")
                elif kind == "retrieve_error":
                    self._footer("检索失败：%s" % msg[1][:120])
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    # --------------------------------------------------------------- 渲染
    def _set_answer(self, text):
        self.answer_text.configure(state="normal")
        self.answer_text.delete("1.0", "end")
        self.answer_text.insert("1.0", text)
        self.answer_text.see("end")
        self.answer_text.configure(state="disabled")

    def _clear_chips(self):
        for w in self.chips.winfo_children():
            w.destroy()

    def _render_chips(self, hits):
        self._clear_chips()
        for i, h in enumerate(hits[:5], 1):
            title = (h.get("title") or "").strip()
            short = title if len(title) <= 26 else title[:26] + "…"
            chip = tk.Label(self.chips, text="[%d] %s" % (i, short), bg=CHIP_BG, fg=CHIP_FG,
                            font=(FONT, 9), padx=10, pady=4, cursor="hand2")
            chip.pack(side="left", padx=(0, 6))
            chip.bind("<Button-1>", lambda e, item=h: self._show_source(item))

    def _show_source(self, h):
        win = tk.Toplevel(self.root)
        win.title("引用详情")
        win.geometry("760x480")
        win.configure(bg=BG)
        txt = tk.Text(win, wrap="word", relief="flat", bg=CARD, fg=TEXT,
                      font=(FONT, 10), padx=16, pady=12)
        txt.pack(fill="both", expand=True, padx=12, pady=12)
        txt.insert("1.0", "标题：%s\n来源：%s    学科：%s\n\n%s" % (
            h.get("title") or "",
            "摘要库" if h.get("source") == "doc" else "题录库",
            h.get("discipline") or "-",
            h.get("text") or "（题录条目无摘要）"))
        txt.configure(state="disabled")

    def open_results(self):
        if not self.hits:
            return
        ResultsWindow(self.root, self.hits)

    # --------------------------------------------------------------- 小工具
    def _pill(self, text, bg, fg):
        self.pill.configure(text=text, bg=bg, fg=fg)

    def _footer(self, text):
        self.footer.configure(text=text)


def main():
    root = tk.Tk()
    app = RagApp(root)
    if "--selftest" in sys.argv:
        def auto():
            app.entry.insert(0, "双目立体视觉的相机标定有哪些常用方法？")
            app.ask()

        def report():
            print("SELFTEST hits=%d answer_chars=%d tokens=%s" % (
                len(app.hits), len(app.answer), getattr(app, "token_count", -1)), flush=True)
            root.destroy()

        root.after(2000, auto)
        root.after(90000, report)
    root.mainloop()


if __name__ == "__main__":
    main()
