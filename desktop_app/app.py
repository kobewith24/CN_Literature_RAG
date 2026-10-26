# -*- coding: utf-8 -*-
"""文献搜索排序系统 · 桌面客户端

启动后是一个窗口：输入查询词 -> 点击「查询」（或回车）
-> 上方页签展示「精排结果（按排序分）」与「召回结果（按召回分）」各 50 条。
双击任意结果可复制文献标题。
"""
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import engine

BG = "#f3f5f9"
CARD = "#ffffff"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
TEXT = "#111827"
MUTED = "#6b7280"
ROW_ALT = "#f8fafc"
OK_GREEN = "#059669"


class SearchApp(object):
    def __init__(self, root):
        self.root = root
        self.engine = None
        self.ready = False
        self.queue = queue.Queue()
        self._build_ui()
        self._load_engine_async()
        self.root.after(120, self._poll_queue)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = self.root
        root.title("文献搜索排序系统 · Search_Rank_System")
        root.geometry("1200x780")
        root.minsize(1000, 640)
        root.configure(bg=BG)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure(
            "Title.TLabel",
            background=BG,
            foreground=TEXT,
            font=("Microsoft YaHei UI", 20, "bold"),
        )
        style.configure(
            "Sub.TLabel", background=BG, foreground=MUTED, font=("Microsoft YaHei UI", 10)
        )
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(18, 8),
            background="#e5e7eb",
        )
        style.map("TNotebook.Tab", background=[("selected", CARD)])
        style.configure(
            "Treeview",
            background=CARD,
            fieldbackground=CARD,
            foreground=TEXT,
            rowheight=30,
            font=("Microsoft YaHei UI", 10),
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 10, "bold"),
            background="#eef2f7",
            foreground=TEXT,
            relief="flat",
            padding=(8, 6),
        )
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", TEXT)])

        # 头部
        header = ttk.Frame(root, padding=(24, 18, 24, 6))
        header.pack(fill="x")
        left = ttk.Frame(header)
        left.pack(side="left")
        ttk.Label(left, text="文献搜索排序系统", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="语义召回（ERNIE-1.0 + HNSW）→ 精排（ERNIE-3.0-medium）",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        self.status_pill = tk.Label(
            header,
            text="● 模型加载中…",
            bg="#fef3c7",
            fg="#92400e",
            font=("Microsoft YaHei UI", 9),
            padx=12,
            pady=5,
        )
        self.status_pill.pack(side="right")

        # 搜索栏
        bar = ttk.Frame(root, padding=(24, 8, 24, 8))
        bar.pack(fill="x")
        card = ttk.Frame(bar, style="Card.TFrame", padding=10)
        card.pack(fill="x")
        self.entry = tk.Entry(
            card,
            font=("Microsoft YaHei UI", 12),
            relief="flat",
            bg="#f8fafc",
            fg=TEXT,
            insertbackground=TEXT,
            highlightthickness=1,
            highlightbackground="#d1d5db",
            highlightcolor=ACCENT,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=9, padx=(2, 10))
        self.entry.bind("<Return>", lambda e: self.start_search())
        self.btn = tk.Button(
            card,
            text="查 询",
            command=self.start_search,
            font=("Microsoft YaHei UI", 11, "bold"),
            bg=ACCENT,
            fg="white",
            activebackground=ACCENT_DARK,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            state="disabled",
            padx=26,
            pady=7,
        )
        self.btn.pack(side="right")

        # 结果区
        body = ttk.Frame(root, padding=(24, 4, 24, 0))
        body.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill="both", expand=True)

        self.rank_tree = self._make_tree(self.notebook, "精排结果（按排序分）")
        self.recall_tree = self._make_tree(self.notebook, "召回结果（按召回分）")
        self.notebook.add(self.rank_tree.master, text="精排结果 · Top50")
        self.notebook.add(self.recall_tree.master, text="召回结果 · Top50")

        # 底部状态栏
        footer = ttk.Frame(root, padding=(24, 8, 24, 14))
        footer.pack(fill="x")
        self.footer_label = ttk.Label(
            footer, text="就绪：输入查询词后按回车或点击「查询」", style="Sub.TLabel"
        )
        self.footer_label.pack(side="left")
        self.hint_label = ttk.Label(footer, text="双击结果可复制标题", style="Sub.TLabel")
        self.hint_label.pack(side="right")

    def _make_tree(self, parent, label):
        frame = ttk.Frame(parent, style="Card.TFrame")
        tree = ttk.Treeview(
            frame,
            columns=("rank", "title", "score"),
            show="headings",
            selectmode="browse",
        )
        tree.heading("rank", text="排名")
        tree.heading("title", text=label)
        tree.heading("score", text="分数")
        tree.column("rank", width=64, anchor="center", stretch=False)
        tree.column("title", width=820, anchor="w")
        tree.column("score", width=110, anchor="e", stretch=False)
        tree.tag_configure("odd", background=ROW_ALT)
        tree.tag_configure("even", background=CARD)

        ysb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

        tree.bind("<ButtonRelease-1>", lambda e, t=tree: self._on_select(t))
        tree.bind("<Double-1>", lambda e, t=tree: self._copy_selected(t))
        return tree

    # ------------------------------------------------------------- 引擎加载
    def _load_engine_async(self):
        def work():
            try:
                t0 = time.perf_counter()
                eng = engine.SearchEngine()
                self.queue.put(("ready", eng, time.perf_counter() - t0))
            except Exception as exc:  # noqa: BLE001
                self.queue.put(("error", "模型加载失败：%s" % exc))

        threading.Thread(target=work, daemon=True).start()

    # ----------------------------------------------------------------- 查询
    def start_search(self):
        if not self.ready:
            return
        query = self.entry.get().strip()
        if not query:
            self._set_footer("请先输入查询词")
            return
        self.btn.configure(state="disabled", text="检索中…")
        self._set_pill("● 检索中…", "#dbeafe", "#1d4ed8")
        self._set_footer("正在检索：「%s」…" % query)
        threading.Thread(target=self._search_async, args=(query,), daemon=True).start()

    def _search_async(self, query):
        try:
            result = self.engine.search(query)
            self.queue.put(("result", result))
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("error", "检索失败：%s" % exc))

    # ------------------------------------------------------------- 队列轮询
    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "ready":
                    self.engine = msg[1]
                    self.ready = True
                    self.btn.configure(state="normal")
                    self._set_pill("● 就绪", "#d1fae5", OK_GREEN)
                    self._set_footer(
                        "模型已加载（%.1fs，语料 %s 条）· 输入查询词后按回车"
                        % (msg[2], format(self.engine.corpus_size, ","))
                    )
                    if "--selftest" in sys.argv:
                        self.entry.insert(0, "计算机视觉")
                        self.start_search()
                elif msg[0] == "result":
                    self._render(msg[1])
                    if "--selftest" in sys.argv:
                        print(
                            "SELFTEST-OK ranked=%d recall=%d"
                            % (len(msg[1]["ranked"]), len(msg[1]["recall"])),
                            flush=True,
                        )
                        self.root.after(800, self.root.destroy)
                elif msg[0] == "error":
                    self._set_pill("● 出错", "#fee2e2", "#b91c1c")
                    self._set_footer(msg[1])
                    self.btn.configure(state="normal", text="查 询")
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    # ------------------------------------------------------------- 渲染结果
    def _render(self, result):
        self._fill_tree(self.rank_tree, result["ranked"])
        self._fill_tree(self.recall_tree, result["recall"])
        self.btn.configure(state="normal", text="查 询")
        self._set_pill("● 就绪", "#d1fae5", OK_GREEN)
        self._set_footer(
            "「%s」检索完成 · 召回 %d 条 · 精排 %d 条 · 耗时 %.1fs"
            % (
                result["query"],
                len(result["recall"]),
                len(result["ranked"]),
                result["elapsed"],
            )
        )

    @staticmethod
    def _fill_tree(tree, items):
        tree.delete(*tree.get_children())
        for i, item in enumerate(items, 1):
            tree.insert(
                "",
                "end",
                values=(i, item["title"], "%.4f" % item["score"]),
                tags=("odd" if i % 2 else "even",),
            )

    def _on_select(self, tree):
        sel = tree.selection()
        if sel:
            values = tree.item(sel[0], "values")
            self._set_footer("第 %s 名 · %s" % (values[0], values[1]))

    def _copy_selected(self, tree):
        sel = tree.selection()
        if sel:
            values = tree.item(sel[0], "values")
            self.root.clipboard_clear()
            self.root.clipboard_append(values[1])
            self._set_footer("已复制标题：%s" % values[1])

    def _set_footer(self, text):
        self.footer_label.configure(text=text)

    def _set_pill(self, text, bg, fg):
        self.status_pill.configure(text=text, bg=bg, fg=fg)


def main():
    root = tk.Tk()
    SearchApp(root)
    if "--smoke" in sys.argv:  # 自检模式：窗口显示几秒后自动退出
        root.after(6000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
