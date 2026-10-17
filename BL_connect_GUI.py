import socket
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import queue
import json
import os
from datetime import datetime

# ─────────────────────────────────────────────
#  Файл истории подключений
# ─────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "bl_connect_history.json")


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_history(records: list):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records[-100:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ─────────────────────────────────────────────
#  TCP-клиент с постоянным соединением
# ─────────────────────────────────────────────
class TCPClient:
    """Держит постоянное TCP-соединение. Принимает данные в фоне."""

    def __init__(self, ip: str, port: int, timeout: int,
                 log_q: queue.Queue, event_q: queue.Queue):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.log_q = log_q
        self.event_q = event_q          # события: ("connected",) / ("disconnected", reason)
        self._sock: socket.socket | None = None
        self._recv_thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── публичные методы ──────────────────────

    def connect(self):
        """Подключиться в фоновом потоке."""
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def disconnect(self):
        """Разорвать соединение."""
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def send(self, data: str | bytes) -> bool:
        """Отправить данные. Возвращает False если не подключены."""
        if not self._sock:
            return False
        try:
            if isinstance(data, str):
                data = data.encode("utf-8")
            self._sock.sendall(data)
            return True
        except Exception as e:
            self._log(f"[!] Ошибка отправки: {e}")
            self._on_disconnect(str(e))
            return False

    @property
    def connected(self) -> bool:
        return self._sock is not None

    # ── внутренние ────────────────────────────

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_q.put(f"[{ts}] {msg}")

    def _connect_worker(self):
        self._stop.clear()
        try:
            self._log(f"[*] Подключение к {self.ip}:{self.port}…")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((self.ip, self.port))
            s.settimeout(None)          # блокирующий режим для recv
            self._sock = s
            self._log(f"[+] Соединение установлено ✓")
            self.event_q.put(("connected",))

            # Запускаем приёмник
            self._recv_thread = threading.Thread(
                target=self._recv_worker, daemon=True)
            self._recv_thread.start()

        except ConnectionRefusedError:
            self._log(f"[!] Отказано в соединении: {self.ip}:{self.port}")
            self.event_q.put(("disconnected", "отказано"))
        except socket.timeout:
            self._log(f"[!] Тайм-аут подключения ({self.timeout} с)")
            self.event_q.put(("disconnected", "тайм-аут"))
        except Exception as e:
            self._log(f"[!] Ошибка: {e}")
            self.event_q.put(("disconnected", str(e)))

    def _recv_worker(self):
        """Бесконечно читаем данные от сервера."""
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    # Сервер закрыл соединение
                    self._log("[!] Сервер закрыл соединение.")
                    self._on_disconnect("сервер закрыл")
                    break
                buf += chunk
                # Пробуем распарсить по строкам (\n)
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        self._log(f"◀ {text}")
                # Если пришли данные без \n — тоже выводим
                if buf and not self._stop.is_set():
                    try:
                        text = buf.decode("utf-8", errors="replace").strip()
                        if text:
                            self._log(f"◀ {text}")
                        buf = b""
                    except Exception:
                        pass
            except OSError:
                if not self._stop.is_set():
                    self._log("[!] Соединение разорвано.")
                    self._on_disconnect("разорвано")
                break

    def _on_disconnect(self, reason: str):
        self._sock = None
        self.event_q.put(("disconnected", reason))


# ─────────────────────────────────────────────
#  Главное GUI-приложение
# ─────────────────────────────────────────────
class App(tk.Tk):
    COLOR_BG     = "#0d1117"
    COLOR_PANEL  = "#161b22"
    COLOR_BORDER = "#30363d"
    COLOR_ACCENT = "#58a6ff"
    COLOR_GREEN  = "#3fb950"
    COLOR_RED    = "#f85149"
    COLOR_YELLOW = "#e3b341"
    COLOR_TEXT   = "#e6edf3"
    COLOR_MUTED  = "#8b949e"
    COLOR_INPUT  = "#21262d"

    FONT_MONO  = ("Consolas", 10)
    FONT_LABEL = ("Segoe UI", 10)
    FONT_TITLE = ("Segoe UI", 14, "bold")
    FONT_BOLD  = ("Segoe UI", 10, "bold")
    FONT_SMALL = ("Segoe UI", 8)

    def __init__(self):
        super().__init__()

        self.title("BL Connect — TCP Client")
        self.geometry("900x640")
        self.minsize(740, 480)
        self.configure(bg=self.COLOR_BG)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._log_q   = queue.Queue()
        self._evt_q   = queue.Queue()
        self._client: TCPClient | None = None
        self._history = load_history()

        self._build_ui()
        self._poll()

    # ─────────────── UI Builder ───────────────

    def _build_ui(self):
        # ── Заголовок
        hdr = tk.Frame(self, bg=self.COLOR_BG)
        hdr.pack(fill="x", padx=20, pady=(14, 0))

        tk.Label(hdr, text="⚡  BL Connect",
                 font=self.FONT_TITLE, bg=self.COLOR_BG, fg=self.COLOR_ACCENT).pack(side="left")

        self._status_lbl = tk.Label(hdr, text="● ОТКЛЮЧЕН",
                                    font=self.FONT_BOLD, bg=self.COLOR_BG, fg=self.COLOR_RED)
        self._status_lbl.pack(side="right")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=8)

        # ── Тело
        body = tk.Frame(self, bg=self.COLOR_BG)
        body.pack(fill="both", expand=True, padx=20)
        body.columnconfigure(0, weight=0, minsize=260)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ══ Левая панель ══
        left = tk.Frame(body, bg=self.COLOR_PANEL,
                        highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)

        tk.Label(left, text="Параметры подключения",
                 font=self.FONT_BOLD, bg=self.COLOR_PANEL, fg=self.COLOR_TEXT
                 ).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 6))

        # ─ Поля ввода ─
        self._fields = {}
        field_defs = [
            ("server_ip", "IP-адрес сервера",     "192.168.67.1"),
            ("port",      "Порт",                  "5000"),
            ("timeout",   "Тайм-аут подкл. (сек)", "10"),
        ]
        for i, (key, label, default) in enumerate(field_defs, start=1):
            row_f = tk.Frame(left, bg=self.COLOR_PANEL)
            row_f.grid(row=i, column=0, sticky="ew", padx=14, pady=4)
            row_f.columnconfigure(1, weight=1)

            tk.Label(row_f, text=label, font=self.FONT_LABEL,
                     bg=self.COLOR_PANEL, fg=self.COLOR_MUTED,
                     anchor="w", width=22).grid(row=0, column=0, sticky="w")

            var = tk.StringVar(value=default)
            self._fields[key] = var
            tk.Entry(row_f, textvariable=var, font=self.FONT_MONO,
                     bg=self.COLOR_INPUT, fg=self.COLOR_TEXT,
                     insertbackground=self.COLOR_TEXT,
                     relief="flat", bd=4).grid(row=0, column=1, sticky="ew")

        sep = len(field_defs) + 1
        ttk.Separator(left, orient="horizontal").grid(
            row=sep, column=0, sticky="ew", padx=10, pady=10)

        # ─ Кнопки Подключить / Отключить ─
        btn_f = tk.Frame(left, bg=self.COLOR_PANEL)
        btn_f.grid(row=sep + 1, column=0, sticky="ew", padx=14, pady=4)
        btn_f.columnconfigure(0, weight=1)
        btn_f.columnconfigure(1, weight=1)

        self._btn_connect = tk.Button(
            btn_f, text="🔗 Подключить", command=self._connect,
            bg=self.COLOR_GREEN, fg="#0d1117", font=self.FONT_BOLD,
            relief="flat", cursor="hand2", pady=7)
        self._btn_connect.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._btn_disconnect = tk.Button(
            btn_f, text="✖ Отключить", command=self._disconnect,
            bg=self.COLOR_RED, fg="white", font=self.FONT_BOLD,
            relief="flat", cursor="hand2", pady=7, state="disabled")
        self._btn_disconnect.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # ─ Очистить лог ─
        tk.Button(left, text="🗑  Очистить лог", command=self._clear_log,
                  bg=self.COLOR_INPUT, fg=self.COLOR_MUTED, font=self.FONT_SMALL,
                  relief="flat", cursor="hand2", pady=5
                  ).grid(row=sep + 2, column=0, sticky="ew", padx=14, pady=(8, 0))

        ttk.Separator(left, orient="horizontal").grid(
            row=sep + 3, column=0, sticky="ew", padx=10, pady=10)

        # ─ История ─
        hist_hdr = tk.Frame(left, bg=self.COLOR_PANEL)
        hist_hdr.grid(row=sep + 4, column=0, sticky="ew", padx=14)
        tk.Label(hist_hdr, text="История подключений", font=self.FONT_BOLD,
                 bg=self.COLOR_PANEL, fg=self.COLOR_TEXT).pack(side="left")
        tk.Button(hist_hdr, text="Очистить", font=self.FONT_SMALL,
                  bg=self.COLOR_INPUT, fg=self.COLOR_MUTED, relief="flat",
                  cursor="hand2", command=self._clear_history).pack(side="right")

        hist_f = tk.Frame(left, bg=self.COLOR_PANEL)
        hist_f.grid(row=sep + 5, column=0, sticky="nsew", padx=14, pady=(4, 14))
        left.rowconfigure(sep + 5, weight=1)

        scroll_h = tk.Scrollbar(hist_f)
        scroll_h.pack(side="right", fill="y")
        self._hist_list = tk.Listbox(hist_f, yscrollcommand=scroll_h.set,
                                     bg=self.COLOR_INPUT, fg=self.COLOR_TEXT,
                                     font=("Consolas", 8),
                                     selectbackground=self.COLOR_ACCENT,
                                     selectforeground="#0d1117",
                                     relief="flat", bd=0)
        self._hist_list.pack(fill="both", expand=True)
        self._hist_list.bind("<Double-Button-1>", self._load_history_item)
        scroll_h.config(command=self._hist_list.yview)
        self._refresh_history_list()

        # ══ Правая панель ══
        right = tk.Frame(body, bg=self.COLOR_BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ─ Лог (входящие + системные) ─
        lbl_row = tk.Frame(right, bg=self.COLOR_BG)
        lbl_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        tk.Label(lbl_row, text="Лог / Входящие данные", font=self.FONT_BOLD,
                 bg=self.COLOR_BG, fg=self.COLOR_TEXT).pack(side="left")

        self._log_box = scrolledtext.ScrolledText(
            right, font=self.FONT_MONO, bg=self.COLOR_PANEL,
            fg=self.COLOR_GREEN, insertbackground=self.COLOR_TEXT,
            relief="flat", bd=0, wrap="word", state="disabled")
        self._log_box.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

        # Цвета тегов
        self._log_box.tag_config("recv",   foreground="#79c0ff")   # голубой — входящие
        self._log_box.tag_config("send",   foreground="#e3b341")   # жёлтый  — отправленные
        self._log_box.tag_config("system", foreground="#8b949e")   # серый    — системные
        self._log_box.tag_config("error",  foreground="#f85149")   # красный  — ошибки

        # ─ Поле команды ─
        cmd_panel = tk.Frame(right, bg=self.COLOR_PANEL,
                             highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        cmd_panel.grid(row=2, column=0, sticky="ew", pady=4)

        tk.Label(cmd_panel, text="Отправить:", font=self.FONT_LABEL,
                 bg=self.COLOR_PANEL, fg=self.COLOR_MUTED).pack(side="left", padx=(8, 4), pady=6)

        self._cmd_var = tk.StringVar()
        self._cmd_entry = tk.Entry(cmd_panel, textvariable=self._cmd_var, font=self.FONT_MONO,
                                   bg=self.COLOR_INPUT, fg=self.COLOR_TEXT,
                                   insertbackground=self.COLOR_TEXT, relief="flat", bd=4)
        self._cmd_entry.pack(side="left", fill="x", expand=True, padx=4, pady=6)
        self._cmd_entry.bind("<Return>", lambda e: self._send_command())

        tk.Button(cmd_panel, text="▶ Отправить", command=self._send_command,
                  bg=self.COLOR_ACCENT, fg="#0d1117", font=self.FONT_BOLD,
                  relief="flat", cursor="hand2", padx=10, pady=4).pack(side="left", padx=6)

        # ─ Быстрые команды ─
        hints = tk.Frame(right, bg=self.COLOR_BG)
        hints.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        tk.Label(hints, text="Быстрые:", font=self.FONT_SMALL,
                 bg=self.COLOR_BG, fg=self.COLOR_MUTED).pack(side="left", padx=(0, 6))

        self._quick_btns = []
        for label, val in [("ping", "ping"), ("info", "info"),
                            ("status", "status"), ("clear log", None)]:
            btn = tk.Button(hints, text=label,
                            command=lambda v=val: self._quick_send(v),
                            bg=self.COLOR_INPUT, fg=self.COLOR_ACCENT, font=self.FONT_SMALL,
                            relief="flat", cursor="hand2", padx=8, pady=2)
            btn.pack(side="left", padx=2)
            if val is not None:
                self._quick_btns.append(btn)

        # ── Строка статуса
        status_bar = tk.Frame(self, bg="#010409")
        status_bar.pack(fill="x", side="bottom")
        self._bar_label = tk.Label(status_bar, text="Не подключен",
                                   font=self.FONT_SMALL, bg="#010409",
                                   fg=self.COLOR_MUTED, anchor="w")
        self._bar_label.pack(side="left", padx=10, pady=3)

        self._bytes_lbl = tk.Label(status_bar, text="",
                                   font=self.FONT_SMALL, bg="#010409",
                                   fg=self.COLOR_MUTED, anchor="e")
        self._bytes_lbl.pack(side="right", padx=10, pady=3)

        self._recv_total = 0
        self._send_total = 0

    # ─────────────── Подключение ───────────────

    def _parse_fields(self):
        try:
            return {
                "server_ip": self._fields["server_ip"].get().strip(),
                "port":      int(self._fields["port"].get().strip()),
                "timeout":   int(self._fields["timeout"].get().strip()),
            }
        except ValueError as e:
            self._log_colored(f"[!] Ошибка параметров: {e}", "error")
            return None

    def _connect(self):
        params = self._parse_fields()
        if not params:
            return
        if self._client and self._client.connected:
            return

        self._client = TCPClient(
            ip=params["server_ip"],
            port=params["port"],
            timeout=params["timeout"],
            log_q=self._log_q,
            event_q=self._evt_q,
        )
        self._client.connect()
        self._set_status("● ПОДКЛЮЧЕНИЕ…", self.COLOR_YELLOW)
        self._bar_label.config(text=f"Подключение → {params['server_ip']}:{params['port']}…")
        self._btn_connect.config(state="disabled")

    def _disconnect(self):
        if self._client:
            self._client.disconnect()
            self._client = None
        self._on_disconnected("пользователь")

    def _on_connected(self):
        p = self._parse_fields()
        srv = f"{p['server_ip']}:{p['port']}" if p else "?"
        self._set_status("● ПОДКЛЮЧЕН", self.COLOR_GREEN)
        self._bar_label.config(text=f"Подключен: {srv}")
        self._btn_connect.config(state="disabled")
        self._btn_disconnect.config(state="normal")

        self._history.append({
            "ts": datetime.now().isoformat(),
            "server": srv,
            "event": "connect",
        })
        save_history(self._history)
        self._refresh_history_list()

    def _on_disconnected(self, reason: str):
        self._set_status("● ОТКЛЮЧЕН", self.COLOR_RED)
        self._bar_label.config(text=f"Отключен ({reason})")
        self._btn_connect.config(state="normal")
        self._btn_disconnect.config(state="disabled")

    # ─────────────── Отправка / приём ───────────────

    def _send_command(self):
        raw = self._cmd_var.get()
        self._cmd_var.set("")
        if not raw:
            return

        if not self._client or not self._client.connected:
            self._log_colored("[!] Нет подключения. Нажмите «Подключить».", "error")
            return

        data = raw + "\n"
        ok = self._client.send(data)
        if ok:
            ts = datetime.now().strftime("%H:%M:%S")
            self._log_colored(f"[{ts}] ▶ {raw}", "send")
            self._send_total += len(data)
            self._update_bytes()

    def _quick_send(self, val):
        if val is None:
            self._clear_log()
            return
        self._cmd_var.set(val)
        self._send_command()

    # ─────────────── Лог ───────────────

    def _log_colored(self, msg: str, tag: str = ""):
        self._log_box.config(state="normal")
        if tag:
            self._log_box.insert("end", msg + "\n", tag)
        else:
            self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    def _clear_log(self):
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")

    def _set_status(self, text, color):
        self._status_lbl.config(text=text, fg=color)

    def _update_bytes(self):
        self._bytes_lbl.config(
            text=f"↑ {self._send_total} Б   ↓ {self._recv_total} Б")

    # ─────────────── История ───────────────

    def _refresh_history_list(self):
        self._hist_list.delete(0, "end")
        for rec in reversed(self._history):
            ts  = rec.get("ts", "")[:16].replace("T", " ")
            srv = rec.get("server", "?")
            ev  = rec.get("event", "")
            ms  = rec.get("ms", "")
            if ms:
                self._hist_list.insert("end", f"{ts}  {srv}  ping {ms} мс")
            else:
                self._hist_list.insert("end", f"{ts}  {srv}  {ev}")

    def _load_history_item(self, event):
        sel = self._hist_list.curselection()
        if not sel:
            return
        idx = len(self._history) - 1 - sel[0]
        rec = self._history[idx]
        self._log_colored(
            f"[История] {rec.get('ts','')}  {rec.get('server','')}  "
            f"{rec.get('event','')}  {rec.get('ms','')}",
            "system"
        )

    def _clear_history(self):
        self._history.clear()
        save_history(self._history)
        self._refresh_history_list()

    # ─────────────── Polling ───────────────

    def _poll(self):
        # ── события подключения/отключения
        while not self._evt_q.empty():
            evt = self._evt_q.get_nowait()
            if evt[0] == "connected":
                self._on_connected()
            elif evt[0] == "disconnected":
                reason = evt[1] if len(evt) > 1 else ""
                self._on_disconnected(reason)

        # ── входящий лог
        while not self._log_q.empty():
            msg = self._log_q.get_nowait()
            # Определяем тег по префиксу
            if msg.strip().startswith(("[*]", "[+]", "[!]", "[History")):
                tag = "system" if msg.strip().startswith(("[*]", "[+]")) else "error"
            elif "◀" in msg:
                tag = "recv"
                # считаем байты входящих
                self._recv_total += len(msg)
                self._update_bytes()
            else:
                tag = "system"
            self._log_colored(msg, tag)

        self.after(100, self._poll)

    def _on_close(self):
        if self._client:
            self._client.disconnect()
        self.destroy()


# ─────────────────────────────────────────────
#  Точка входа
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
