"""
Отправка команд в COM-порт: из txt-файла построчно / блоком и вручную.
Зависимости: pip install pyserial
"""
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import serial
import serial.tools.list_ports

BAUDS = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]
EOLS = {"\\n": "\n", "\\r\\n": "\r\n", "\\r": "\r", "нет": ""}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMMA Serial Sender")
        self.geometry("700x620")

        self.ser = None
        self.lock = threading.Lock()          
        self.rx_queue = queue.Queue()         
        self.stop_flag = threading.Event()    
        self.reader_stop = threading.Event()  
        self.sending = False
        self.history, self.hist_pos = [], 0

        self.t_start = None
        self.expected_count = None

        self.file_var = tk.StringVar()
        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="115200") # Поставил 115200 как в ESP32
        self.delay_var = tk.StringVar(value="1")
        self.chunk_var = tk.StringVar(value="8192")  # Идеально под Serial.setRxBufferSize(8192)
        self.cpause_var = tk.StringVar(value="0")    # Без пауз для максимальной скорости
        self.eol_var = tk.StringVar(value="\\n")
        self.skip_empty = tk.BooleanVar(value=True)
        self.wait_reply = tk.BooleanVar(value=False)
        self.block_mode = tk.BooleanVar(value=True)  # Блочный режим включен по умолчанию
        self.cmd_var = tk.StringVar()

        self._build()
        self.refresh_ports()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI ----------
    def _build(self):
        pad = {"padx": 6, "pady": 4}

        conn = ttk.LabelFrame(self, text="Подключение")
        conn.pack(fill="x", **pad)
        conn.columnconfigure(1, weight=1)
        ttk.Label(conn, text="Порт:").grid(row=0, column=0, sticky="w", padx=4)
        self.port_box = ttk.Combobox(conn, textvariable=self.port_var, state="readonly")
        self.port_box.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(conn, text="Обновить", command=self.refresh_ports).grid(row=0, column=2, padx=2)
        ttk.Label(conn, text="Скорость:").grid(row=0, column=3, padx=(10, 2))
        ttk.Combobox(conn, textvariable=self.baud_var, values=BAUDS, width=9).grid(row=0, column=4)
        self.conn_btn = ttk.Button(conn, text="Подключиться", command=self.toggle_connect)
        self.conn_btn.grid(row=0, column=5, padx=6)

        manual = ttk.LabelFrame(self, text="Ручная отправка")
        manual.pack(fill="x", **pad)
        manual.columnconfigure(0, weight=1)
        self.cmd_entry = ttk.Entry(manual, textvariable=self.cmd_var)
        self.cmd_entry.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self.cmd_entry.bind("<Return>", lambda e: self.send_manual())
        self.cmd_entry.bind("<Up>", lambda e: self.hist_move(-1))
        self.cmd_entry.bind("<Down>", lambda e: self.hist_move(1))
        ttk.Button(manual, text="Отправить", command=self.send_manual).grid(row=0, column=1, padx=4)
        ttk.Label(manual, text="Конец строки:").grid(row=0, column=2, padx=(10, 2))
        ttk.Combobox(manual, textvariable=self.eol_var, values=list(EOLS), width=6,
                     state="readonly").grid(row=0, column=3, padx=4)

        filef = ttk.LabelFrame(self, text="Отправка из файла")
        filef.pack(fill="x", **pad)
        filef.columnconfigure(1, weight=1)
        ttk.Label(filef, text="Файл:").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(filef, textvariable=self.file_var).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(filef, text="Выбрать...", command=self.pick_file).grid(row=0, column=2, padx=4)

        row2 = ttk.Frame(filef)
        row2.grid(row=1, column=0, columnspan=3, sticky="ew", padx=4)
        ttk.Label(row2, text="Пауза, мс:").pack(side="left")
        ttk.Entry(row2, textvariable=self.delay_var, width=7).pack(side="left", padx=4)
        ttk.Checkbutton(row2, text="Пропускать пустые, # и // строки",
                        variable=self.skip_empty).pack(side="left", padx=10)
        ttk.Checkbutton(row2, text="Ждать ответ после каждой команды",
                        variable=self.wait_reply).pack(side="left")

        row2b = ttk.Frame(filef)
        row2b.grid(row=2, column=0, columnspan=3, sticky="ew", padx=4)
        ttk.Checkbutton(row2b, text="Слать блоком (пауза и ожидание ответа игнорируются)",
                        variable=self.block_mode).pack(side="left")
        ttk.Label(row2b, text="Кусок, байт:").pack(side="left", padx=(10, 2))
        ttk.Entry(row2b, textvariable=self.chunk_var, width=6).pack(side="left")
        ttk.Label(row2b, text="пауза, мс:").pack(side="left", padx=(6, 2))
        ttk.Entry(row2b, textvariable=self.cpause_var, width=5).pack(side="left")

        row3 = ttk.Frame(filef)
        row3.grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
        self.start_btn = ttk.Button(row3, text="Старт", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(row3, text="Стоп", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(row3, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=6)

        logf = ttk.Frame(self)
        logf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(logf, state="disabled", wrap="none")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        ttk.Button(self, text="Очистить лог", command=self.clear_log).pack(anchor="e", padx=6, pady=(0, 6))

        self.update_state()

    def log_line(self, text):
        def _do():
            self.log.configure(state="normal")
            self.log.insert("end", text + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        self.after(0, _do)

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def set_progress(self, value, maximum):
        self.after(0, lambda: self.progress.configure(maximum=maximum, value=value))

    def update_state(self):
        def _do():
            connected = self.ser is not None
            self.conn_btn.configure(text="Отключиться" if connected else "Подключиться")
            self.start_btn.configure(state="normal" if connected and not self.sending else "disabled")
            self.stop_btn.configure(state="normal" if self.sending else "disabled")
        self.after(0, _do)

    def pick_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")])
        if path:
            self.file_var.set(path)

    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_box["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])

    # ---------- подключение ----------
    def toggle_connect(self):
        if self.ser is None:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("Внимание", "Выбери порт")
            return
        try:
            baud = int(self.baud_var.get())
            self.ser = serial.Serial(port, baud, timeout=0.1)
        except (ValueError, serial.SerialException) as e:
            self.ser = None
            messagebox.showerror("Ошибка", str(e))
            return
        self.reader_stop.clear()
        threading.Thread(target=self.reader_loop, args=(self.ser,), daemon=True).start()
        self.log_line(f"--- Подключено: {port} @ {baud} ---")
        self.update_state()

    def disconnect(self):
        self.stop_flag.set()
        self.reader_stop.set()
        ser, self.ser = self.ser, None
        if ser:
            time.sleep(0.15)
            with self.lock:
                try:
                    ser.close()
                except serial.SerialException:
                    pass
        self.log_line("--- Отключено ---")
        self.update_state()

    def reader_loop(self, ser):
        buf = b""
        last_rx = time.time()
        while not self.reader_stop.is_set():
            try:
                data = ser.read(ser.in_waiting or 1)
            except (serial.SerialException, OSError, TypeError):
                if not self.reader_stop.is_set():
                    self.log_line("--- Потеряно соединение с портом ---")
                    self.after(0, self.disconnect)
                return
            if data:
                buf += data
                last_rx = time.time()
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self.emit_rx(line)
            elif buf and time.time() - last_rx > 0.3:
                self.emit_rx(buf)  
                buf = b""

    def emit_rx(self, raw):
        text = raw.decode("utf-8", errors="replace").strip("\r\n")
        self.log_line(f"<< {text}")
        self.rx_queue.put(text)

        m = re.search(r"\[QUEUE\] Loaded (\d+) commands", text)
        if m and self.t_start is not None:
            got = int(m.group(1))
            dt = time.perf_counter() - self.t_start
            self.log_line(f"=== ESP подтвердил загрузку: {got} команд, "
                          f"{dt:.2f} с от начала отправки ===")
            if self.expected_count is not None and got != self.expected_count:
                self.log_line(f"!!! ВНИМАНИЕ: ожидалось {self.expected_count}, "
                              f"загружено {got} — часть строк потеряна !!!")
            self.t_start = None
            self.expected_count = None

    def write_port(self, text):
        eol = EOLS[self.eol_var.get()]
        with self.lock:
            if self.ser is None:
                raise serial.SerialException("порт не подключён")
            self.ser.write((text + eol).encode("utf-8"))

    # ---------- ручная отправка ----------
    def send_manual(self):
        if self.ser is None:
            messagebox.showwarning("Внимание", "Сначала подключись к порту")
            return
        text = self.cmd_var.get()
        try:
            self.write_port(text)
        except serial.SerialException as e:
            self.log_line(f"Ошибка порта: {e}")
            return
        self.log_line(f">> {text}")
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self.hist_pos = len(self.history)
        self.cmd_var.set("")

    def hist_move(self, step):
        if not self.history:
            return "break"
        self.hist_pos = max(0, min(len(self.history), self.hist_pos + step))
        self.cmd_var.set(self.history[self.hist_pos] if self.hist_pos < len(self.history) else "")
        self.cmd_entry.icursor("end")
        return "break"

    # ---------- отправка файла ----------
    def start(self):
        path = self.file_var.get()
        if not path:
            messagebox.showwarning("Внимание", "Выбери файл")
            return
        try:
            delay = float(self.delay_var.get().replace(",", ".")) / 1000.0
            chunk = max(1, int(self.chunk_var.get()))
            cpause = max(0.0, float(self.cpause_var.get().replace(",", ".")) / 1000.0)
            with open(path, encoding="utf-8") as f:
                all_lines = f.read().splitlines()
        except ValueError:
            messagebox.showerror("Ошибка", "Пауза и размер куска должны быть числами")
            return
        except OSError as e:
            messagebox.showerror("Ошибка", str(e))
            return

        lines = all_lines
        if self.skip_empty.get():
            lines = [l for l in all_lines
                     if l.strip() and not l.lstrip().startswith(("#", "//"))]
        n_empty = sum(1 for l in all_lines if not l.strip())
        n_comm = sum(1 for l in all_lines if l.lstrip().startswith(("#", "//")))
        self.log_line(f"--- Файл: {len(all_lines)} строк; пустых {n_empty}, "
                      f"комментариев (# и //) {n_comm}; к отправке {len(lines)} ---")
        if not lines:
            messagebox.showinfo("Инфо", "В файле нет команд")
            return

        self.stop_flag.clear()
        self.sending = True
        self.update_state()
        threading.Thread(
            target=self.run_send,
            args=(delay, lines, self.wait_reply.get(), self.block_mode.get(), chunk, cpause),
            daemon=True).start()

    def stop(self):
        self.stop_flag.set()

    @staticmethod
    def count_queued(lines):
        """Сколько строк между /load (или /load_test) и /end — столько ESP должна загрузить в очередь."""
        stripped = [l.strip() for l in lines]
        try:
            # Ищем новую или старую команду старта
            if "/load" in stripped:
                a = stripped.index("/load")
            else:
                a = stripped.index("/load_test")
                
            b = stripped.index("/end", a + 1)
        except ValueError:
            return None
        return sum(1 for l in stripped[a + 1:b] if l)

    def run_send_block(self, lines, chunk_size, cpause):
        total = len(lines)
        eol = EOLS[self.eol_var.get()] or "\n"
        data = "".join(l + eol for l in lines).encode("utf-8")

        self.expected_count = self.count_queued(lines)
        self.t_start = time.perf_counter()
        sent = 0
        with self.lock:
            if self.ser is None:
                raise serial.SerialException("порт не подключён")
            while sent < len(data):
                if self.stop_flag.is_set():
                    self.log_line("--- Остановлено ---")
                    return
                piece = data[sent:sent + chunk_size]
                
                self.ser.write(piece)
                self.ser.flush()  # <--- ДОБАВЬ ВОТ ЭТУ СТРОКУ! (Она заставит ПК отправить данные мгновенно)
                
                sent += len(piece)
                self.set_progress(sent, len(data))
                if cpause > 0 and sent < len(data):
                    time.sleep(cpause)
            self.ser.flush()
        dt = time.perf_counter() - self.t_start
        self.log_line(f"=== Блок отправлен: {total} строк, {len(data)} байт, {dt:.3f} с "
                      f"(кусок {chunk_size} Б, пауза {cpause * 1000:.0f} мс) ===")
        if self.expected_count is None:
            self.t_start = None

    def run_send(self, delay, lines, wait_reply, block, chunk, cpause):
        total = len(lines)
        try:
            if block:
                self.run_send_block(lines, chunk, cpause)
                return

            self.expected_count = self.count_queued(lines)
            self.t_start = time.perf_counter()
            for i, line in enumerate(lines, 1):
                if self.stop_flag.is_set():
                    self.log_line("--- Остановлено ---")
                    break
                while not self.rx_queue.empty():
                    self.rx_queue.get_nowait()
                self.write_port(line)
                self.log_line(f"[{i}/{total}] >> {line}")
                self.set_progress(i, total)
                if wait_reply:
                    try:
                        self.rx_queue.get(timeout=2)
                    except queue.Empty:
                        self.log_line("    (нет ответа за 2 с)")
                time.sleep(delay)
            else:
                dt = time.perf_counter() - self.t_start
                self.log_line(f"=== Построчно отправлено {total} строк за {dt:.2f} с ===")
        except serial.SerialException as e:
            self.log_line(f"Ошибка порта: {e}")
        finally:
            self.sending = False
            self.update_state()

    def on_close(self):
        self.disconnect()
        self.destroy()

if __name__ == "__main__":
    App().mainloop()