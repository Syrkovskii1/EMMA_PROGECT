# -*- coding: utf-8 -*-
"""Общее окно управления EMMA и редактора сценариев.

Установка на каждом ПК:
    pip install pyserial numpy sounddevice websocket-client
"""

from __future__ import annotations

import base64
import json
import importlib.util
import os
import re
import socket
import sys
import threading
import time
import types
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter as tk

import numpy as np
import serial
import serial.tools.list_ports
import sounddevice as sd

try:
    import websocket
except ImportError as exc:
    raise SystemExit("Нужна библиотека websocket-client: pip install websocket-client") from exc


SERVER_URL = "wss://scenario.vlessvps.com/ws"
ROOM = "emma-violin"
# Общий ключ приложения, а не пароль от сервера. Не публикуйте этот файл открыто.
ROOM_TOKEN = "X6DEK3cAZ9DXT98NPY2H9M6C5wzWCsW7"
BAUD_RATE = 115200
SEND_DELAY = 0.002

def load_original_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Не удалось открыть оригинал: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Редактор всегда берётся рядом с основной программой, а не из чужого профиля Windows.
ORIGINAL_SCENARIO = load_original_module("original_scenario", Path(__file__).with_name("scenario_editor.py"))
EMMA_ORIGINAL_B85 = r"""c-rk8*>c;~^<6Xihda=Nz=Wni(sFGn@{D6UolcX~6FZ%(gMr9P2^9!10BA{XvYpgv<gt@XI_acYnkEl@=~R&=o7VaZz%S%Wdd|66E*47ilBDTSwng&dF6W+oqbJy@=T5Pf(>Cpe8P@kZr_SQY_?V(7<71a!dF2AT%3ag2;KgopcYN$r;b(j-_$FKn4#FGZ&0r(E$AUxr7~F^78^Ip@*<is|u!~O)V9FtlZw5QTfWfZ=oDl2>TZ~ryfOEoIjLr@YgM$cwQ}{ky!)G-(@r!g_xXxg=!(adati!vF@Gi`{fyZq!IBz@HA<z!s*+H;Pn+Chl8vqGl+lSwe7`$vSoJxS~&_Qt2KjGtc@I!bj*pB!8AsE2FUHE+<4uFNdRgVCj86RWt(=&U_v_0RjtmF}G_c=g)7~TqR1UvE6Re#a38Si!t-6J&DdaDr$#>dQV&vAX`n_ZrMSai98G)qU8Oxx!!Gdy@ZKGt!aZZZ~6@t3O1Yq^eP@wU%b{3`2m&odVIyt5*m6vLYNSo0mn@@kgp`8p1JxL|y2g8lf}8vJF!SAh2zb&rU);>pwh9Upt~!t0mx*DhSU%;s3VKHbnXmi$aG#3ZtY;K1+>i#Y{|q9SR69pnU$`~99;5mdf>^-8l*86UfP`PEDMrOPj0coVkPYMM69<gH<+4WRo&<lsG^^8?_vTM~;9e-8x15Axg%47LfdZ3p**gDRtUI|6Gh%kVsQ8R#vfD&m7S?=W3AZPV9v)#FyD%3R0sE74>8!O<EH0=W26@}%aQzQt9gD7;FgZ2AIsy4-hHRb~2&wsNMS;k4XD-~l-BRmbKsazI$ak4U_3Mh8{J9Fj1U3BYwEZ(Hzr7X<S*%m-;JoS@J5O{=Z<O|<|d@3`Ds6kMk!u%(>$O+<L42yQ;WGY7#wN+z(u`!M(ju)o2EGgyNC1PVM_M*IW=O_!lOQg@xy)f#?Cd5%<Kyv}!efH{Au_L6IKxtgH8l9^KL8LcI?V_MdnvZ7R3&uFjAou${+xicAhz*a9C^W0KlS(Tv}&nbigIDAH_h@CvM&F2)0cl;7!A|iDVd4vmgyXbW19r!3VTsGVO;#{Kvn!-1H*aQ`gV_T~Q=83l{oEU+BIuG`XecyNN4BTG>{`N^R4^VPpY0Ckru-kJ*(pQEZ)W}2Da%`Ko065<kxGy>Rn+#8up6)vf3l`U--HtfqD8TZ91W4_B0;(bzhe-=;*jYeQv<p-a8xDiuYz6|s!yvTlf{%fHw=)(PE$IOsN)~SmR9c3Y-|L&LB?<N%ODEU(E>VKJsX$8mhIhid@<nLhBSl~sFVV{Q+$qa3+Pcp@yuRS$dZU^yOMhZATSj-jZOp_v&Ti|LX@lyNcB%|zyQT!&agf2*DNNdv!GwD}8)$l#vHH~D&?e6Y9uH*Tso|kbP6kgwA`*u%@=|4|G$R3BvwZ^o&-$k7v2JEX76v#%7KM4oUxYjGl^)*yn%n0`iUwHaPi5V~`gYr)W%1=mGKG^3(moKLF}g44zzo3E*&~@1a{Gv>7k$@y{;j;)!J7gmWN)%|HTrN3KB#bwDqG~{!lI8-$U2Vg&#B5q$MziXe&9;=dZn6zS9ZXO@Njw<9=@X+Y?2QE)*mi^2YiBcB*3B2MBxV%E!<>yNA^6@eV5eMApB5@Spb=cFj$DVBX1L@p^hGuK|OEUZB=;zuId9qPW_&*UI8oSod@uRtFs7K0K?)ff>~?z!LsEbCvOtz5Uz<B1zgZQWKp0ga@8QbTXrr)8{G9(rC!sNoD4}C?HyamKbv=!XX~?dP19!U(^|;{&&KCNR-QV3X`|!;h;GR!Y84nR6Ve?LpF<Hg=HtkWl!8h2j&r(j&J&p&T$uVV5*Py!dwd#v8{AM5-apy^rCJQa#4_EL;rAS*&B_zNC+i*f!0?p>gaW8w@9`mT^<9_SKI!2nxk7=^yO=x3Qq*!5Iww%Pf5(-wGR1XIp-t%o*dy|mhif}LCqdlVGHrLRQi{%Fej*>dZF<JMRpR|cusxxoOhYK}!zH@_{xF8j_bI5jNAB?l8Rf`Sh^hvdRza=;wTS5qRE+R?x1c4#z_CK0mDhn$yvwfxE6H)!5?=+(({K&md#hcz^fGtlFfpS%zaS+sAa33i>N41#5jK5`o%u}!`$L$54k^O9A#A>^`!7xf-&W8b?h3QGm!BeOezTLp*K*pN&CTICj!VFsHd7~Ct1b}7Z0X#atg^}Q#-x15tQx1i^aXBn7l;pr1T0B!_5696`1s>;zwg>inTH5%DW^!Y4uMF_k*3^8Fmd+G$uRN9so=|#VuD*`xPX(&anVIw1bdH#iz|lb^QS-tYDD4TY%uk8@IERDM$Ty`!^m4x!GE3(BXCkFMpg~tj(d-V5xm6l2q}?xIr7Hz6lGA+GNj22c+QSVnPk&5`Lqd*f&jnWtF`%cv&BI*x(qX;!}5v{Sa+jT&$~l14sR>h4Xe*R<z3i0N=8b;NF!wkOfnXc$x^iveX1&$$_yTX=GsPoF+oPSKEo975>?QyO6^aMX{#bHk(>$C_Bf+3KD-IBNNK7npLJkU!kfts9)t93YNCL304Pb@fu2o1)J#uzdfd(-XCe#123&j(^V}sU#7$b3<8if8{xnJxmOT`i3V^v|E~rnG@hDBcWH&6!>y+&vC>`a^7-o3EP_2dQNVqadlHCGIi_hhFYJvMQPJjfq<I~g}w$q$@4Xoetvf<jMy&&?n$Xeh(f^s4{?50`9ZINM)c=31`eRp+cD1{=?6FFPr;wnpW>{Ti?IDLrQa;M;U&9gY~sf`Np4W-eOc#5!iAH9L#0di4XEs?V05spQ)nWRil$}J^~B4kor0Xb&D<=lQup7}a2en)dm9t2SS2@sWGrYDlK?}NXAhw?3If`7o<zXo3hAA<psYC;T5&_Se{56}k_c)A%Uoui->BlJTGV%MnXwI59mZ&N`EgT$!Zh7qvP<Y|2^kMg9RRJ%s+RNrucp@o;P<A`oKn5b+UyzAH^Vn-Ue8K&neE#C9lWqL6k8>><LX!hr>>$np0e+3Ew_-Bi_AM^TmGK?ctcV-|?1gv|$tMbZm+Nfauih&y?Xfo9u_;7s5Ak9Gu50=6p>CX5Wz`z$km}~Y*v;;n!DVONnO!Y{;UU^(qljy&q9u7YV_J^sN<CRQp*)_pEQR(*2GX-P%8n@8_QWYvQE0v)_cK$rc6Jhf4m`f7WuY-SqfPDhe_HiLyAe2VJIVWHaD&)!KpCikgIqrbizRX1b@6mH2iz!ev=n7!*l*Z8`qAVIF3S_mkT@`u<3xZfe5V@KR6r1E+l2B)5!;38`xwmLVMFNW{l<=;3J<IfIqAWN^E36pn`m*CLarel`8`gPq#HU`hGEBiqfopDWN$NeBT<GQ!^Mj{L4|AkEQY{><8blh#_D#EA>SrYiDJMWV6P}j7E4f;ChMK~}lWjUg`p4<X{qP=&MwtUmS)8oMZ3q`jWMe;!;xCw|oJjmKDp@d8A#ufuawlh|iUQbP$#9-&xA}^yh$L#vcdAlTDsOA=3Iq|~Q&v`11O-)}<z_qw6d<v5{GK?wP-4%KEsvJ(CAVN(vh$+2Cfj+H`f&zAcDE=80)b5ULFKrxBDM4dbS*EM7Jsa$qG%k%eeevz6u|lbpxhF!B1WK?^v3J$rY_|^Isut_38>!5cWelhsq2G^7b2_wP^cH!6bm1KhnN5GTm8ywzkBuCOH$KE5u~=^8!_4NIx&EGiHxKYr_YpYl&C6#Ga%fRja`+IqvK&n)3b~gM-D3!d+8wDZDzRYn#f8@qO3*wROoD}<cj$`k)qfy^(#vnCWfD#%K8g0kS#@6b2rJ`>yoQba-zJUL%@aZDuxvYq?~S0V02GVTJ(F}p^J(uKT&H@%=&m4p~}17LbmY(d~_jMTdZbuu%Sxg&W$Vq(Dq`$CMt^isf1E99A=l2%u<w8O)HH^8qjH2m_LqM*F<MmEd%Di@v%_PSiC2lK&SpCk}!0lQKQO0h%TjdEPqB7!IH+AB!Y`-n45HjNTbEhK&){@88y*44A+tQ7jh{gzdd;LVKAtb^&=6ARV^--VT!wN>ir%_6&KXW*yGIeauQ}z<x%JtW!7!S?Hbn5yn0+x6ZL!8HuhzQ%=b7KlPf1WUC!%M10ME&*+!S^I;IH~UGEyEt?Np3nW*v0U#eYOiWP^^!v>?YmrRukU4&2Ef>b&TPEJma8VY~bhQd)l;i#W*)K56-Cmi(?j`|5l{e+`_!e3cG;WKRZdnO%zqb|8om)xjJZqz0BOLfVOdS*sFGozlFQP0e%XXYpCnVE?4`_xl>Ggh_lpS1clYHS%bwu~BEMvX0_#+FfI%c!yC|Esa(mv3MBmG+AKwEY^N|BUSzqgsAk%f~-d&yQZFrWj=!r!`H(!KvrqpPYJZOw%w7k&MhXH5_e<*}1v(X)PL&2XaH2R$P~ki9_lQ98Px2!D6`G+36+#ak^P+Hnn7|S%)d-q@w(fYo7`Ih+o>?h+G#&-Cws!dA>1sTGQB@b6Cn(3kCdX%v#XSr}1zBXu7j#oOyr>2hDq7xqMSZZ$>C+gB#S+a$97A2EjviMMD6u)X6-s>8M-}W@tayigs$0?v#Rp3mch*;W~lQUC_Zx%>59CHghQvOo+!Va^qvI;%W3%O*mqgzWiCX2v0*T$4V|O&WY#&Al`?=v4x1e@upaasr1=%gzedS3g$u45aMEN{5z*rOYvWo!MerjV*jZ8nFFlDd3)YyEeT41>g>0`A@z9|E2ED49wz&|9!vsFOP;$+;xy%PUkBHO&o=Wny%l|hQqKNSYa;$nWldPx0gsKl(H%1+!4|OxG0Ps3qqrUZnJ(bjvyIGI5@v5wN&uaSMm0JL-Zu%(>!t%X2PC(!F2*7<4d8DxVrKGd4uwrm*P|emFK?C+0l)hQ&jBC}Q(d%E%dpIO{32rE+B<a90e!FdAo7${`ajdix1g^K-^I?f@Lol1RJ2-0Y=oV4!#0EMEVJY5{a(p-hfz3uFV5g5$O&(y3uiMOLFEgJdLq0FxY6QW+Rn0FIQ}4?*C+)5my(uh3KcPQX@)=zl9DaN7P(MSL=!!2BFV(|SlTROIXoh|X!sF?ReP9a!avO>Yo24Lr<pJU&rK1Mrx+{K^Z6%dB)bsVtMRe_0y-6Ns{"""


def load_embedded_emma():
    module = types.ModuleType("original_emma")
    sys.modules[module.__name__] = module
    source = zlib.decompress(base64.b85decode(EMMA_ORIGINAL_B85)).decode("utf-8")
    exec(compile(source, "original_emma.py", "exec"), module.__dict__)
    return module


ORIGINAL_EMMA = load_embedded_emma()

OPEN_STRINGS = {4: 55, 3: 62, 2: 69, 1: 76}
STRING_NAMES = {1: "Mi", 2: "La", 3: "Re", 4: "Sol"}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DURATIONS = (250, 500, 750, 1000, 1250, 1500, 1750, 2000)
BOW_SPEEDS = {250: 200, 500: 100, 750: 90, 1000: 80, 1250: 75, 1500: 70, 1750: 65, 2000: 60}
WHITE_NOTES = {0, 2, 4, 5, 7, 9, 11}
MIDI_MIN = OPEN_STRINGS[4]
MIDI_MAX = OPEN_STRINGS[1] + 7


def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_freq(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def speed_for_duration(duration_ms: int) -> int:
    return BOW_SPEEDS[min(BOW_SPEEDS, key=lambda value: abs(value - duration_ms))]


def playable_positions(midi: int) -> list[tuple[int, int]]:
    return [(string, midi - open_midi) for string, open_midi in OPEN_STRINGS.items() if 0 <= midi - open_midi <= 7]


def choose_position(midi: int, previous_string: int | None) -> tuple[int, int]:
    choices = playable_positions(midi)
    if not choices:
        raise ValueError(f"{midi_name(midi)} вне диапазона G3-B5")
    return min(choices, key=lambda item: (0 if item[0] == previous_string else 10, item[1], item[0]))


def parse_last_state(text: str) -> tuple[int | None, int]:
    last_string: int | None = None
    last_direction: int | None = None
    for raw in text.splitlines():
        parts = raw.strip().lower().split(";")
        try:
            if len(parts) >= 3 and parts[0] == "/sn":
                last_string = int(parts[1])
            elif len(parts) >= 4 and parts[0] == "/bow":
                last_direction = int(parts[1])
        except ValueError:
            pass
    return last_string, 1 if last_direction is None else 1 - last_direction


@dataclass(frozen=True)
class PlayEvent:
    line: int
    string: int
    position: int
    duration_ms: int

    @property
    def midi(self) -> int:
        return OPEN_STRINGS[self.string] + self.position


def parse_events(text: str) -> list[PlayEvent]:
    events: list[PlayEvent] = []
    current_note: tuple[int, int] | None = None
    for line_number, raw in enumerate(text.splitlines(), 1):
        parts = raw.strip().lower().split(";")
        try:
            if len(parts) >= 3 and parts[0] == "/sn":
                candidate = (int(parts[1]), int(parts[2]))
                current_note = candidate if candidate[0] in OPEN_STRINGS and 0 <= candidate[1] <= 7 else None
            elif len(parts) >= 4 and parts[0] == "/bow" and current_note:
                direction, duration_ms, speed = map(int, parts[1:4])
                if direction in (0, 1) and duration_ms > 0 and speed > 0:
                    events.append(PlayEvent(line_number, current_note[0], current_note[1], duration_ms))
        except ValueError:
            continue
    return events


def make_tone(midi: int, duration_ms: int, sample_rate: int = 44_100) -> np.ndarray:
    count = max(1, int(sample_rate * duration_ms / 1000))
    t = np.arange(count, dtype=np.float32) / sample_rate
    freq = midi_to_freq(midi)
    wave = .75 * np.sin(2 * np.pi * freq * t) + .25 * np.sin(4 * np.pi * freq * t) + .1 * np.sin(6 * np.pi * freq * t)
    attack, release = max(1, int(sample_rate * .015)), max(1, int(sample_rate * min(.045, duration_ms / 4000)))
    envelope = np.ones(count, dtype=np.float32)
    envelope[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    envelope[-release:] = np.linspace(1, 0, release, dtype=np.float32)
    return (wave * envelope * .18).astype(np.float32)


class SyncClient:
    """Постоянное подключение; сервер хранит только текущий текст сценария."""

    def __init__(self, root: tk.Tk, receive, set_status) -> None:
        self.root, self.receive, self.set_status = root, receive, set_status
        self.client_id = uuid.uuid4().hex
        self.permissions: set[str] = set()
        self.device_key = self._device_key()
        self.device_number = None
        self.ws = None
        self.stopped = threading.Event()
        self.send_lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="scenario-sync").start()

    @staticmethod
    def _device_key() -> str:
        path = Path(os.environ.get("APPDATA", str(Path.home()))) / "EMMA Control" / "device.json"
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved.get("key"), str):
                return saved["key"]
        except (OSError, ValueError, AttributeError):
            pass
        key = uuid.uuid4().hex + uuid.uuid4().hex
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key}), encoding="utf-8")
        return key

    def _ui_status(self, text: str, ok: bool) -> None:
        self.root.after(0, self.set_status, text, ok)

    def _run(self) -> None:
        while not self.stopped.is_set():
            app = websocket.WebSocketApp(SERVER_URL, on_open=self._open, on_message=self._message, on_close=self._close, on_error=self._error)
            self.ws = app
            try:
                app.run_forever(ping_interval=25, ping_timeout=10)
            except Exception:
                pass
            if not self.stopped.wait(2):
                self._ui_status("Синхронизация: переподключение…", False)

    def _open(self, ws) -> None:
        try:
            ws.send(json.dumps({"type": "hello", "token": ROOM_TOKEN, "client_id": self.client_id,
                                "device_key": self.device_key, "device_name": socket.gethostname()}))
            self._ui_status("Синхронизация: подключено", True)
        except Exception:
            pass

    def _message(self, _ws, raw: str) -> None:
        try:
            message = json.loads(raw)
            if isinstance(message, dict):
                if message.get("type") in {"welcome", "permissions"}:
                    self.permissions = set(message.get("permissions", ()))
                    self.device_number = message.get("number", self.device_number)
                self.root.after(0, self.receive, message)
        except (TypeError, ValueError):
            pass

    def _close(self, *_args) -> None:
        if not self.stopped.is_set():
            self._ui_status("Синхронизация: нет связи", False)

    def _error(self, *_args) -> None:
        pass

    def send(self, message: dict) -> bool:
        payload = dict(message, client_id=self.client_id)
        try:
            with self.send_lock:
                if self.ws and self.ws.sock and self.ws.sock.connected:
                    self.ws.send(json.dumps(payload, ensure_ascii=False))
                    return True
        except Exception:
            pass
        return False

    def allowed(self, permission: str) -> bool:
        return True

    def stop(self) -> None:
        self.stopped.set()
        if self.ws:
            self.ws.close()


class EmbeddedRoot(tk.Frame):
    """Настоящий Tk-контейнер с no-op методами окна для неизменённых оригиналов."""

    def title(self, *_args):
        pass

    def geometry(self, *_args):
        pass

    def minsize(self, *_args):
        pass

    def protocol(self, *_args):
        pass


class SyncedEmmaSender(ORIGINAL_EMMA.EmmaSender):
    def __init__(self, root, sync: SyncClient):
        self.sync = sync
        self.receiving_log = False
        super().__init__(root)
        self.last_text = self.text.get("1.0", "end-1c")
        self.root.after(400, self.poll)

    def poll(self) -> None:
        text = self.text.get("1.0", "end-1c")
        if text != self.last_text and self.sync.allowed("sender_edit"):
            self.last_text = text
            self.sync.send({"type": "state", "scope": "sender", "text": text})
        self.root.after(400, self.poll)

    def publish_text(self) -> None:
        self.last_text = self.text.get("1.0", "end-1c")
        if self.sync.allowed("sender_edit"):
            self.sync.send({"type": "state", "scope": "sender", "text": self.last_text})

    def apply_remote_text(self, text: str) -> None:
        if text == self.text.get("1.0", "end-1c"):
            return
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.text.edit_modified(False)
        self.last_text = text

    def log(self, message):
        ORIGINAL_EMMA.EmmaSender.log(self, message)
        if not self.receiving_log and self.sync.allowed("log_send"):
            self.sync.send({"type": "log", "text": str(message)})

    def receive_log(self, text: str) -> None:
        self.receiving_log = True
        try:
            ORIGINAL_EMMA.EmmaSender.log(self, "[Другой ПК] " + text)
        finally:
            self.receiving_log = False

    def send_line(self, line):
        line = line.strip()
        if not line:
            return
        if not self.sync.allowed("robot_send"):
            self.log("Доступ к отправке на робота запрещён администратором")
            return
        if self.ser and self.ser.is_open:
            ORIGINAL_EMMA.EmmaSender.send_line(self, line)
        else:
            self.log("Робот на этом ПК не подключён — команда отправлена по сети")
        if not self.sync.send({"type": "command", "kind": "line", "line": line}):
            self.log("Нет связи с сервером: команда отправлена только локально")

    def send_all(self):
        if not self.sync.allowed("robot_send"):
            self.log("Доступ к отправке на робота запрещён администратором")
            return
        lines = [line.strip() for line in self.text.get("1.0", "end-1c").splitlines() if line.strip()]
        if self.ser and self.ser.is_open:
            threading.Thread(target=self._send_lines_local, args=(lines,), daemon=True).start()
        else:
            self.log("Робот на этом ПК не подключён — сценарий отправлен по сети")
        if not self.sync.send({"type": "command", "kind": "all", "lines": lines}):
            self.log("Нет связи с сервером: сценарий отправлен только локально")

    def _send_lines_local(self, lines):
        for line in lines:
            ORIGINAL_EMMA.EmmaSender.send_line(self, line)
            time.sleep(SEND_DELAY)
        self.log("--- отправка завершена ---")

    def receive_command(self, message: dict) -> None:
        if not (self.ser and self.ser.is_open):
            self.log("Команда с другого ПК: робот здесь не подключён")
            return
        if message.get("kind") == "line":
            ORIGINAL_EMMA.EmmaSender.send_line(self, str(message.get("line", "")))
        elif message.get("kind") == "all" and isinstance(message.get("lines"), list):
            threading.Thread(target=self._send_lines_local, args=([str(line) for line in message["lines"]],), daemon=True).start()

    def shutdown(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()


class SerialTab:
    def __init__(self, root: tk.Tk, parent: ttk.Frame, sync: SyncClient) -> None:
        self.root, self.sync = root, sync
        self.ser = None
        self.serial_lock = threading.Lock()
        self._build(parent)
        self.refresh_ports()

    def _build(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="Bluetooth / Serial порт:").pack(side="left")
        self.port_box = ttk.Combobox(top, width=20, state="readonly")
        self.port_box.pack(side="left", padx=6)
        ttk.Button(top, text="Обновить", command=self.refresh_ports).pack(side="left")
        self.connect_button = ttk.Button(top, text="Подключить", command=self.toggle_connection)
        self.connect_button.pack(side="left", padx=6)
        self.status = ttk.Label(top, text="Робот не подключён", foreground="#a11")
        self.status.pack(side="left", padx=8)

        quick = ttk.LabelFrame(parent, text="Управление очередью", padding=8)
        quick.pack(fill="x", padx=12, pady=(0, 8))
        for command in ("/load_test", "/end", "/play", "/stop"):
            ttk.Button(quick, text=command, command=lambda value=command: self.local_line(value)).pack(side="left", padx=3)

        editor_frame = ttk.LabelFrame(parent, text="Команды (по одной на строку)", padding=6)
        editor_frame.pack(fill="both", expand=True, padx=12)
        self.text = scrolledtext.ScrolledText(editor_frame, height=15, font=("Cascadia Mono", 11), undo=True)
        self.text.pack(fill="both", expand=True)
        self.suppress_sync = False
        self.sync_job = None
        self.text.insert("1.0", "/load_test\n/bow;1;1000;150\n/sn;1;1\n/bow;0;1000;150\n/end\n/play\n")
        self.text.edit_modified(False)
        self.text.bind("<<Modified>>", self.on_text_changed)
        buttons = ttk.Frame(parent, padding=12)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Отправить всё", command=self.local_all).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(buttons, text="Отправить строку под курсором", command=self.local_current_line).pack(side="left", expand=True, fill="x", padx=3)
        log_frame = ttk.LabelFrame(parent, text="Лог", padding=6)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log_box = scrolledtext.ScrolledText(log_frame, height=7, font=("Cascadia Mono", 9), state="disabled", bg="#fbfcfd", fg="#121417", relief="flat")
        self.log_box.pack(fill="both", expand=True)
        
        self.log_box.tag_configure("recv", foreground="#2e7d32")
        self.log_box.tag_configure("send", foreground="#0b2c38", font=("Cascadia Mono", 9, "bold"))
        self.log_box.tag_configure("error", foreground="#d32f2f", font=("Cascadia Mono", 9, "bold"))
        self.log_box.tag_configure("info", foreground="#1c4f63")

    def on_text_changed(self, _event=None) -> None:
        if not self.text.edit_modified():
            return
        self.text.edit_modified(False)
        if self.suppress_sync:
            return
        if self.sync_job:
            self.root.after_cancel(self.sync_job)
        self.sync_job = self.root.after(350, self.publish_text)

    def publish_text(self) -> None:
        self.sync_job = None
        self.sync.send({"type": "state", "scope": "sender", "text": self.text.get("1.0", "end-1c")})

    def apply_remote_text(self, text: str) -> None:
        if text == self.text.get("1.0", "end-1c"):
            return
        self.suppress_sync = True
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.text.edit_modified(False)
        self.suppress_sync = False

    def refresh_ports(self) -> None:
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_box["values"] = ports
        if ports:
            self.port_box.current(0)
        self.log("Найдены порты: " + (", ".join(ports) or "нет"))

    def toggle_connection(self) -> None:
        if self.ser and self.ser.is_open:
            self.disconnect()
            return
        port = self.port_box.get()
        if not port:
            messagebox.showwarning("EMMA", "Сначала выберите порт.")
            return
        try:
            self.ser = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)
            self.status.configure(text=f"Робот подключён: {port}", foreground="#187a2f")
            self.connect_button.configure(text="Отключить")
            self.log("Подключено к " + port)
            threading.Thread(target=self._read_loop, args=(self.ser,), daemon=True).start()
        except Exception as exc:
            self.ser = None
            messagebox.showerror("EMMA", f"Не удалось открыть порт:\n{exc}")

    def disconnect(self) -> None:
        ser, self.ser = self.ser, None
        if ser:
            try:
                ser.close()
            except Exception:
                pass
        self.status.configure(text="Робот не подключён", foreground="#a11")
        self.connect_button.configure(text="Подключить")
        self.log("Порт закрыт")

    def is_connected(self) -> bool:
        return bool(self.ser and self.ser.is_open)

    def _write(self, line: str, source: str) -> None:
        line = line.strip()
        if not line:
            return
        ser = self.ser
        if not ser or not ser.is_open:
            self.log(f"{source}: робот здесь не подключён — команда ждёт другого ПК")
            return
        try:
            with self.serial_lock:
                ser.write((line + "\n").encode("utf-8"))
            self.log(f">> {line}" + (" (из сети)" if source == "Сеть" else ""))
        except Exception as exc:
            self.log("Ошибка отправки: " + str(exc))

    def local_line(self, line: str) -> None:
        self._write(line, "Этот ПК")
        if not self.sync.send({"type": "command", "kind": "line", "line": line}):
            self.log("Нет связи с сервером: команда отправлена только локально")

    def local_current_line(self) -> None:
        line_number = self.text.index("insert").split(".")[0]
        self.local_line(self.text.get(line_number + ".0", line_number + ".end"))

    def local_all(self) -> None:
        lines = [line.strip() for line in self.text.get("1.0", "end-1c").splitlines() if line.strip()]
        self._send_all(lines, "Этот ПК")
        if not self.sync.send({"type": "command", "kind": "all", "lines": lines}):
            self.log("Нет связи с сервером: сценарий отправлен только локально")

    def receive_command(self, message: dict) -> None:
        if message.get("kind") == "line":
            self._write(str(message.get("line", "")), "Сеть")
        elif message.get("kind") == "all" and isinstance(message.get("lines"), list):
            lines = [str(line) for line in message["lines"]]
            self._send_all(lines, "Сеть")

    def _send_all(self, lines: list[str], source: str) -> None:
        threading.Thread(target=self._send_all_worker, args=(lines, source), daemon=True).start()

    def _send_all_worker(self, lines: list[str], source: str) -> None:
        for line in lines:
            self._write(line, source)
            time.sleep(SEND_DELAY)
        self.log(f"--- отправка завершена ({source.lower()}) ---")

    def _read_loop(self, ser) -> None:
        while ser is self.ser and ser.is_open:
            try:
                data = ser.readline()
                if data:
                    text = data.decode("utf-8", errors="replace").strip()
                    if text:
                        self.log("<< " + text)
            except Exception:
                break

    def log(self, text: str) -> None:
        self.root.after(0, self._log_ui, text)

    def _log_ui(self, text: str) -> None:
        tag = "info"
        if "<< " in text or "[Другой ПК]" in text or "(из сети)" in text:
            tag = "recv"
        elif ">> " in text or "отправка завершена" in text:
            tag = "send"
        if "Ошибка" in text or "не подключён" in text or "Нет связи" in text:
            tag = "error"
            
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")


class ScenarioTab:
    def __init__(self, root: tk.Tk, parent: ttk.Frame, sync: SyncClient) -> None:
        self.root, self.sync, self.path = root, sync, None
        self.stop_flag = threading.Event()
        self.play_thread = None
        self.suppress_sync = False
        self.sync_job = None
        self.duration = tk.IntVar(value=500)
        self.status = tk.StringVar(value="Готов")
        self.choice = tk.StringVar(value="Нажмите клавишу, чтобы добавить /sn и /bow")
        self.key_rects: dict[int, int] = {}
        self.key_fills: dict[int, str] = {}
        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent, padding=(12, 12, 12, 6))
        toolbar.pack(fill="x")
        for label, command in (("Новый", self.new_file), ("Открыть", self.open_file), ("Сохранить", self.save_file), ("Вставить заголовок", self.insert_header), ("Добавить /end + /play", self.ensure_footer), ("Играть всё", self.play_all), ("Играть с курсора", self.play_from_cursor), ("Стоп", self.stop)):
            ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=3)
        self.editor = scrolledtext.ScrolledText(parent, wrap="none", font=("Cascadia Mono", 11), undo=True)
        self.editor.pack(fill="both", expand=True, padx=12, pady=6)
        self.editor.insert("1.0", "New scenario\n/load_test\n/end\n/play\n")
        self.editor.mark_set("insert", "3.0")
        self.editor.edit_modified(False)
        self.editor.bind("<<Modified>>", self.on_text_changed)
        controls = ttk.LabelFrame(parent, text="Автоматически добавить ноту", padding=8)
        controls.pack(fill="x", padx=12, pady=6)
        ttk.Label(controls, text="Длина:").pack(side="left")
        for duration in DURATIONS:
            ttk.Radiobutton(controls, text=str(duration), variable=self.duration, value=duration).pack(side="left", padx=2)
        piano_frame = ttk.LabelFrame(parent, text="Пианино — добавить ноту в сценарий", padding=(8, 8, 8, 5))
        piano_frame.pack(fill="x", padx=12, pady=6)
        self.keyboard = tk.Canvas(piano_frame, height=180, bg="#ffffff", highlightthickness=0)
        self.keyboard.pack(fill="x", expand=True)
        self.keyboard.bind("<Configure>", lambda _event: self.draw_keyboard())
        ttk.Label(piano_frame, textvariable=self.choice).pack(anchor="w", pady=(5, 0))
        footer = ttk.Frame(parent, padding=(12, 0, 12, 12))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status).pack(side="left")

    def draw_keyboard(self) -> None:
        self.keyboard.delete("all")
        self.key_rects.clear()
        self.key_fills.clear()
        width, height = max(1, self.keyboard.winfo_width()), max(1, self.keyboard.winfo_height())
        white_midis = [midi for midi in range(MIDI_MIN, MIDI_MAX + 1) if midi % 12 in WHITE_NOTES]
        white_width = width / len(white_midis)
        white_x: dict[int, float] = {}
        for index, midi in enumerate(white_midis):
            x0, x1 = index * white_width, (index + 1) * white_width
            white_x[midi] = x0
            self._make_key(midi, x0, 0, x1, height - 4, "#fbfcfd", "#ccd4dc", midi_name(midi))
        for midi in range(MIDI_MIN, MIDI_MAX + 1):
            if midi % 12 in WHITE_NOTES:
                continue
            previous = midi - 1
            while previous >= MIDI_MIN and previous % 12 not in WHITE_NOTES:
                previous -= 1
            if previous in white_x:
                black_width = white_width * .62
                x0 = white_x[previous] + white_width - black_width / 2
                self._make_key(midi, x0, 0, x0 + black_width, height * .62, "#121417", "#121417", "")

    def _make_key(self, midi: int, x0: float, y0: float, x1: float, y1: float, fill: str, outline: str, label: str) -> None:
        tag = f"piano_{midi}"
        rect = self.keyboard.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, tags=(tag,))
        self.key_rects[midi], self.key_fills[midi] = rect, fill
        if label:
            self.keyboard.create_text((x0 + x1) / 2, y1 - 18, text=label, fill="#4d5660", font=("Segoe UI", 8), tags=(tag,))
        self.keyboard.tag_bind(tag, "<Button-1>", lambda _event, value=midi: self.add_note(value))

    def flash_key(self, midi: int) -> None:
        rect = self.key_rects.get(midi)
        if rect is None:
            return
        base = self.key_fills[midi]
        self.keyboard.itemconfigure(rect, fill="#d7edf4" if base != "#121417" else "#2c7892")
        self.root.after(100, lambda: self.keyboard.itemconfigure(rect, fill=base))

    def on_text_changed(self, _event=None) -> None:
        if not self.editor.edit_modified():
            return
        self.editor.edit_modified(False)
        if self.suppress_sync:
            return
        if self.sync_job:
            self.root.after_cancel(self.sync_job)
        self.sync_job = self.root.after(350, self.publish_text)

    def publish_text(self) -> None:
        self.sync_job = None
        text = self.editor.get("1.0", "end-1c")
        if self.sync.send({"type": "state", "scope": "scenario", "text": text}):
            self.status.set("Сценарий синхронизирован")
        else:
            self.status.set("Нет связи: изменения пока только на этом ПК")

    def apply_remote_text(self, text: str) -> None:
        if text == self.editor.get("1.0", "end-1c"):
            return
        self.suppress_sync = True
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", text)
        self.editor.edit_modified(False)
        self.suppress_sync = False
        self.status.set("Сценарий обновлён с другого ПК")

    def add_note(self, midi: int) -> None:
        self.flash_key(midi)
        duration = self.duration.get()
        previous_string, direction = parse_last_state(self.editor.get("1.0", "insert"))
        string, position = choose_position(midi, previous_string)
        block = f"/sn;{string};{position}\n/bow;{direction};{duration};{speed_for_duration(duration)}\n"
        footer = self.footer_start_index()
        if self.editor.compare("insert", ">=", footer):
            self.editor.mark_set("insert", footer)
        self.editor.edit_separator()
        self.editor.insert("insert", block)
        self.editor.edit_separator()
        self.choice.set(f"{midi_name(midi)} → /sn;{string};{position}, /bow;{direction};{duration};{speed_for_duration(duration)}")
        self.status.set(f"Добавлено: {midi_name(midi)}")

    def footer_start_index(self) -> str:
        lines = self.editor.get("1.0", "end-1c").splitlines()
        if len(lines) >= 2 and [x.strip().lower() for x in lines[-2:]] == ["/end", "/play"]:
            return f"{len(lines) - 1}.0"
        return "end-1c"

    def insert_header(self) -> None:
        self.editor.insert("insert", "Scenario\n/load_test\n")

    def ensure_footer(self) -> None:
        lines = self.editor.get("1.0", "end-1c").splitlines()
        while lines and lines[-1].strip().lower() in {"/end", "/play"}:
            lines.pop()
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", "\n".join(lines).rstrip() + "\n/end\n/play\n")

    def new_file(self) -> None:
        self.path = None
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", "New scenario\n/load_test\n/end\n/play\n")

    def open_file(self) -> None:
        filename = filedialog.askopenfilename(title="Открыть сценарий", filetypes=(("Scenario files", "*.txt *.scenario *.scr"), ("All files", "*.*")))
        if filename:
            self.path = Path(filename)
            self.apply_remote_text(self.path.read_text(encoding="utf-8"))
            self.publish_text()

    def save_file(self) -> None:
        self.ensure_footer()
        if not self.path:
            name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", self.editor.get("1.0", "1.end")).strip() or "scenario"
            filename = filedialog.asksaveasfilename(title="Сохранить сценарий", defaultextension=".txt", initialfile=name[:80] + ".txt")
            if not filename:
                return
            self.path = Path(filename)
        self.path.write_text(self.editor.get("1.0", "end-1c"), encoding="utf-8")
        self.status.set("Сохранено: " + self.path.name)

    def play_all(self) -> None:
        self.play_from_line(1)

    def play_from_cursor(self) -> None:
        self.play_from_line(int(self.editor.index("insert").split(".")[0]))

    def play_from_line(self, start_line: int) -> None:
        if self.play_thread and self.play_thread.is_alive():
            return
        events = [event for event in parse_events(self.editor.get("1.0", "end-1c")) if event.line >= start_line]
        if not events:
            messagebox.showinfo("Сценарий", "Нет корректных пар /sn + /bow.")
            return
        self.stop_flag.clear()
        self.play_thread = threading.Thread(target=self._play, args=(events,), daemon=True)
        self.play_thread.start()

    def _play(self, events: list[PlayEvent]) -> None:
        try:
            for event in events:
                if self.stop_flag.is_set():
                    break
                sd.play(make_tone(event.midi, event.duration_ms), 44_100, blocking=True)
        finally:
            sd.stop()
            self.root.after(0, self.status.set, "Готов")

    def stop(self) -> None:
        self.stop_flag.set()
        sd.stop()


class EmbeddedScenarioTab:
    """Точная исходная вкладка редактора; синхронизация не меняет её интерфейс."""

    def __init__(self, root: tk.Tk, parent: ttk.Frame, sync: SyncClient) -> None:
        self.root, self.sync = root, sync
        host = EmbeddedRoot(parent)
        host.pack(fill="both", expand=True)
        self.editor_app = ORIGINAL_SCENARIO.ScenarioEditorApp(host)
        self.editor = self.editor_app.editor
        self.last_text = self.editor.get("1.0", "end-1c")
        self.receiving = False
        self.root.after(400, self.poll)

    def poll(self) -> None:
        text = self.editor.get("1.0", "end-1c")
        if not self.receiving and text != self.last_text and self.sync.allowed("scenario_edit"):
            self.last_text = text
            self.sync.send({"type": "state", "scope": "scenario", "text": text})
        self.root.after(400, self.poll)

    def publish_text(self) -> None:
        self.last_text = self.editor.get("1.0", "end-1c")
        if self.sync.allowed("scenario_edit"):
            self.sync.send({"type": "state", "scope": "scenario", "text": self.last_text})

    def apply_remote_text(self, text: str) -> None:
        if text == self.editor.get("1.0", "end-1c"):
            return
        self.receiving = True
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", text)
        self.editor.edit_modified(False)
        self.last_text = text
        self.editor_app.status_var.set("Updated from another PC")
        self.receiving = False

    def stop(self) -> None:
        self.editor_app.stop()


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("EMMA — управление и сценарий")
        root.geometry("1180x780")
        root.minsize(900, 620)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.sync_status = tk.StringVar(value="Синхронизация: подключение…")
        top = ttk.Frame(root, padding=(12, 8, 12, 0))
        top.pack(fill="x")
        ttk.Label(top, text="EMMA", font=("Segoe UI", 15, "bold")).pack(side="left")
        
        def open_emma_studio():
            import subprocess
            import sys
            subprocess.Popen([sys.executable, str(Path(__file__).with_name("emma_studio.py"))])
            
        ttk.Button(top, text="Открыть EMMA Studio", command=open_emma_studio).pack(side="left", padx=20)
        
        self.sync_label = ttk.Label(top, textvariable=self.sync_status, foreground="#9a6800")
        self.sync_label.pack(side="right")
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
        serial_page, scenario_page = ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        self.tabs.add(serial_page, text="  Робот / BT Serial  ")
        self.tabs.add(scenario_page, text="  Сценарий  ")
        self.sync = SyncClient(root, self.receive_network, self.set_sync_status)
        serial_host = EmbeddedRoot(serial_page)
        serial_host.pack(fill="both", expand=True)
        self.serial_tab = SyncedEmmaSender(serial_host, self.sync)
        self.scenario_tab = EmbeddedScenarioTab(root, scenario_page, self.sync)

    def set_sync_status(self, text: str, ok: bool) -> None:
        self.sync_status.set(text)
        self.sync_label.configure(foreground="#187a2f" if ok else "#9a6800")

    def receive_network(self, message: dict) -> None:
        kind = message.get("type")
        if kind in {"welcome", "permissions", "snapshot"}:
            state = message.get("snapshot") if kind != "snapshot" else message.get("text")
            if isinstance(state, dict):
                if isinstance(state.get("scenario"), str):
                    self.scenario_tab.apply_remote_text(state["scenario"])
                if isinstance(state.get("sender"), str):
                    self.serial_tab.apply_remote_text(state["sender"])
            else:
                self.scenario_tab.publish_text()
                self.serial_tab.publish_text()
            if kind == "welcome":
                self.set_sync_status(f"Синхронизация: подключено, ПК №{message.get('number')}", True)
        elif kind == "state":
            text = message.get("text")
            if message.get("scope") == "scenario" and isinstance(text, str):
                self.scenario_tab.apply_remote_text(text)
            elif message.get("scope") == "sender" and isinstance(text, str):
                self.serial_tab.apply_remote_text(text)
        elif kind == "command":
            self.serial_tab.receive_command(message)
        elif kind == "log" and isinstance(message.get("text"), str):
            self.serial_tab.receive_log(message["text"])

    def close(self) -> None:
        self.scenario_tab.stop()
        self.serial_tab.shutdown()
        self.sync.stop()
        self.root.destroy()


def run_self_test() -> int:
    assert choose_position(76, None) == (1, 0)
    assert speed_for_duration(500) == 100
    assert len(parse_events("/sn;2;0\n/bow;1;500;100\n")) == 1
    print("OK")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(run_self_test())
    window = tk.Tk()
    App(window)
    window.mainloop()
