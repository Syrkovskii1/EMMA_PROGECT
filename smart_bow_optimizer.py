from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import heapq

# ================= НАСТРОЙКИ ПАУЗ ДЛЯ РОБОТА =================
# Пауза (мс), которая автоматически вставляется при смене направления смычка
PAUSE_ON_BOW_CHANGE = 50 

# Пауза (мс) между двумя отдельными нотами (разделение 2 нот)
PAUSE_BETWEEN_NOTES = 20
# =============================================================

@dataclass
class BowStroke:
    index: int
    raw_line_idx: int
    pause_before: Optional[int]
    string: int
    position: int
    direction: int
    duration_ms: int
    speed: int
    is_sub: bool = False
    parent_note_idx: Optional[int] = None
    phrase_id: Optional[int] = None

@dataclass
class StrategyResult:
    name: str
    description: str
    strokes: List[BowStroke]
    score: int
    is_green: bool
    scenario_text: str
    min_speed: int
    max_speed: int
    avg_speed: float
    min_x: float
    max_x: float
    dir_changes: int
    legato_count: int
    phrases_count: int
    bow_utilization_pct: float
    warnings: List[str]


class ScenarioParser:
    """Парсинг сценария и извлечение нот."""
    @staticmethod
    def parse(lines: List[str]) -> Tuple[List[str], List[BowStroke], List[str]]:
        header = []
        strokes = []
        footer = []
        
        state = "header" 
        
        current_pause = None
        current_string = 0
        current_pos = 0
        note_idx = 0
        
        for i, line in enumerate(lines):
            line_s = line.strip()
            
            if state == "header":
                if line_s.startswith("/sn;"):
                    state = "body"
                elif line_s.startswith("/end") or line_s.startswith("/play"):
                    state = "footer"
                    footer.append(line)
                    continue
                else:
                    header.append(line)
                    continue
                    
            if state == "body":
                if line_s.startswith("/end") or line_s.startswith("/play"):
                    state = "footer"
                    footer.append(line)
                    continue
                    
                if line_s.startswith("/p;"):
                    parts = line_s.split(";")
                    if len(parts) >= 2:
                        current_pause = int(parts[1])
                elif line_s.startswith("/sn;"):
                    parts = line_s.split(";")
                    if len(parts) >= 3:
                        current_string = int(parts[1])
                        current_pos = int(parts[2])
                elif line_s.startswith("/bow;"):
                    parts = line_s.split(";")
                    if len(parts) >= 4:
                        direction = int(parts[1])
                        duration = int(parts[2])
                        speed = int(parts[3])
                        
                        stroke = BowStroke(
                            index=note_idx,
                            raw_line_idx=i,
                            pause_before=current_pause,
                            string=current_string,
                            position=current_pos,
                            direction=direction,
                            duration_ms=duration,
                            speed=speed
                        )
                        strokes.append(stroke)
                        note_idx += 1
                        current_pause = None 
            elif state == "footer":
                footer.append(line)
                
        return header, strokes, footer


class PhraseAggregator:
    """Музыкальное фразирование: группировка нот по правилам."""
    @staticmethod
    def aggregate(strokes: List[BowStroke], mode: str) -> List[List[int]]:
        if mode == "none" or not strokes:
            return [[i] for i in range(len(strokes))]
            
        phrases = []
        i = 0
        n = len(strokes)
        
        while i < n:
            def can_join(*durations):
                if i + len(durations) > n:
                    return False
                for j, dur in enumerate(durations):
                    if strokes[i+j].duration_ms != dur:
                        return False
                    # Запрет слияния, если перед присоединяемой нотой пауза
                    if j > 0 and strokes[i+j].pause_before is not None:
                        return False
                return True
                
            if mode in ("standard", "max_legato") and can_join(250, 250, 250, 250):
                phrases.append([i, i+1, i+2, i+3])
                i += 4
                continue
            if can_join(250, 250, 250):
                phrases.append([i, i+1, i+2])
                i += 3
                continue
            if can_join(500, 250):
                phrases.append([i, i+1])
                i += 2
                continue
            if can_join(250, 500):
                phrases.append([i, i+1])
                i += 2
                continue
            if can_join(500, 500):
                phrases.append([i, i+1])
                i += 2
                continue
            if mode in ("standard", "max_legato") and can_join(500, 750):
                phrases.append([i, i+1])
                i += 2
                continue
                
            phrases.append([i])
            i += 1
            
        return phrases


@dataclass(order=True)
class SearchState:
    cost: float
    group_idx: int = field(compare=False)
    X: float = field(compare=False)
    last_dir: int = field(compare=False)
    path: List[Tuple[int, int]] = field(compare=False) # list of (direction, speed)
    last_spd: int = field(compare=False)

class BowOptimizerEngine:
    """Ядро алгоритма: поиск пути с помощью очереди с приоритетом (A* / Dijkstra)."""
    def __init__(self, X_MIN, X_MAX, target_stroke, max_spd, smoothness):
        self.X_MIN = X_MIN
        self.X_MAX = X_MAX
        self.target_stroke = target_stroke
        self.max_spd = max_spd
        self.smoothness = smoothness

    def optimize(self, phrases: List[List[int]], strokes: List[BowStroke]) -> Optional[List[Tuple[int, int]]]:
        pq = []
        heapq.heappush(pq, SearchState(cost=0.0, group_idx=0, X=10000.0, last_dir=0, path=[], last_spd=0))
        visited = {} 
        
        while pq:
            state = heapq.heappop(pq)
            
            if state.group_idx == len(phrases):
                return state.path
                
            x_bucket = int(round(state.X / 1000.0))
            cache_key = (state.group_idx, x_bucket, state.last_dir)
            
            if cache_key in visited and visited[cache_key] <= state.cost:
                continue
            visited[cache_key] = state.cost
            
            phrase_indices = phrases[state.group_idx]
            tot_dur = sum(strokes[idx].duration_ms for idx in phrase_indices)
            if tot_dur == 0:
                heapq.heappush(pq, SearchState(state.cost, state.group_idx + 1, state.X, state.last_dir, state.path + [(state.last_dir, 0)], state.last_spd))
                continue
            
            if state.group_idx == 0:
                dirs = [1, 0]
            else:
                dirs = [1 - state.last_dir, state.last_dir]
                
            lookahead_dur = 0
            if state.group_idx + 1 < len(phrases):
                lookahead_dur = sum(strokes[idx].duration_ms for idx in phrases[state.group_idx + 1])
                
            for d in dirs:
                room = (self.X_MAX - state.X) if d == 1 else (state.X - self.X_MIN)
                if room <= 0:
                    continue
                    
                max_possible_spd = min(self.max_spd, int(room / tot_dur))
                if max_possible_spd < 26:
                    continue
                    
                ideal_spd = min(self.max_spd, max(36, int(round(self.target_stroke / tot_dur))))
                base_spd = min(ideal_spd, max_possible_spd)
                
                test_speeds = [base_spd - delta for delta in [0, 4, 8, 12, 16, 20]]
                test_speeds = sorted([s for s in test_speeds if s >= 26], reverse=True)
                
                for spd in test_speeds:
                    disp = tot_dur * spd
                    new_X = state.X + disp if d == 1 else state.X - disp
                    
                    if new_X < self.X_MIN or new_X > self.X_MAX:
                        continue
                        
                    # Расчет штрафов и бонусов
                    dir_pen = 20.0 if (state.group_idx > 0 and d == state.last_dir) else 0.0
                    full_bow_bonus = -(disp / 1000.0) * 0.45
                    
                    smooth_pen = 0.0
                    if state.group_idx > 0:
                        smooth_pen = abs(spd - state.last_spd) * 0.4 * self.smoothness
                        
                    look_pen = 0.0
                    if lookahead_dur >= 1000:
                        dist_edge = min(new_X - self.X_MIN, self.X_MAX - new_X)
                        look_pen = (dist_edge / 1000.0) * 0.4
                        
                    step_cost = dir_pen + full_bow_bonus + smooth_pen + look_pen
                    new_cost = state.cost + step_cost
                    
                    new_path = list(state.path)
                    new_path.append((d, spd))
                    
                    heapq.heappush(pq, SearchState(new_cost, state.group_idx + 1, new_X, d, new_path, spd))
                    
        return None


def score_and_build_result(name: str, desc: str, original_strokes: List[BowStroke], 
                           phrases: List[List[int]], path: List[Tuple[int, int]], 
                           X_MIN: float, X_MAX: float) -> StrategyResult:
    """Генерация итогового результата и оценка баллов."""
    if not path:
        return StrategyResult(
            name=name, description=desc, strokes=[], score=0, is_green=False, scenario_text="",
            min_speed=0, max_speed=0, avg_speed=0.0, min_x=0.0, max_x=0.0,
            dir_changes=0, legato_count=0, phrases_count=len(phrases), bow_utilization_pct=0.0,
            warnings=["No path found"]
        )
        
    strokes_out = []
    
    current_X = 10000.0
    min_x = current_X
    max_x = current_X
    dir_changes = 0
    legato_count = 0
    total_disp = 0.0
    speeds = []
    points_X = [current_X]
    
    last_dir = -1
    
    for p_idx, phrase in enumerate(phrases):
        d, spd = path[p_idx]
        
        if p_idx > 0 and d != last_dir:
            dir_changes += 1
            
        if len(phrase) > 1:
            legato_count += (len(phrase) - 1)
            
        for i, note_idx in enumerate(phrase):
            orig = original_strokes[note_idx]
            new_s = BowStroke(
                index=orig.index,
                raw_line_idx=orig.raw_line_idx,
                pause_before=orig.pause_before,
                string=orig.string,
                position=orig.position,
                direction=d,
                duration_ms=orig.duration_ms,
                speed=spd,
                phrase_id=p_idx
            )
            strokes_out.append(new_s)
            speeds.append(spd)
            
        disp = sum(original_strokes[idx].duration_ms for idx in phrase) * spd
        total_disp += disp
        current_X = current_X + disp if d == 1 else current_X - disp
        points_X.append(current_X)
        
        min_x = min(min_x, current_X)
        max_x = max(max_x, current_X)
        
        last_dir = d
        
    # Подсчет баллов
    score = 92
    
    avg_disp = total_disp / len(phrases) if phrases else 0
    util_pct = (avg_disp / (X_MAX - X_MIN)) * 100.0 if X_MAX > X_MIN else 0
    
    if util_pct >= 85.0:
        score += 5
    elif util_pct >= 75.0:
        score += 2
        
    score += min(8, legato_count * 2)
    
    warnings = []
    for spd in speeds:
        if spd < 28:
            score -= 4
            warnings.append(f"Штраф: низкая скорость {spd}")
            
    for x in points_X:
        if x < 2000 or x > 98000:
            score -= 5
            warnings.append(f"Штраф: опасная близость к краю смычка X={x}")
            
    score = max(50, min(99, score))
    is_green = (score >= 88)
    
    return StrategyResult(
        name=name,
        description=desc,
        strokes=strokes_out,
        score=score,
        is_green=is_green,
        scenario_text="", 
        min_speed=min(speeds) if speeds else 0,
        max_speed=max(speeds) if speeds else 0,
        avg_speed=sum(speeds)/len(speeds) if speeds else 0.0,
        min_x=min_x,
        max_x=max_x,
        dir_changes=dir_changes,
        legato_count=legato_count,
        phrases_count=len(phrases),
        bow_utilization_pct=util_pct,
        warnings=warnings
    )

STRATEGIES = [
    {"name": "Full-Bow Master", "mode": "standard", "X_MIN": 3000, "X_MAX": 97000, "target": 90000, "max_spd": 120, "smoothness": 1.0},
]

class SmartBower:
    """Обертка для запуска всех стратегий и выбора лучшей."""
    @staticmethod
    def process_scenario(scenario_lines: List[str]) -> Tuple[StrategyResult, List[StrategyResult]]:
        header, strokes, footer = ScenarioParser.parse(scenario_lines)
        
        results = []
        for st in STRATEGIES:
            phrases = PhraseAggregator.aggregate(strokes, st["mode"])
            engine = BowOptimizerEngine(
                X_MIN=st["X_MIN"],
                X_MAX=st["X_MAX"],
                target_stroke=st["target"],
                max_spd=st["max_spd"],
                smoothness=st["smoothness"]
            )
            
            path = engine.optimize(phrases, strokes)
            res = score_and_build_result(
                name=st["name"],
                desc=f"mode={st['mode']}",
                original_strokes=strokes,
                phrases=phrases,
                path=path,
                X_MIN=st["X_MIN"],
                X_MAX=st["X_MAX"]
            )
            
            if res.strokes:
                # Реконструкция текстового сценария
                rebuilt_lines = list(header)
                last_dir = -1
                for i, s in enumerate(res.strokes):
                    if s.pause_before is not None:
                        rebuilt_lines.append(f"/p;{s.pause_before}")
                    elif i > 0:
                        if s.direction != last_dir:
                            if PAUSE_ON_BOW_CHANGE > 0:
                                rebuilt_lines.append(f"/p;{PAUSE_ON_BOW_CHANGE}")
                        else:
                            if PAUSE_BETWEEN_NOTES > 0:
                                rebuilt_lines.append(f"/p;{PAUSE_BETWEEN_NOTES}")
                                
                    rebuilt_lines.append(f"/sn;{s.string};{s.position}")
                    rebuilt_lines.append(f"/bow;{s.direction};{s.duration_ms};{s.speed}")
                    last_dir = s.direction
                rebuilt_lines.extend(footer)
                res.scenario_text = "\n".join(rebuilt_lines)
                
            results.append(res)
            
        valid_results = [r for r in results if r.score > 0]
        if not valid_results:
            # Возвращаем хоть что-то
            return results[0], results
            
        # Выбираем лучшую стратегию по максимальному score
        best = max(valid_results, key=lambda x: x.score)
        return best, results

