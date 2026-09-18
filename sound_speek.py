from RealtimeSTT import AudioToTextRecorder
from sentence_transformers import SentenceTransformer, util
import torch


import os
# Вставь сюда свой токен от Hugging Face
os.environ["HF_TOKEN"] = "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"

from RealtimeSTT import AudioToTextRecorder
from sentence_transformers import SentenceTransformer, util
import torch

# ... дальше весь твой остальной код ...

# 1. Описание переходов состояний и ключевых фраз
states_commands = {
    "KEY": [
        "emma hello", 
        "hello emma", 
        "hi emma"
    ],
    "START": [
        "hello emma are you ready", 
        "emma are you ready", 
        "are you ready emma"
    ],
    "PLAY": [
        "emma play something", 
        "emma play a game", 
        "play something emma", 
        "play music"
    ]
}

# Плоский список для векторизации
all_phrases = []
phrase_to_state = {}
for state, phrases in states_commands.items():
    for phrase in phrases:
        all_phrases.append(phrase)
        phrase_to_state[phrase] = state

# Текущее состояние бота (начинаем с начального)
current_state = "IDLE"

def print_status(message=""):
    """Вспомогательная функция для логирования текущего состояния"""
    print("=" * 50)
    print(f"STATUS: Current State -> [{current_state}]")
    if message:
        print(f"INFO: {message}")
    print("=" * 50)

# 2. Функция обработки входящего звука/текста
# 2. Функция обработки входящего звука/текста
def process_text(text):
    global current_state

    text = text.lower().strip()
    if not text:
        return

    # --- ЖЕСТКИЙ ФИЛЬТР WAKE WORD ---
    # Если к боту не обратились по имени, игнорируем весь остальной текст
    if "emma" not in text:
        # Можно раскомментировать строку ниже, если хочешь видеть, что система отсеивает:
        # print(f"[IGNORED]: {text}") 
        return

    # Если имя есть, выводим услышанное
    print(f"\n[HEARD]: \"{text}\" | (Current State: {current_state})")

    # Превращаем услышанное в вектор и сравниваем
    user_embedding = nlp_model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, command_embeddings)[0]
    best_match_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_match_idx].item()

    # --- ПОВЫШЕННЫЙ ПОРОГ РАСПОЗНАВАНИЯ ---
    # 0.75 защитит от ложных срабатываний
    if best_score > 0.75:
        matched_phrase = all_phrases[best_match_idx]
        target_state = phrase_to_state[matched_phrase]

        # Если новое состояние отличается от текущего — переключаем
        if current_state != target_state:
            current_state = target_state
            print_status(f"Transitioned to new state via phrase: '{matched_phrase}' (Confidence: {best_score:.2f})")
        else:
            print(f"[REMAIN]: Already in state [{current_state}]")
    else:
        print(f"[UNKNOWN]: Command not recognized (Confidence: {best_score:.2f})")

# 3. Запуск прослушивания и загрузка моделей
if __name__ == '__main__':
    print("Loading NLP model on GPU (CUDA)...")
    # Явно указываем использование видеокарты (cuda)
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    print("Pre-computing command embeddings...")
    command_embeddings = nlp_model.encode(all_phrases, convert_to_tensor=True)

    print("\nSetting up microphone (RealtimeSTT) for GPU...")
    # Инициализируем recorder с настройками для GPU
    recorder = AudioToTextRecorder(
        model="large-v3", 
        language="en",
        device="cuda",             # Использование видеокарты
        compute_type="float16",     # Формат вычислений для GPU (максимальная скорость)
        initial_prompt="Emma",

        post_speech_silence_duration=1,
        min_length_of_recording=0.3
        
    )

    print_status("System initialized. Say one of the transition commands.")

    while True:
        recorder.text(process_text)