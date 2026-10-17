import serial
from serial.tools import list_ports
import sys
import time
import os

# Глобальная переменная для хранения объекта последовательного порта
arduino_serial = None 
BAUD_RATE = 115200

#scenario_moanin_5.txt  scenario_riorita.txt
COMMAND_FILE = r"C:\Users\syrko\Documents\RC_finall_bluetooth\shedulers_melody\scenario_riorita.txt"

#COMMAND_FILE = r"C:\Users\syrko\Documents\RC_finall_bluetooth\shedulers_melody\orkestr_pirate_sveg.txt"

#COMMAND_FILE = r"C:\Users\98195\Documents\RC_finall_bluetooth\orch_waltz.txt"

#COMMAND_FILE = r"C:\Users\98195\Downloads\Telegram Desktop\test.txt"

#COMMAND_FILE = r"C:\Users\98195\Documents\RC_finall_bluetooth\palez.txt"

# --- Функция поиска порта ---
def find_arduino_port():
    """
    Автоматически находит COM-порт Arduino.
    """
    # Добавлен более широкий набор поисковых терминов
    search_terms = ['arduino', 'usb serial device', 'ch340', 'cp210x', 'usb-s  erial', 'ftdi', 'ttl']
    
    for port in list_ports.comports():
        desc_lower = port.description.lower()
        
        if any(term in desc_lower for term in search_terms):
            print(f"✅ Arduino найдена на порту: {port.device} ({port.description})")
            return port.device
        
    raise RuntimeError("❌ Порт Arduino не найден. Проверьте подключение и драйверы.")


# --- Функция инициализации и открытия порта (вызывается 1 раз) ---
def initialize_serial_connection(port, baud_rate):
    """
    Устанавливает последовательное соединение с Arduino.
    Отключение DTR/RTS помогает предотвратить автосброс (auto-reset).
    """
    try:
        ser = serial.Serial(
            port, 
            baudrate=baud_rate, 
            timeout=1, # Таймаут для чтения
            rtscts=False, 
            dsrdtr=False,
            write_timeout=0.1 # Таймаут для записи, чтобы не блокировать программу
        )
        
        # Ждем немного для стабилизации соединения
        time.sleep(2) 
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        print("✅ Соединение с Arduino установлено. Автосброс отключен.")
        return ser
        
    except serial.SerialException as e:
        print(f"❌ Ошибка COM-порта при открытии: {e}")
        sys.exit(1)


# --- Функция отправки команды (использует уже открытый порт) ---
def send_command(COMMAND_TO_SEND):
    """
    Отправляет команду через ранее установленное последовательное соединение, 
    добавляя символ завершения строки ('\n').
    """
    global arduino_serial
    
    if arduino_serial is None or not arduino_serial.is_open:
        print("❌ Ошибка: Последовательный порт не открыт или был закрыт.")
        # Не выходим, а только печатаем, чтобы не прерывать весь скрипт
        return 

    try:
        # Убираем пробелы и добавляем символ новой строки
        message_to_send = COMMAND_TO_SEND.strip() + "\n" 
        
        # КЛЮЧЕВАЯ ОПЕРАЦИЯ: Отправка байтов в COM-порт
        arduino_serial.write(message_to_send.encode('utf-8'))
        print(f"-> Отправлена команда: {COMMAND_TO_SEND.strip()}")
        
    except serial.SerialException as e:
        print(f"❌ Ошибка COM-порта при записи: {e}")
        # Попытка закрыть порт в случае критической ошибки
        try:
            if arduino_serial.is_open:
                arduino_serial.close()
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"❌ Произошла непредвиденная ошибка при отправке: {e}")
        sys.exit(1)


# --- Функция чтения и парсинга команд из файла ---
def read_commands_from_file(filepath):
    """
    Читает команды из TXT файла, парсит временные метки (в мс) и команды.
    Возвращает отсортированный список кортежей: (timestamp_ms, command_string).
    """
    commands = []
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Файл команд не найден: {filepath}")

    print(f"Загрузка команд из файла: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Игнорируем пустые строки и строки-комментарии
            if not line or line.startswith('#'):
                continue
            
            try:
                # Ищем первое вхождение разделителя ';'
                parts = line.split(';', 1)
                if len(parts) != 2:
                    raise ValueError("Неправильный формат строки (ожидается 'TIME;COMMAND')")
                
                timestamp_ms = int(parts[0].strip())
                command_body = parts[1].strip()
                
                if not command_body:
                    print(f"⚠️ Предупреждение (строка {line_num}): Пустая команда после временной метки. Пропускаем.")
                    continue
                
                commands.append((timestamp_ms, command_body))
                
            except ValueError as e:
                print(f"❌ Ошибка парсинга в строке {line_num} ('{line}'): {e}. Пропускаем.")
            
    # Сортируем команды по временной метке, чтобы обеспечить правильный порядок
    commands.sort(key=lambda x: x[0])
    
    print(f"✅ Загружено {len(commands)} команд.")
    return commands


# --- Функция выполнения команд по таймингу ---
def execute_script(commands):
    """
    Выполняет команды, ожидая заданное время (в мс) с момента начала.
    """
    if not commands:
        print("⚠️ Нет команд для выполнения.")
        return

    # Запоминаем время начала выполнения скрипта
    start_time = time.time() 
    
    print("\n--- Запуск последовательности команд по таймеру ---")

    for target_ms, command in commands:
        # Вычисляем целевое время (абсолютное) в секундах
        target_time_s = start_time + (target_ms / 1000.0)
        
        # Вычисляем, сколько времени осталось ждать
        wait_time = target_time_s - time.time()
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Отправляем команду
        send_command(command)
        
    # Дополнительная пауза после последней команды для ее гарантированного выполнения
    print("--- Последовательность команд завершена ---")
    time.sleep(1) 


# --- Основная логика программы ---
if __name__ == "__main__":
    
    try:
        # 1. Поиск порта
        arduino_port = find_arduino_port() 
        
        # 2. Инициализация и открытие порта (ТОЛЬКО ОДИН РАЗ)
        arduino_serial = initialize_serial_connection(arduino_port, BAUD_RATE)

        # 3. Чтение команд из файла
        script_commands = read_commands_from_file(COMMAND_FILE)
        
        # 4. Выполнение команд по таймингу
        execute_script(script_commands)

        # 5. Завершающие команды (опционально)
        print("\nПрограмма завершила выполнение команд из файла.")


    except RuntimeError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
    finally:
        # 6. Закрытие порта в конце программы
        if arduino_serial is not None and arduino_serial.is_open:
            arduino_serial.close()
            print("\n✅ Соединение с Arduino закрыто.")

