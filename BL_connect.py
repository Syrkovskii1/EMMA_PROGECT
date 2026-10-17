import socket
import time

SERVER_IP = "192.168.67.1"
PORT = 5000
BUFFER_SIZE = 65536  # 64 KB
TEST_DURATION = 10   # Длительность теста в секундах

def run_speed_test():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        print(f"[*] Подключение к {SERVER_IP}:{PORT}...")
        client.connect((SERVER_IP, PORT))
        print(f"[+] Соединение установлено. Начинаем тест скорости на {TEST_DURATION} секунд...")
        
        # Создаем фиктивный буфер данных для отправки
        payload = b'\x00' * BUFFER_SIZE
        
        start_time = time.time()
        total_bytes = 0
        
        # Отправляем данные, пока не истечет время теста
        while time.time() - start_time < TEST_DURATION:
            client.sendall(payload)
            total_bytes += BUFFER_SIZE
            
        elapsed = time.time() - start_time
        speed_mbps = (total_bytes * 8) / (elapsed * 1_000_000)
        speed_mb_s = total_bytes / (elapsed * 1024 * 1024)
        
        print("\n--- Результаты клиента ---")
        print(f"Время тестирования : {elapsed:.2f} с")
        print(f"Всего отправлено   : {total_bytes / (1024*1024):.2f} МБ")
        print(f"Средняя скорость   : {speed_mbps:.2f} Мбит/с ({speed_mb_s:.2f} МБ/с)")
        print("--------------------------\n")

    except ConnectionRefusedError:
        print(f"[!] Сервер {SERVER_IP}:{PORT} недоступен.")
    except Exception as e:
        print(f"[!] Ошибка: {e}")
    finally:
        client.close()
        print("[*] Сокет закрыт.")

if __name__ == "__main__":
    run_speed_test()