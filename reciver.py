import socket

def main():
    TCP_IP = "0.0.0.0"
    TCP_PORT = 9000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(5)

    print(f"[*] Ожидание подключений на порту {TCP_PORT}...")
    
    while True:
        try:
            conn, addr = sock.accept()
            data = conn.recv(4096)
            if data:
                message = data.decode('utf-8', errors='replace')
                print(f"\n--- Получено от {addr} ---")
                print(message)
                print("---------------------------")
            conn.close()
        except KeyboardInterrupt:
            print("\n[*] Завершение работы.")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
