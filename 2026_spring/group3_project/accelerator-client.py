import socket

#HOST = "166.117.102.167"
HOST = "18.221.150.233"
PORT = 9000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(5)
    s.connect((HOST, PORT))

    msg = b"hello through global accelerator\n"
    s.sendall(msg)
    print(f"Sent: {msg!r}")

    data = s.recv(4096)
    print(f"Received: {data!r}")
    print(data.decode(errors="replace"))
