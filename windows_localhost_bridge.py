"""Forward Windows 127.0.0.1:8766 to the WSL serve. No admin. This machine only."""
from __future__ import annotations

import socket
import sys
import threading

LISTEN = ("127.0.0.1", 8766)


def pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            src.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            dst.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        src.close()
        dst.close()


def handle(client: socket.socket, dest: tuple[str, int]) -> None:
    try:
        upstream = socket.create_connection(dest, timeout=5)
        upstream.settimeout(None)
    except OSError as exc:
        try:
            client.sendall(
                b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n"
                + str(exc).encode()
            )
        except OSError:
            pass
        client.close()
        return
    threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
    pump(upstream, client)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: windows_localhost_bridge.py <wsl-ipv4>")
        return 2
    dest = (sys.argv[1], 8766)
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(LISTEN)
    ls.listen(64)
    print(f"Windows bridge http://127.0.0.1:8766/  ->  {dest[0]}:8766")
    print("Close this window to stop.")
    try:
        while True:
            client, _ = ls.accept()
            threading.Thread(target=handle, args=(client, dest), daemon=True).start()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        ls.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
