import socket
import uvicorn

from app import app, get_port


def get_available_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", preferred_port))
            return preferred_port
        except OSError:
            sock.bind(("0.0.0.0", 0))
            return sock.getsockname()[1]


if __name__ == "__main__":
    port = get_port()
    uvicorn.run(app, host="0.0.0.0", port=port)
