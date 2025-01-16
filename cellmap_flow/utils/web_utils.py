def get_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def get_public_ip():
    import requests

    try:
        return requests.get("https://api.ipify.org").text
    except:
        return None
