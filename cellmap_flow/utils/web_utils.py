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

import json
import base64

def encode_to_str(data):
    """Encodes a JSON object into a URL-safe string without '/', '+', or '='."""
    json_str = json.dumps(data, separators=(',', ':'))  # Minify JSON
    encoded_bytes = base64.urlsafe_b64encode(json_str.encode())  # Base64 encode
    return encoded_bytes.decode().rstrip('=')  # Remove padding ('=')

def decode_to_json(encoded_str):
    """Decodes a URL-safe string back into a JSON object."""
    padding_needed = 4 - (len(encoded_str) % 4)
    encoded_str += "=" * (padding_needed % 4)  # Add padding back if needed
    json_str = base64.urlsafe_b64decode(encoded_str.encode()).decode()  # Decode Base64
    return json.loads(json_str)  # Convert back to JSON


INPUT_NORM_KEY= "__input_norm__"
