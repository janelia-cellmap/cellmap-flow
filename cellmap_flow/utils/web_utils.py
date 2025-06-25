import socket
import json
import base64


ARGS_KEY = "__CFLOW_ARGS__"
IP_PATTERN = ["CELLMAP_FLOW_SERVER_IP(", ")CELLMAP_FLOW_SERVER_IP"]
INPUT_NORM_DICT_KEY = "input_norm"
POSTPROCESS_DICT_KEY = "postprocess"


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 0))
    free_port = s.getsockname()[1]
    s.close()
    return free_port


def get_public_ip():
    """
    Return the local/private IP address in use on this machine
    (e.g., 10.x.x.x or 192.168.x.x if behind NAT).
    This *does not* return the real Internet-facing public IP
    if you're behind NAT.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 8.8.8.8 doesn't need to be reachable;
        # the connect() call will assign a local IP regardless.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        # Fallback if something fails
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def encode_to_str(data):
    """Encodes a JSON object into a URL-safe string without '/', '+', or '='."""
    json_str = json.dumps(data, separators=(",", ":"))  # Minify JSON
    encoded_bytes = base64.urlsafe_b64encode(json_str.encode())  # Base64 encode
    return encoded_bytes.decode().rstrip("=")  # Remove padding ('=')


def decode_to_json(encoded_str):
    """Decodes a URL-safe string back into a JSON object."""
    padding_needed = 4 - (len(encoded_str) % 4)
    encoded_str += "=" * (padding_needed % 4)  # Add padding back if needed
    json_str = base64.urlsafe_b64decode(encoded_str.encode()).decode()  # Decode Base64
    return json.loads(json_str)  # Convert back to JSON


def list_cls_to_dict(ll):
    args = {}
    norms = {}
    for n in ll:
        name = n.name()
        elms = n.to_dict()
        elms.pop("name")
        elms = {k: str(v) for k, v in elms.items()}
        norms[name] = elms

    return norms


def kill_n_remove_from_neuroglancer(jobs, s):
    for job in jobs:
        if job.model_name in s.layers:
            del s.layers[job.model_name]
        job.kill()


def get_norms_post_args(input_norms, postprocess):
    args = {}

    args[INPUT_NORM_DICT_KEY] = list_cls_to_dict(input_norms)
    args[POSTPROCESS_DICT_KEY] = list_cls_to_dict(postprocess)
    st_data = encode_to_str(args)
    return st_data
