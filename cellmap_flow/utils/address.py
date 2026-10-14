"""This machine's address, as the links handed to browsers give it."""

import socket


def this_machine_ip():
    """This machine's address on its network (10.x.x.x, say): the interface a
    connection out would use. Nothing is sent; 127.0.0.1 if there is none.

    The links a browser is given -- the dashboard's, its viewers', the
    inference servers' and MinIO's -- name the machine by this address, not
    by its hostname. A hostname can point somewhere else: a laptop off campus
    keeps the name DNS registered for its campus address, and a link by name
    then led to an address it no longer held.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
