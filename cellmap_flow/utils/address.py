"""This machine's address, as the links handed to browsers give it."""

import functools
import logging
import socket

logger = logging.getLogger(__name__)

# How long a connection from this machine to its own address may take before
# the address is taken as blocked. A VPN that blocks it drops the packets, so
# the connection hangs rather than failing.
SELF_CONNECT_TIMEOUT = 1.0


def this_machine_ip():
    """This machine's address on its network (10.x.x.x, say): the interface a
    connection out would use. 127.0.0.1 if there is none, or if this machine
    cannot reach itself there.

    The links a browser is given -- the dashboard's, its viewers', the
    inference servers' and MinIO's -- name the machine by this address, not
    by its hostname. A hostname can point somewhere else: a laptop off campus
    keeps the name DNS registered for its campus address, and a link by name
    then led to an address it no longer held.

    Nor does the address always work from the machine itself. Off campus over
    the VPN, a Mac could not connect to its own tunnel address, curl no more
    than the browser, while localhost answered. Where there is no cluster the
    browser is on this machine, so 127.0.0.1 is the address that works.
    """
    ip = _outgoing_address()
    if ip != "127.0.0.1" and not _reachable_from_here(ip):
        return "127.0.0.1"
    return ip


def _outgoing_address():
    """The address of the interface a connection out would use. Nothing is
    sent; 127.0.0.1 if there is none."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


@functools.lru_cache(maxsize=None)
def _reachable_from_here(ip):
    """Whether this machine can connect to itself at ``ip``, as a server here
    bound to every interface would be connected to.

    Asked once per address per process: a blocked address costs
    SELF_CONNECT_TIMEOUT, and a different network gives a different address.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("0.0.0.0", 0))
        listener.listen(1)
        try:
            with socket.create_connection((ip, listener.getsockname()[1]), timeout=SELF_CONNECT_TIMEOUT):
                return True
        except OSError:
            logger.info(f"This machine cannot connect to itself at {ip} (a VPN can block that); "
                        "links to it use 127.0.0.1, for a browser on this machine")
            return False
