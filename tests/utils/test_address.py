"""The address the links a browser is given name this machine by (utils.address)."""

import socket

import pytest

from cellmap_flow.utils import address


@pytest.mark.parametrize("reachable, expected", [
    pytest.param(True, "192.168.100.109", id="reachable"),
    # Over the VPN, a Mac could not connect to its own tunnel address; only
    # localhost answered, for curl as for the browser.
    pytest.param(False, "127.0.0.1", id="blocked-by-a-vpn"),
])
def test_links_name_this_machine_by_an_address_it_can_reach_itself_at(monkeypatch, reachable, expected):
    monkeypatch.setattr(address, "_outgoing_address", lambda: "192.168.100.109")
    monkeypatch.setattr(address, "_reachable_from_here", lambda ip: reachable)
    assert address.this_machine_ip() == expected


def test_a_connection_to_this_machine_gets_through_on_loopback():
    assert address._reachable_from_here("127.0.0.1")


def test_a_connection_that_times_out_is_blocked(monkeypatch):
    def dropped(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "create_connection", dropped)
    # Uncached, so the timeout is this call's own.
    assert not address._reachable_from_here.__wrapped__("192.168.100.109")
