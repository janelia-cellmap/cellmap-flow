"""The Review tab's routes: a session over a review index, the path rule on
the index it opens, and the t-key pick reaching every open stream."""

import itertools
import json
import os
import queue
import sqlite3
import stat
import threading

import neuroglancer
import pytest
from neuroglancer.viewer_base import ViewerBase

from cellmap_flow.dashboard.app import app
from cellmap_flow.dashboard.state import get_session

OPEN = "/api/review/open"

# id, vox, centroid (nm), rank in the "smallest" queue
INSTANCES = [(1, 30, (8.0, 16.0, 24.0), 2), (2, 10, (80.0, 160.0, 240.0), 0), (3, 20, (4.0, 4.0, 4.0), 1)]


def _make_index(path):
    """A review index as older builders wrote it: no ledger.entry_method."""
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE instances (id INTEGER PRIMARY KEY, cz REAL, cy REAL, cx REAL,"
        " cz_nm REAL, cy_nm REAL, cx_nm REAL, bz0 INT, bz1 INT, by0 INT, by1 INT,"
        " bx0 INT, bx1 INT, vox INT, sphericity REAL, fm_score REAL, rank_smallest INT);"
        "CREATE TABLE ledger (instance_id INTEGER PRIMARY KEY, review_state TEXT,"
        " reviewed_at TEXT, reviewer TEXT, edit_details_json TEXT);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
    )
    for i, vox, (z, y, x), rank in INSTANCES:
        conn.execute(
            "INSERT INTO instances VALUES (?,?,?,?,?,?,?,0,1,0,1,0,1,?,0.5,NULL,?)",
            (i, z / 8, y / 8, x / 8, z, y, x, vox, rank),
        )
        conn.execute("INSERT INTO ledger (instance_id) VALUES (?)", (i,))
    conn.commit()
    conn.close()
    return path


def _ledger_columns(path):
    with sqlite3.connect(path) as conn:
        return [r[1] for r in conn.execute("PRAGMA table_info(ledger)")]


@pytest.fixture
def client():
    return app.test_client()


@pytest.fixture
def index(tmp_path):
    return str(_make_index(tmp_path / "review.sqlite"))


@pytest.fixture
def viewer():
    v = ViewerBase()  # neuroglancer's state, actions and config, no server
    with v.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=[8, 8, 8])
    get_session().viewer = v
    return v


# (method, url, body, status, what the JSON must contain)
SESSION = [
    ("get", "/api/review/next?order=smallest&min_vox=5", None, 200, {"id": 2, "navigated": True}),
    ("get", "/api/review/next?order=smallest&skip_rank=0", None, 200, {"id": 3}),
    # The queue list is still empty when the page first sends this: the first queue.
    ("get", "/api/review/next?order=&min_vox=5", None, 200, {"id": 2}),
    ("post", "/api/review/verdict", {"id": 2, "verdict": "blessed", "entry_method": "next"}, 200, {"success": True}),
    ("get", "/api/review/next?order=smallest", None, 200, {"id": 3}),
    ("get", "/api/review/progress", None, 200, {"by_state": {"blessed": 1, "unreviewed": 2}}),
    ("post", "/api/review/undo", {"id": 2}, 200, {"success": True}),
    ("get", "/api/review/progress", None, 200, {"by_state": {"unreviewed": 3}}),
    ("get", "/api/review/next?order=smallest&min_vox=abc", None, 400, {}),
    ("get", "/api/review/next?order=smallest&skip_rank=1.5", None, 400, {}),
    ("get", "/api/review/next?order=nope", None, 400, {}),
    ("post", "/api/review/verdict", {"id": "x", "verdict": "blessed"}, 400, {}),
    ("post", "/api/review/verdict", {"id": 2, "verdict": "maybe"}, 400, {}),
    ("post", "/api/review/undo", {}, 400, {}),
    ("get", "/api/review/debug/viewer_state", None, 404, None),
]


def test_a_review_session(client, index, viewer):
    reviewer = "<img src=x onerror=alert(1)>"
    opened = client.post(OPEN, json={"db_path": index, "reviewer": reviewer})
    assert opened.status_code == 200 and opened.get_json()["n_instances"] == 3
    # Opening, and every read, leaves the index as it was.
    assert "entry_method" not in _ledger_columns(index)

    for method, url, body, status, expected in SESSION:
        response = getattr(client, method)(url, json=body)
        assert response.status_code == status, (url, body, response.get_data(as_text=True))
        if expected is not None:
            got = response.get_json()
            assert {k: got[k] for k in expected} == expected, (url, body)
        if url.startswith("/api/review/next?order=smallest&min_vox=5"):
            # Moved to instance 2's centroid, in the viewer's 8 nm voxels.
            assert list(viewer.state.position) == [10.0, 20.0, 30.0]

    # Verdicts go in under the session's reviewer, through a writable
    # connection that added the column older indexes lack.
    assert "entry_method" in _ledger_columns(index)
    ledger = client.post("/api/review/verdict", json={"id": 1, "verdict": "erased"}).get_json()["ledger"]
    assert (ledger["review_state"], ledger["reviewer"]) == ("erased", reviewer)


def test_an_index_that_cannot_be_written_or_has_gone_is_an_error_the_tab_can_read(client, index, tmp_path):
    """Not Flask's HTML 500, which the tab's JSON parse turned into a SyntaxError."""
    client.post(OPEN, json={"db_path": index})
    os.chmod(index, stat.S_IRUSR)
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    try:
        refused = client.post("/api/review/verdict", json={"id": 2, "verdict": "blessed"})
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)
        os.chmod(index, stat.S_IRUSR | stat.S_IWUSR)
    assert refused.status_code == 500 and "readonly" in refused.get_json()["error"]
    os.remove(index)
    for url in ("/api/review/progress", "/api/review/next?order=smallest", "/api/review/show/2"):
        gone = client.get(url)
        assert (gone.status_code, gone.get_json()) == (404, {"error": f"review index not found: {index}"}), url


def test_only_a_review_index_by_name_is_opened(client, tmp_path):
    secret = tmp_path / "passwd"
    secret.write_text("secret")
    (tmp_path / "link.sqlite").symlink_to(secret)
    (tmp_path / "garbage.db").write_text("secret")
    with sqlite3.connect(tmp_path / "other.sqlite") as conn:
        conn.execute("CREATE TABLE ledger (instance_id INTEGER)")

    refused = ["passwd", "missing.txt", "review.sqlite.bak", "link.sqlite", "garbage.db", "other.sqlite"]
    for name in refused:
        response = client.post(OPEN, json={"db_path": str(tmp_path / name)})
        assert response.status_code == 400, name
        assert "secret" not in response.get_data(as_text=True)
    assert client.post(OPEN, json={"db_path": str(tmp_path / "missing.sqlite")}).status_code == 404
    # Not a review index, so not migrated either.
    assert _ledger_columns(tmp_path / "other.sqlite") == ["instance_id"]
    assert get_session().review is None


def test_every_pick_stream_sees_every_pick(client, index, viewer):
    client.post(OPEN, json={"db_path": index, "segmentation_layer": "labels"})
    received = queue.Queue()

    def watch():  # one browser tab: its own connection, on its own thread
        response = app.test_client().get("/api/review/pick_stream", buffered=False)
        for chunk in itertools.islice(response.response, 2):
            received.put(json.loads(chunk.decode().removeprefix("data: ")))
        response.close()

    tabs = [threading.Thread(target=watch, daemon=True) for _ in range(2)]
    for tab in tabs:
        tab.start()
    assert [received.get(timeout=10)["pick"] for _ in tabs] == [None, None]
    viewer.actions.invoke("review-pick", {"selectedValues": {"labels": {"value": 3}}})
    # One stream reading the pick must not hide it from the other.
    picks = [received.get(timeout=10) for _ in tabs]
    assert [(e["seq"], e["pick"]["id"]) for e in picks] == [(1, 3), (1, 3)]
    for tab in tabs:
        tab.join(timeout=10)
