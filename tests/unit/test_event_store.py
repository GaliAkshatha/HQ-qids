from src.contracts import IncidentEvent
from src.contracts.incident import ASSESSING, DETECTED, DETECTION_CREATED
from src.incident.event_store import InMemoryEventStore, JsonlEventStore


def make_event(event_id, correlation_id="s1", incident_id="inc-1", previous=DETECTED, new=DETECTED):
    return IncidentEvent(
        event_id=event_id, correlation_id=correlation_id, incident_id=incident_id, event_type=DETECTION_CREATED,
        previous_state=previous, new_state=new, timestamp="2026-01-01T00:00:00+00:00", reason="ok",
    )


def test_in_memory_append_and_read_all():
    store = InMemoryEventStore()
    store.append(make_event("e1"))
    store.append(make_event("e2"))
    events = store.read_all()
    assert [e.event_id for e in events] == ["e1", "e2"]


def test_in_memory_duplicate_event_id_not_appended_twice():
    store = InMemoryEventStore()
    first = store.append(make_event("e1"))
    second = store.append(make_event("e1"))  # same event_id again
    assert first is True
    assert second is False
    assert len(store.read_all()) == 1


def test_in_memory_read_all_filters_by_correlation_id():
    store = InMemoryEventStore()
    store.append(make_event("e1", correlation_id="s1"))
    store.append(make_event("e2", correlation_id="s2"))
    assert [e.event_id for e in store.read_all("s1")] == ["e1"]


def test_in_memory_event_exists():
    store = InMemoryEventStore()
    store.append(make_event("e1"))
    assert store.event_exists("e1") is True
    assert store.event_exists("e2") is False


def test_jsonl_append_and_read_all(tmp_path):
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path)
    store.append(make_event("e1"))
    store.append(make_event("e2"))
    events = store.read_all()
    assert [e.event_id for e in events] == ["e1", "e2"]
    assert path.exists()


def test_jsonl_duplicate_event_id_not_written_twice(tmp_path):
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path)
    store.append(make_event("e1"))
    result = store.append(make_event("e1"))
    assert result is False
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 1


def test_jsonl_reload_from_disk_reconstructs_events(tmp_path):
    """The literal 'restart' proof at the event-store level: a brand new
    JsonlEventStore instance pointed at the same file sees everything the
    first instance wrote, without that first instance still existing."""
    path = tmp_path / "events.jsonl"
    first_store = JsonlEventStore(path)
    first_store.append(make_event("e1", previous=DETECTED, new=DETECTED))
    first_store.append(make_event("e2", previous=DETECTED, new=ASSESSING))
    del first_store

    second_store = JsonlEventStore(path)
    events = second_store.read_all()
    assert [e.event_id for e in events] == ["e1", "e2"]
    assert second_store.event_exists("e1") is True
    assert second_store.event_exists("e2") is True


def test_jsonl_reload_preserves_dedup_across_restart(tmp_path):
    path = tmp_path / "events.jsonl"
    first_store = JsonlEventStore(path)
    first_store.append(make_event("e1"))
    del first_store

    second_store = JsonlEventStore(path)
    result = second_store.append(make_event("e1"))  # same event_id, "arrives again" post-restart
    assert result is False
    assert len(second_store.read_all()) == 1


def test_jsonl_event_ordering_is_preserved_across_many_appends(tmp_path):
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path)
    for i in range(20):
        store.append(make_event(f"e{i}"))
    reloaded = JsonlEventStore(path)
    assert [e.event_id for e in reloaded.read_all()] == [f"e{i}" for i in range(20)]
