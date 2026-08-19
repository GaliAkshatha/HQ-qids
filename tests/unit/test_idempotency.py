from src.runtime.idempotency import InMemoryIdempotencyStore


def test_in_memory_store_marks_and_checks():
    store = InMemoryIdempotencyStore()
    assert store.already_processed("e1") is False
    store.mark_processed("e1")
    assert store.already_processed("e1") is True
    assert store.already_processed("e2") is False


def test_in_memory_store_distinguishes_event_ids():
    store = InMemoryIdempotencyStore()
    store.mark_processed("e1")
    store.mark_processed("e2")
    assert store.already_processed("e1") is True
    assert store.already_processed("e2") is True
    assert store.already_processed("e3") is False
