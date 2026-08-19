import json

from src.runtime.tracing import format_trace, trace_correlation


def write_log(path, records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_trace_filters_by_correlation_id_across_multiple_files(tmp_path):
    write_log(tmp_path / "detection_worker.log", [
        {"timestamp": "2026-01-01T00:00:01+00:00", "service": "detection_worker", "correlation_id": "c1", "event_type": "detection.completed", "message": "processed"},
        {"timestamp": "2026-01-01T00:00:01+00:00", "service": "detection_worker", "correlation_id": "c2", "event_type": "detection.completed", "message": "unrelated"},
    ])
    write_log(tmp_path / "quantum_worker.log", [
        {"timestamp": "2026-01-01T00:00:02+00:00", "service": "quantum_worker", "correlation_id": "c1", "event_type": "quantum.completed", "message": "processed"},
    ])

    entries = trace_correlation("c1", log_dir=tmp_path)
    assert len(entries) == 2
    assert entries[0].service == "detection_worker"
    assert entries[1].service == "quantum_worker"


def test_trace_is_ordered_by_timestamp_regardless_of_file_order(tmp_path):
    write_log(tmp_path / "b_service.log", [
        {"timestamp": "2026-01-01T00:00:05+00:00", "service": "b_service", "correlation_id": "c1", "message": "later"},
    ])
    write_log(tmp_path / "a_service.log", [
        {"timestamp": "2026-01-01T00:00:01+00:00", "service": "a_service", "correlation_id": "c1", "message": "earlier"},
    ])

    entries = trace_correlation("c1", log_dir=tmp_path)
    assert [e.message for e in entries] == ["earlier", "later"]


def test_trace_ignores_malformed_lines(tmp_path):
    path = tmp_path / "broken.log"
    with open(path, "w") as f:
        f.write("not valid json\n")
        f.write(json.dumps({"timestamp": "t", "service": "s", "correlation_id": "c1", "message": "ok"}) + "\n")

    entries = trace_correlation("c1", log_dir=tmp_path)
    assert len(entries) == 1


def test_trace_returns_empty_for_unknown_correlation_id(tmp_path):
    write_log(tmp_path / "svc.log", [{"timestamp": "t", "service": "s", "correlation_id": "c1", "message": "m"}])
    entries = trace_correlation("does-not-exist", log_dir=tmp_path)
    assert entries == []


def test_trace_returns_empty_for_missing_log_dir(tmp_path):
    entries = trace_correlation("c1", log_dir=tmp_path / "does_not_exist")
    assert entries == []


def test_format_trace_produces_readable_output(tmp_path):
    write_log(tmp_path / "svc.log", [
        {"timestamp": "2026-01-01T00:00:00+00:00", "service": "svc", "correlation_id": "c1", "event_type": "x.y", "message": "hello"},
    ])
    entries = trace_correlation("c1", log_dir=tmp_path)
    output = format_trace(entries)
    assert "svc" in output
    assert "hello" in output


def test_format_trace_handles_empty_list():
    assert "no trace entries" in format_trace([]).lower()
