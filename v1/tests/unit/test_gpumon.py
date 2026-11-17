"""gpumon (the out-of-process GPU monitor) without a GPU: event alignment, process grouping, the VLM configuration's
environment overrides and the run summary on a synthetic run folder."""

import csv
import json

import pytest

pytest.importorskip("numpy")
psutil = pytest.importorskip("psutil")


def test_events_are_anchored_by_the_least_delayed_arrival():
    from gpumon.report import align
    ev = [{"t": 0.0, "arrival": 1000.30, "file": "a"}, {"t": 2.0, "arrival": 1002.05, "file": "a"},
          {"t": 0.0, "arrival": 2000.10, "file": "b"}]
    out = align(ev)
    assert [round(e["time"], 2) for e in out] == [1000.05, 1002.05, 2000.10]  # offset per file = min(arrival - t)


def test_only_python_processes_are_grouped():
    from gpumon.procs import _match
    assert _match(["/r/.venv/bin/python", "-u", "-m", "sim.run_scenario", "--headless"]) == "sim"
    assert _match(["/r/.venv-seg/bin/python", "-u", "-m", "agent.run", "--settings"]) == "agent"
    assert _match(["/r/.venv-seg/bin/python3.12", "-u", "-m", "evaluation.lab_exp1_agent"]) == "agent"
    assert _match(["/bin/bash", "-c", "python -m agent.run"]) == "other"  # a shell mentioning it is not it
    assert _match(["grep", "sim.run_scenario"]) == "other"
    assert _match([]) == "other"


def test_vlm_config_reads_overrides_and_keeps_defaults(monkeypatch):
    from vlm.config import VLMConfig
    for k in ("VLM_QUANTIZATION", "VLM_EMBEDDINGS_ON_CPU", "VLM_THINKING_TOKENS", "VLM_IMAGE_MAX_SIDE"):
        monkeypatch.delenv(k, raising=False)
    d = VLMConfig()
    assert (d.quantization, d.per_layer_embeddings_on_cpu, d.thinking_tokens, d.image_max_side) == ("4bit", True, 3072, 960)
    monkeypatch.setenv("VLM_QUANTIZATION", "8bit")
    monkeypatch.setenv("VLM_EMBEDDINGS_ON_CPU", "0")
    monkeypatch.setenv("VLM_IMAGE_MAX_SIDE", "640")
    c = VLMConfig()
    assert (c.quantization, c.per_layer_embeddings_on_cpu, c.image_max_side) == ("8bit", False, 640)
    assert VLMConfig(quantization="16bit").quantization == "16bit"  # explicit arguments still win
    monkeypatch.setenv("VLM_QUANTIZATION", "3bit")
    with pytest.raises(ValueError):
        VLMConfig()


def test_summary_of_a_synthetic_run(tmp_path):
    pytest.importorskip("pynvml")  # gpumon.sampler (column list) needs it; installed in .venv-seg
    from gpumon.report import summarize
    from gpumon.sampler import COLUMNS
    t0 = 1_000_000.0
    used = lambda t: 600 + (3000 if t >= 2 else 0) + (5000 if 4 <= t < 5 else 0) + (300 if 6 <= t < 7 else 0)  # noqa: E731
    with open(tmp_path / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i in range(100):  # 10 s at 10 Hz
            t = i / 10
            w.writerow({**{c: 0 for c in COLUMNS}, "time": t0 + t, "used_mib": used(t), "free_mib": 12282 - used(t),
                        "total_mib": 12282, "gpu_agent_mib": max(0, used(t) - 600 - 3000) if t >= 4 else 0,
                        "gpu_sim_mib": 3000 if t >= 2 else 0})
    marks = [("launch", 1.0), ("gemma_load_start", 3.5), ("models_loaded", 5.0), ("ready", 5.5),
             ("instruction_start", 5.6), ("instruction_end", 9.0), ("agent_exit", 9.5)]
    (tmp_path / "markers.jsonl").write_text("".join(
        json.dumps({"time": t0 + t, "marker": m, **({"index": 1} if m.startswith("instruction") else {}),
                    **({"status": "COMPLETED"} if m == "instruction_end" else {})}) + "\n" for m, t in marks))
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in [
        {"kind": "vlm", "role": "planner", "seconds": 1.0, "tokens": 40, "t": 1.4, "arrival": t0 + 7.0, "file": "x"},
        {"kind": "report", "status": "COMPLETED", "t": 3.0, "arrival": t0 + 8.7, "file": "x"}]))
    (tmp_path / "meta.json").write_text(json.dumps({"mode": "bench", "label": "syn", "instructions": ["a"]}))
    s = summarize(tmp_path)
    assert s["idle_mib"] == 600
    assert s["peak"]["mib"] == 8600 and s["phases"]["load"]["peak"]["mib"] == 8600  # the loading spike
    assert s["steady"]["median_used_mib"] == 3600
    assert s["vlm"]["latency_s"]["mean"] == 1.0 and s["vlm"]["peak_during_calls"]["mib"] == 3900
    assert s["tasks"]["completed"] == 1 and s["feasibility"]["load_s"] == 1.5


def _tp_rows(label, batches, oom_at=None):
    from gpumon.throughput import _row
    rows = []
    for b in batches:
        if b == oom_at:
            rows.append(_row(label, batch=b, status="oom", peak_alloc_gib=6.9))
            break
        for rep, tps in ((1, 10.0 * b), (2, 10.0 * b + 2)):
            rows.append(_row(label, batch=b, repeat=rep, status="ok", tokens_per_s=tps, peak_alloc_gib=4 + 0.1 * b,
                             proc_gpu_peak_gib=4.5 + 0.1 * b, weights_gib=3.7))
    return rows


def test_throughput_summary_means_repeats_and_marks_the_batch_limit():
    from gpumon.throughput import summarise
    lines = summarise(_tp_rows("4bit_offload", [1, 2, 4]) + _tp_rows("4bit_gpu", [1, 2, 4], oom_at=4))
    by = {(l["config"], l["batch"]): l for l in lines}
    assert by[("4bit_offload", 4)]["tokens_per_s"] == 41.0 and by[("4bit_offload", 4)]["status"] == "ok"
    assert by[("4bit_offload", 1)]["tokens_per_s_per_gib"] == round(11.0 / 4.6, 2)  # throughput per GiB of GPU
    assert by[("4bit_gpu", 4)]["status"] == "oom" and by[("4bit_gpu", 4)]["tokens_per_s"] is None


def test_throughput_report_merges_children_and_names_a_load_failure(tmp_path):
    from gpumon.throughput import _row, _write_rows, report
    _write_rows(tmp_path / "4bit_offload.csv", _tp_rows("4bit_offload", [1, 2]))
    _write_rows(tmp_path / "16bit_offload.csv", [_row("16bit_offload", batch=0, status="load_oom")])
    assert report(tmp_path, figures=False)
    txt = (tmp_path / "summary.txt").read_text()
    assert "4bit_offload    batch 2: 21.0 tokens/s" in txt and "16bit_offload   did not run (load_oom)" in txt
    with open(tmp_path / "results.csv") as f:
        assert len(list(csv.DictReader(f))) == 5


def test_every_throughput_configuration_is_a_valid_vlm_setting():
    from gpumon.throughput import CONFIGS
    from vlm.config import VLMConfig
    for precision, offload in CONFIGS.values():
        VLMConfig(quantization=precision, per_layer_embeddings_on_cpu=offload)  # raises on an unknown precision
