# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The read-only Store selection reads large rows in chunks, without the trajectory values it never uses."""

import json
import random
import tracemalloc

import pytest

import nemo_gym.rollouts.rollout_records as records
from nemo_gym.eval.recovery.evidence import STORE_SKIP_KEYS
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore


_SKIP = frozenset({"ng_trajectory", "ng_model_call_capture"})


def _value(rng: random.Random, depth: int = 0):
    kind = rng.choice(["str", "num", "lit", "list", "obj"] if depth < 3 else ["str", "num", "lit"])
    if kind == "str":
        return rng.choice(['a "quoted" {brace} [bracket], colon: x', "back\\slash\\", "ünï ✓", "", '\\"', ",:}]"])
    if kind == "num":
        return rng.choice([0, -1.5e10, 3, 7.25])
    if kind == "lit":
        return rng.choice([None, True, False])
    if kind == "list":
        return [_value(rng, depth + 1) for _ in range(rng.randrange(4))]
    return {
        rng.choice(["ng_trajectory", "k", '"x"', "a,b"]) + str(i): _value(rng, depth + 1)
        for i in range(rng.randrange(4))
    }


def _row(rng: random.Random) -> dict:
    row = {f"k{i}": _value(rng) for i in range(rng.randrange(1, 5))}
    for key in rng.sample(sorted(_SKIP) + ["response"], rng.randrange(4)):
        row[key] = _value(rng)
    keys = list(row)
    rng.shuffle(keys)
    return {key: row[key] for key in keys}


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 64])
def test_the_projection_equals_the_row_without_its_skipped_top_level_values(seed, chunk):
    rng = random.Random(seed)
    row = _row(rng)
    text = json.dumps(row, ensure_ascii=rng.random() < 0.5, separators=rng.choice([(",", ":"), (", ", ": ")]))
    raw = (" " if rng.random() < 0.3 else "") + text
    data = raw.encode()
    projector = records._TopLevelProjector(_SKIP)
    for start in range(0, len(data), chunk):
        projector.feed(data[start : start + chunk])
    projected = json.loads(projector.result())
    assert {k: v for k, v in projected.items() if k not in _SKIP} == {k: v for k, v in row.items() if k not in _SKIP}
    assert all(projected[key] is None for key in _SKIP & row.keys())


def _store(tmp_path, rows: int, trajectory_bytes: int):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    inputs = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(rows)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in inputs))

    def prepare():
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 2})

    output = artifacts / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for i, original in enumerate(inputs):
            row = dict(original)
            store.allocate_attempt(row)
            trajectory = {"model_calls": [{"text": "t" * 1000, "n": n} for n in range(trajectory_bytes // 1000)]}
            if i % 2:
                store.record_outcome(row | {"reward": 1.0, "response": {"id": f"r{i}"}, "ng_trajectory": trajectory})
            else:
                failure = {
                    "_ng_failure_class": "infrastructure_error",
                    "failure_kind": None,
                    "evaluation_completed": False,
                    "ng_trajectory": trajectory,
                }
                store.record_outcome(row | failure)
    return output


def _peak(read):
    tracemalloc.start()
    try:
        return read(), tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_the_coverage_a_continuation_checks_is_the_same_without_the_skipped_values(tmp_path, monkeypatch):
    # The in-job admission compares this coverage with rollouts_coverage.json before it continues a segment.
    output = _store(tmp_path, rows=4, trajectory_bytes=2 << 20)
    monkeypatch.setattr(records, "PROJECTED_ROW_BYTES", 1 << 20)
    full = RolloutStore.read(output, import_legacy=False)
    projected = RolloutStore.read(output, import_legacy=False, skip_keys=STORE_SKIP_KEYS)
    assert projected.coverage() == full.coverage()
    assert full.coverage()["expected"] == 4


def test_a_large_store_selects_the_same_outcomes_with_bounded_memory(tmp_path, monkeypatch):
    output = _store(tmp_path, rows=4, trajectory_bytes=12 << 20)
    full, full_peak = _peak(lambda: RolloutStore.read(output, import_legacy=False))
    monkeypatch.setattr(records, "PROJECTED_ROW_BYTES", 1 << 20)
    projected, peak = _peak(lambda: RolloutStore.read(output, import_legacy=False, skip_keys=STORE_SKIP_KEYS))
    # Each row carries 12 MiB of trajectory; the store indexes neither read with it parsed, only a few chunks.
    assert peak < 6 << 20, peak
    assert [row["_ng_attempt_index"] for row in projected.pending()] == [
        row["_ng_attempt_index"] for row in full.pending()
    ]
    for disposition in ("success", "failure", "omitted"):
        assert projected.selected_records(disposition).keys() == full.selected_records(disposition).keys()
    [record] = list(projected.selected_records("success").values())[:1]
    assert "ng_trajectory" not in record.read() and record.read()["reward"] == 1.0


def test_an_incomplete_final_large_row_is_ignored_like_a_small_one(tmp_path, monkeypatch):
    output = _store(tmp_path, rows=2, trajectory_bytes=2 << 20)
    with output.open("ab") as handle:
        handle.write(b'{"_ng_task_index": 9, "ng_trajectory": "' + b"x" * (3 << 20))
    monkeypatch.setattr(records, "PROJECTED_ROW_BYTES", 1 << 20)
    rows = list(records._indexed_records(output, skip_keys=STORE_SKIP_KEYS))
    assert len(rows) == 1 and "ng_trajectory" not in rows[0][1]


def test_a_resumed_run_indexes_large_rows_without_their_trajectories(tmp_path, monkeypatch):
    output = _store(tmp_path, rows=4, trajectory_bytes=12 << 20)
    monkeypatch.setattr(records, "PROJECTED_ROW_BYTES", 1 << 20)

    inputs = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(4)]

    def resume():
        def prepare():  # the run's own inputs: a resume checks they did not change
            return inputs, RunManifest.create(tmp_path / "source.jsonl", inputs, {}, {}).model_copy(
                update={"max_rollout_attempts": 2}
            )

        with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
            return (
                store.coverage(),
                [r["_ng_attempt_index"] for r in store.pending()],
                store.selected_records("success"),
            )

    (coverage, pending, successes), peak = _peak(resume)
    # 12 MiB of trajectory per row: indexing never parses it, so the peak is a few chunks.
    assert peak < 6 << 20, peak
    assert coverage == RolloutStore.read(output, import_legacy=False).coverage()
    assert pending == [1, 1]  # the two failed rollouts are retried as their second attempt
    # the records still read back whole
    rows = [record.read() for record in successes.values()]
    assert rows and all(len(row["ng_trajectory"]["model_calls"]) == (12 << 20) // 1000 for row in rows)


def test_a_record_reads_back_without_its_model_call_history_in_bounded_memory(tmp_path, monkeypatch):
    # The batch status a resumed run rebuilds reads every success this way: it needs only identities and classes.
    output = _store(tmp_path, rows=2, trajectory_bytes=12 << 20)
    monkeypatch.setattr(records, "PROJECTED_ROW_BYTES", 1 << 20)
    [record] = RolloutStore.read(output, import_legacy=False).selected_records("success").values()
    row, peak = _peak(lambda: record.read(without=records.MODEL_CALL_HISTORY_KEYS))
    assert peak < 6 << 20, peak
    assert "ng_trajectory" not in row and row["reward"] == 1.0 and row["_ng_task_index"] == 1
    assert "ng_trajectory" in record.read()  # a plain read is whole


def test_the_end_of_run_judge_check_reads_one_saved_row_at_a_time(tmp_path):
    # Every collection ends with this check; holding all selected rows at once grows with the run's size.
    from nemo_gym.eval.rollout_reverification import due_judge_failures

    inputs = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(8)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in inputs))
    output = tmp_path / "artifacts" / "rollouts.jsonl"
    output.parent.mkdir()

    def prepare():
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 2})

    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for original in inputs:
            row = dict(original)
            store.allocate_attempt(row)
            trajectory = {"model_calls": [{"text": "t" * 1000, "n": n} for n in range(4 << 10)]}  # about 4 MiB
            store.record_outcome(row | {"reward": 1.0, "response": {"id": "r"}, "ng_trajectory": trajectory})
    store = RolloutStore.read(output, import_legacy=False)
    _, one_row = _peak(lambda: next(iter(store.selected_records("success").values())).read())
    due, peak = _peak(lambda: due_judge_failures(output))
    assert due == []
    # eight successes: the check costs about one parsed row, not all of them at once
    assert peak < 1.4 * one_row, (peak, one_row)
