"""Kiểm thử đơn vị cho module matcher."""

import numpy as np
import pytest
from facecls.matcher import KnownFaceIndex


def test_known_face_index_exact_match():
    # Tạo 2 người với các vector giả lập 512 chiều
    dim = 512
    v_a = np.zeros(dim, dtype=np.float32)
    v_a[0] = 1.0

    v_b = np.zeros(dim, dtype=np.float32)
    v_b[1] = 1.0

    known = {
        "person_a": [v_a],
        "person_b": [v_b],
    }

    index = KnownFaceIndex(known, engine="insightface")
    assert index.total_vectors == 2

    # Match vector khớp hoàn toàn với person_a
    person, reason, conf = index.match_single(v_a, threshold=0.35, margin_threshold=0.04)
    assert person == "person_a"
    assert conf == 100.0


def test_known_face_index_margin_threshold():
    dim = 512
    # Hai vector rất gần nhau
    v1 = np.zeros(dim, dtype=np.float32)
    v1[0] = 1.0

    v2 = np.zeros(dim, dtype=np.float32)
    v2[0] = 0.99
    v2[1] = 0.1
    v2 /= np.linalg.norm(v2)

    known = {
        "person_1": [v1],
        "person_2": [v2],
    }
    index = KnownFaceIndex(known, engine="insightface")

    # Vector query
    q = v1.copy()
    # Khi margin_threshold lớn hơn chênh lệch giữa person_1 và person_2 -> mập mờ (unknown)
    person, reason, conf = index.match_single(q, threshold=0.35, margin_threshold=0.1)
    assert person == "unknown"
    assert "mập mờ" in reason


def test_known_face_index_below_threshold():
    dim = 512
    v1 = np.zeros(dim, dtype=np.float32)
    v1[0] = 1.0

    known = {"person_1": [v1]}
    index = KnownFaceIndex(known, engine="insightface")

    # Vector vuông góc (sim = 0)
    q = np.zeros(dim, dtype=np.float32)
    q[2] = 1.0

    person, reason, conf = index.match_single(q, threshold=0.35)
    assert person == "unknown"
    assert "không khớp" in reason
