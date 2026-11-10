"""Module đối soát và so khớp khuôn mặt véc-tơ hóa (Matrix Multiplication & Top-K)."""

from typing import Dict, List, Tuple, Optional
import numpy as np


class KnownFaceIndex:
    """
    Chỉ mục khuôn mặt đã biết, tối ưu hóa bằng ma trận duy nhất (N, D).
    Thay vì duyệt từng người và dựng lại mảng numpy ở mỗi ảnh,
    toàn bộ điểm tương đồng được tính trong MỘT phép nhân ma trận (Matmul).
    """

    def __init__(self, known_faces: Dict[str, List[np.ndarray]], engine: str = "insightface"):
        self.engine = engine.lower()
        self.person_names: List[str] = sorted(list(known_faces.keys()))
        self.person_slices: Dict[str, slice] = {}

        all_encodings = []
        current_idx = 0

        for name in self.person_names:
            encs = known_faces.get(name, [])
            if not encs:
                continue
            encs_arr = np.asarray(encs, dtype=np.float32)
            if encs_arr.ndim == 1:
                encs_arr = encs_arr.reshape(1, -1)

            # Đảm bảo chuẩn hóa L2
            norms = np.linalg.norm(encs_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            encs_norm = encs_arr / norms

            count = encs_norm.shape[0]
            self.person_slices[name] = slice(current_idx, current_idx + count)
            current_idx += count
            all_encodings.append(encs_norm)

        if all_encodings:
            self.matrix = np.vstack(all_encodings)  # Kích thước: (Total_Vectors, Dim)
        else:
            self.matrix = np.empty((0, 512 if self.engine == "insightface" else 128), dtype=np.float32)

    @property
    def total_vectors(self) -> int:
        return self.matrix.shape[0]

    def match_single(
        self,
        face_enc: np.ndarray,
        top_k: int = 3,
        threshold: float = 0.35,
        margin_threshold: float = 0.04,
        unknown_prefix: str = "unknown"
    ) -> Tuple[str, str, float]:
        """
        So khớp 1 vector khuôn mặt với toàn bộ tập dữ liệu đã biết.
        Returns:
            (person_name, reason_description, confidence_percentage)
        """
        if self.matrix.shape[0] == 0:
            return unknown_prefix, "chưa có dữ liệu known faces", 0.0

        face_enc = np.asarray(face_enc, dtype=np.float32).ravel()
        norm = np.linalg.norm(face_enc)
        if norm > 0:
            face_enc = face_enc / norm

        if self.engine == "insightface":
            # 1 phép nhân ma trận tính Cosine Similarity với TOÀN BỘ database
            all_sims = self.matrix @ face_enc  # Shape: (Total_Vectors,)

            scores: Dict[str, float] = {}
            for person, s_slice in self.person_slices.items():
                p_sims = all_sims[s_slice]
                if len(p_sims) == 0:
                    continue
                k = min(top_k, len(p_sims))
                if k == len(p_sims):
                    scores[person] = float(np.mean(p_sims))
                else:
                    # np.partition tối ưu hơn np.sort: O(N) thay vì O(N log N)
                    top_k_vals = np.partition(p_sims, -k)[-k:]
                    scores[person] = float(np.mean(top_k_vals))

            if not scores:
                return unknown_prefix, "không có khuôn mặt phù hợp", 0.0

            sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            best_person, best_sim = sorted_candidates[0]

            # Kiểm tra biên sai biệt (Margin check)
            if len(sorted_candidates) > 1:
                second_person, second_sim = sorted_candidates[1]
                if best_sim >= threshold and (best_sim - second_sim) < margin_threshold:
                    return (
                        unknown_prefix,
                        f"mập mờ giữa {best_person} ({best_sim:.3f}) và {second_person} ({second_sim:.3f})",
                        0.0
                    )

            confidence = round(best_sim * 100, 1)
            if best_sim >= threshold:
                return best_person, f"confidence {confidence}% (ArcFace sim={best_sim:.3f})", confidence
            else:
                return (
                    unknown_prefix,
                    f"không khớp (ArcFace sim={best_sim:.3f} < threshold={threshold})",
                    0.0
                )

        else:
            # dlib (Euclidean distance)
            diff = self.matrix - face_enc
            all_dists = np.linalg.norm(diff, axis=1)

            scores = {}
            for person, s_slice in self.person_slices.items():
                p_dists = all_dists[s_slice]
                if len(p_dists) == 0:
                    continue
                k = min(top_k, len(p_dists))
                if k == len(p_dists):
                    scores[person] = float(np.mean(p_dists))
                else:
                    top_k_vals = np.partition(p_dists, k)[:k]
                    scores[person] = float(np.mean(top_k_vals))

            if not scores:
                return unknown_prefix, "không có khuôn mặt phù hợp", 0.0

            sorted_candidates = sorted(scores.items(), key=lambda item: item[1])
            best_person, best_dist = sorted_candidates[0]

            if len(sorted_candidates) > 1:
                second_person, second_dist = sorted_candidates[1]
                if best_dist <= threshold and (second_dist - best_dist) < margin_threshold:
                    return (
                        unknown_prefix,
                        f"mập mờ giữa {best_person} ({best_dist:.3f}) và {second_person} ({second_dist:.3f})",
                        0.0
                    )

            if best_dist <= threshold:
                confidence = round((1.0 - best_dist) * 100, 1)
                return best_person, f"confidence {confidence}% (dist={best_dist:.3f})", confidence
            else:
                return (
                    unknown_prefix,
                    f"không khớp (dist={best_dist:.3f} > threshold={threshold})",
                    0.0
                )

    def match_video(
        self,
        frames: List[np.ndarray],
        engine_obj,
        top_k: int = 3,
        threshold: float = 0.35,
        margin_threshold: float = 0.04,
        unknown_prefix: str = "unknown",
        return_bbox: bool = False,
    ) -> Union[Tuple[str, str, float], Tuple[str, str, float, Optional[Dict[str, Any]]]]:
        """
        Nhận diện video bằng cơ chế trích xuất mặt lớn nhất từng frame và bỏ phiếu (Majority Voting).
        det_thresh được điều chỉnh linh hoạt cho video để tránh bỏ sót do motion blur.
        """
        if not frames:
            res = (unknown_prefix, "video không có khung hình hợp lệ", 0.0)
            return (*res, None) if return_bbox else res

        person_votes: Dict[str, List[float]] = {}
        valid_frames_count = 0
        best_bbox = None
        best_det_score = -1.0

        # Ngưỡng phát hiện mềm hơn cho khung hình video (motion blur / nén MP4)
        video_det_thresh = min(getattr(engine_obj, "det_thresh", 0.6), 0.42)

        for frame in frames:
            # Ưu tiên lấy cả bbox nếu engine hỗ trợ
            if hasattr(engine_obj, "extract_faces_with_bboxes"):
                faces_data = engine_obj.extract_faces_with_bboxes(
                    frame, filter_quality=True, det_thresh=video_det_thresh
                )
                if not faces_data:
                    continue

                valid_frames_count += 1
                # Chọn khuôn mặt có điểm phát hiện hoặc diện tích tốt nhất
                top_face = max(faces_data, key=lambda f: f.get("det_score", 0.0))
                face_enc = top_face.get("embedding")
                det_sc = float(top_face.get("det_score", 0.0))
                if det_sc > best_det_score:
                    best_det_score = det_sc
                    best_bbox = top_face.get("bbox")
            else:
                embs = engine_obj.extract_face_embeddings(
                    frame, filter_quality=True, select_single=True, det_thresh=video_det_thresh
                )
                if not embs:
                    continue

                valid_frames_count += 1
                face_enc = embs[0]

            if face_enc is None:
                continue

            person, reason, conf = self.match_single(
                face_enc=face_enc,
                top_k=top_k,
                threshold=threshold,
                margin_threshold=margin_threshold,
                unknown_prefix=unknown_prefix
            )

            if person != unknown_prefix:
                person_votes.setdefault(person, []).append(conf)

        if valid_frames_count == 0:
            res = (unknown_prefix, f"video ({len(frames)} frames): không phát hiện khuôn mặt", 0.0)
            return (*res, None) if return_bbox else res

        if not person_votes:
            res = (unknown_prefix, f"video ({len(frames)} frames): phát hiện mặt ({valid_frames_count} frames) nhưng là người lạ", 0.0)
            return (*res, best_bbox) if return_bbox else res

        # Lấy người có số frame bầu chọn cao nhất và confidence trung bình cao nhất
        top_person, confs = max(person_votes.items(), key=lambda item: (len(item[1]), np.mean(item[1])))
        avg_conf = round(float(np.mean(confs)), 1)
        votes_count = len(confs)

        reason = f"video: {votes_count}/{valid_frames_count} frames khớp {top_person} (conf TB {avg_conf}%)"
        res = (top_person, reason, avg_conf)
        return (*res, best_bbox) if return_bbox else res
