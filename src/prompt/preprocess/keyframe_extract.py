import os
import cv2
import numpy as np
import logging
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from google.cloud import videointelligence_v1 as vi

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def detect_shot_intervals_local(video_path: str) -> list[tuple[float, float]]:
    logger.info(f"Detecting shot intervals for video: {video_path}")
    client = vi.VideoIntelligenceServiceClient()
    with open(video_path, "rb") as f:
        input_content = f.read()

    op = client.annotate_video(
        request={
            "input_content": input_content,
            "features": [vi.Feature.SHOT_CHANGE_DETECTION],
        }
    )
    response = op.result(timeout=300)
    if not response or not response.annotation_results:
        logger.error("No annotation_results found in video intelligence response.")
        return []
    result = response.annotation_results[0]
    intervals = []
    for shot in result.shot_annotations:
        start = (
            shot.start_time_offset.seconds + shot.start_time_offset.microseconds / 1e6
        )
        end = shot.end_time_offset.seconds + shot.end_time_offset.microseconds / 1e6
        intervals.append((start, end))
    logger.info(f"Detected {len(intervals)} shot intervals.")
    return intervals


def color_histogram(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def sample_frames_per_shot(
    video_path: str, start: float, end: float, step: float = 1.0
) -> list[np.ndarray]:
    logger.info(f"Sampling frames from {start:.2f}s to {end:.2f}s every {step:.2f}s")
    cap = cv2.VideoCapture(video_path)
    frames = []
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame at {t:.2f}s")
            break
        frames.append(frame)
        t += step
    cap.release()
    logger.info(f"Sampled {len(frames)} frames for shot interval.")
    return frames


def kmeans_init(features: np.ndarray):
    n, _ = features.shape
    k = int(np.sqrt(n)) or 1
    idx = np.random.choice(n, k, replace=False)
    centers = features[idx]
    clusters = np.argmin(cdist(features, centers), axis=1)
    return clusters, centers


def kmeans_silhouette(features: np.ndarray):
    k = max(int(np.sqrt(len(features))), 2)
    best_k, best_score = k, -1
    clusters, centers = kmeans_init(features)
    best_centers = centers.copy()
    while k > 2:
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        clusters = np.where(clusters == j, i, clusters)
        clusters = np.where(clusters > j, clusters - 1, clusters)
        new_centers = []
        for cid in range(k - 1):
            cluster_feats = features[clusters == cid]
            if cluster_feats.size == 0:
                continue
            mean_vec = np.mean(cluster_feats, axis=0)
            idx_close = np.argmin(np.linalg.norm(cluster_feats - mean_vec, axis=1))
            new_centers.append(cluster_feats[idx_close])
        centers = new_centers
        k -= 1
        if len(np.unique(clusters)) > 1:
            score = silhouette_score(features, clusters)
            if score > best_score:
                best_score, best_k = score, k
                best_centers = centers.copy()
    center_indices = []
    for c in best_centers:
        matches = np.where((features == c).all(axis=1))[0]
        if matches.size > 0:
            center_indices.append(int(matches[0]))
    logger.info(f"KMeans silhouette: best_k={best_k}, best_score={best_score:.4f}")
    return best_k, best_centers, center_indices


def redundancy_filter(
    video_path: str, indices: list[int], threshold: float
) -> list[int]:
    logger.info(f"Filtering redundant frames with threshold {threshold}")
    histos = []
    cap = cv2.VideoCapture(video_path)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            histos.append(color_histogram(frame))
    cap.release()
    keep = []
    for i, h in enumerate(histos):
        if not any(
            np.dot(h, nh) / (np.linalg.norm(h) * np.linalg.norm(nh)) > threshold
            for nh in histos[:i]
        ):
            keep.append(indices[i])
    logger.info(f"Filtered down to {len(keep)} non-redundant frames.")
    return keep


def extract_and_save_keyframes(
    video_path: str,
    output_dir: str,
    start_index: int = 0,
    step: float = 1.0,
    threshold: float = 0.7,
    k_min: int = 2,
    k_max: int = 8,
) -> int:
    logger.info(f"Starting keyframe extraction for {video_path}")
    os.makedirs(output_dir, exist_ok=True)

    # Get FPS to convert seconds to frame indices
    cap_meta = cv2.VideoCapture(video_path)
    video_fps = cap_meta.get(cv2.CAP_PROP_FPS) or 1.0
    cap_meta.release()

    intervals = detect_shot_intervals_local(video_path)
    cap = cv2.VideoCapture(video_path)
    output_idx = start_index

    for shot_idx, (start, end) in enumerate(intervals):
        logger.info(
            f"Processing shot {shot_idx + 1}/{len(intervals)}: {start:.2f}s to {end:.2f}s"
        )

        # Sample frames & extract features
        frames = sample_frames_per_shot(video_path, start, end, step)
        feats = (
            np.vstack([color_histogram(f) for f in frames])
            if frames
            else np.empty((0,))
        )

        # Determine intra-shot keyframe indices
        if feats.size < k_min or feats.ndim == 1:
            idxs = list(range(len(frames)))
        else:
            _, centers, cidxs = kmeans_silhouette(feats)
            idxs = cidxs

        # Map to global frame numbers and dedupe
        global_idxs = [int(start * video_fps) + i for i in idxs]
        filtered = redundancy_filter(video_path, global_idxs, threshold)

        # Save each keyframe sequentially into output_dir
        for frame_no in filtered:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                continue
            out_path = os.path.join(output_dir, f"image_{output_idx:03d}.jpg")
            cv2.imwrite(out_path, frame)
            output_idx += 1
        logger.info(
            f"Shot {shot_idx + 1}: saved {len(filtered)} keyframes. Total so far: {output_idx}"
        )

    cap.release()
    logger.info(f"Extraction complete. Total frames saved: {output_idx}")
    return output_idx


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "acmmm2025-grand-challenge-gg-credentials.json"
    )
    video_file = "C:\\Users\\tungd\\OneDrive - MSFT\\Second Year\\ML\\ACMMM25 - Grand Challenge on Multimedia Verification\\ACMMM25-Grand-Challenge-Geolocation\\src\\data\\input_data\\ID239\\ID239-1 drone-tunnel.mp4"
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)
    extract_and_save_keyframes(video_file, images_dir)
