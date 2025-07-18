import os

import cv2
import numpy as np
from google.cloud import videointelligence_v1 as vi
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---- Step 1: Detect shot intervals from local video content ----
def detect_shot_intervals_local(video_path: str) -> list[tuple[float, float]]:
    client = vi.VideoIntelligenceServiceClient()
    with open(video_path, "rb") as f:
        input_content = f.read()

    op = client.annotate_video(
        request={
            "input_content": input_content,
            "features": [vi.Feature.SHOT_CHANGE_DETECTION],
        }
    )
    result = op.result(timeout=300).annotation_results[0]
    intervals = []
    for shot in result.shot_annotations:
        start = (
            shot.start_time_offset.seconds + shot.start_time_offset.microseconds / 1e6
        )
        end = shot.end_time_offset.seconds + shot.end_time_offset.microseconds / 1e6
        intervals.append((start, end))
    return intervals


# ---- Step 2: Sample frames every 1 second within a shot ----
def sample_frames_per_shot(
    video_path: str, start: float, end: float, step: float = 1.0
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        t += step
    cap.release()
    return frames


# ---- Step 3: Simple HSV histogram feature embedding ----
def hist_embedding_from_array(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


# ---- Step 4: Determine optimal k via silhouette and select keyframes ----
def select_keyframes_by_shot(
    frames: list[np.ndarray], k_min: int = 2, k_max: int = 8
) -> list[np.ndarray]:
    # Extract embeddings for all frames
    embeddings = np.vstack([hist_embedding_from_array(f) for f in frames])
    # embeddings = get_transvpr_embeddings(frames, model)
    n_samples = len(embeddings)
    if n_samples < k_min:
        return frames  # not enough frames to cluster

    # Ensure k_max doesn't exceed reasonable limits for silhouette score
    # Silhouette score requires at least 2 samples and k < n_samples
    effective_k_max = min(k_max, n_samples - 1)
    effective_k_min = min(k_min, effective_k_max)

    # If we can't do proper clustering, return a subset
    if effective_k_min >= n_samples:
        return frames[:effective_k_min] if len(frames) >= effective_k_min else frames

    best_k = effective_k_min
    best_score = -1

    for k in range(effective_k_min, effective_k_max + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)

            # Check if we have at least 2 different labels for silhouette score
            unique_labels = len(np.unique(labels))
            if unique_labels < 2:
                continue

            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except ValueError as e:
            print(f"⚠️ Warning: Clustering with k={k} failed: {e}")
            continue

    # Final clustering with best_k
    try:
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(embeddings)
        labels = km.labels_
        centers = km.cluster_centers_

        # Select representative frame closest to each centroid
        keyframes = []
        for cid in range(best_k):
            idxs = np.where(labels == cid)[0]
            if len(idxs) == 0:
                continue
            dists = np.linalg.norm(embeddings[idxs] - centers[cid], axis=1)
            best_idx = idxs[np.argmin(dists)]
            keyframes.append(frames[best_idx])

        return keyframes if keyframes else frames[:1]  # Return at least one frame

    except Exception as e:
        print(f"⚠️ Warning: Final clustering failed: {e}. Returning original frames.")
        return frames


# ---- Step 5: Save keyframes to folder ----
def save_keyframes_to_folder(
    keyframes: list[np.ndarray], out_dir: str, start_index: int = 0
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    idx = start_index
    for frame in keyframes:
        out_path = os.path.join(out_dir, f"frame_{idx:04d}.jpg")
        cv2.imwrite(out_path, frame)
        idx += 1
    return idx


# ---- Main pipeline ----
def extract_keyframes(
    video_file: str,
    images_dir: str,
    start_index: int = 0,
    step: float = 1.0,
    k_min: int = 2,
    k_max: int = 8,
) -> None:
    os.makedirs(images_dir, exist_ok=True)
    print("🚀 Detecting shot intervals...")
    intervals = detect_shot_intervals_local(video_file)
    print(f"⏱ Detected {len(intervals)} shots.")

    index = start_index
    for i, (start, end) in enumerate(intervals):
        print(f"🔍 Processing shot {i + 1}: {start:.2f}s to {end:.2f}s")
        frames = sample_frames_per_shot(video_file, start, end, step)
        if not frames:
            print(f"⚠️ No frames sampled for shot {i + 1}, skipping.")
            continue
        keyframes = select_keyframes_by_shot(frames, k_min, k_max)
        index = save_keyframes_to_folder(keyframes, images_dir, start_index=index)
        print(f"✅ Shot {i + 1}: saved {len(keyframes)} keyframes.")

    print(f"🎉 Extraction complete. Total frames: {index}")


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "acmmm2025-grand-challenge-gg-credentials.json"
    )
    video_file = "C:\\Users\\tungd\\OneDrive - MSFT\\Second Year\\ML\\ACMMM25 - Grand Challenge on Multimedia Verification\\G3-Original\\g3\\data\\input_data\\ID264\\Snapinsta.app_video_410066595_6884160505002934_8075895171049602368_n.mp4"
    images_dir = "g3/data/prompt_data/ID264/images"
    os.makedirs(images_dir, exist_ok=True)
    extract_keyframes(video_file, images_dir)
