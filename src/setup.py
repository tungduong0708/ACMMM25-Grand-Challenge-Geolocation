import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

base_path = Path(__file__).parent


def setup(
    local_path: Path,
    repo_id: str,
    filename: str,
    subfolder: str | None = None,
    repo_type: str | None = None,
) -> None:
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)

        cached_path = hf_hub_download(
            repo_id=repo_id,
            subfolder=subfolder,
            filename=filename,
            repo_type=repo_type,
        )
        shutil.copy(cached_path, local_path)


if __name__ == "__main__":
    checkpoint_path = (
        base_path / "data/checkpoints/mercator_finetune_weight.pth"
    ).resolve()
    index_path = (base_path / "data/index/G3.index").resolve()
    database_path = (base_path / "data/dataset/mp16/MP16_Pro_filtered.csv").resolve()

    repo_id = "tduongvn/Checkpoints-ACMMM25"

    setup(checkpoint_path, repo_id, "mercator_finetune_weight.pth")
    setup(index_path, repo_id, "G3.index", "index")
    setup(database_path, repo_id, "MP16_Pro_filtered.csv", "data/mp16")
