import glob
import os
import shutil
import sys

# Đảm bảo console trên Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DIRECTORIES = [
    "data/raw",
    "data/interim",
    "data/processed",
    "models/baselines",
    "models/gnns",
    "results/metrics",
    "src/data_prep",
    "src/models",
    "src/training",
    "src/evaluation",
    "app/dashboard",
    "docs",
]

GITIGNORE_CONTENT = """# Virtual Environment
env/
venv/
.venv/

# Python cache & bytecode
__pycache__/
*.py[cod]
*$py.class
*.pyc

# Data directories & heavy data files
data/
*.csv
*.parquet
*.zip

# Trained model weights & checkpoints
models/
*.pt
*.pth

# OS & IDE files
.DS_Store
Thumbs.db
.vscode/
.idea/
"""


def safe_move(src_path: str, dst_dir: str) -> bool:
    """Di chuyển file an toàn sang thư mục đích (bỏ qua nếu file không tồn tại)."""
    if not os.path.isfile(src_path):
        return False

    os.makedirs(dst_dir, exist_ok=True)
    filename = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, filename)

    # Nếu đường dẫn nguồn và đích trùng nhau thì bỏ qua
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return False

    # Nếu ở đích đã có file trùng tên thì ghi đè
    if os.path.exists(dst_path):
        os.remove(dst_path)

    shutil.move(src_path, dst_path)
    print(f"   [+] Đã chuyển: {src_path:<38} -> {dst_path}")
    return True


def move_by_candidates(filenames: list[str], search_dirs: list[str], dst_dir: str):
    """Tìm danh sách file trong các thư mục search_dirs và chuyển về dst_dir."""
    for fname in filenames:
        for sdir in search_dirs:
            candidate = os.path.join(sdir, fname) if sdir else fname
            if safe_move(candidate, dst_dir):
                break


def move_by_glob(patterns: list[str], search_dirs: list[str], dst_dir: str):
    """Tìm các file khớp wildcard pattern trong search_dirs và chuyển về dst_dir."""
    for sdir in search_dirs:
        for pat in patterns:
            full_pattern = os.path.join(sdir, pat) if sdir else pat
            for matched_file in glob.glob(full_pattern):
                safe_move(matched_file, dst_dir)


def main():
    print("=" * 80)
    print("TỔ CHỨC LẠI WORKSPACE DỰ ÁN THEO CHUẨN MLOPS (GITHUB READY)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Tạo cấu trúc thư mục chuẩn MLOps
    # -------------------------------------------------------------------------
    print("\n[1/3] Đang khởi tạo cấu trúc thư mục MLOps...")
    for d in DIRECTORIES:
        os.makedirs(d, exist_ok=True)
        print(f"   [+] Đã đảm bảo thư mục: {d}/")

    # -------------------------------------------------------------------------
    # 2. Dọn dẹp và di chuyển các file về đúng phân khu
    # -------------------------------------------------------------------------
    print("\n[2/3] Đang dọn dẹp và di chuyển các file về đúng thư mục...")

    # 2.1. NF-UQ-NIDS-v2.csv và các file .zip -> data/raw/
    move_by_candidates(["NF-UQ-NIDS-v2.csv"], ["data", "."], "data/raw")
    move_by_glob(["*.zip"], ["data", "."], "data/raw")

    # 2.2. train.csv, val.csv, test_seen.csv, test_unseen.csv -> data/interim/
    move_by_candidates(
        ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"],
        ["data", "."],
        "data/interim",
    )

    # 2.3. Tất cả *.parquet và preprocessor_params.json -> data/processed/
    move_by_glob(["*.parquet"], ["data", "."], "data/processed")
    move_by_candidates(["preprocessor_params.json"], ["data", "."], "data/processed")

    # 2.4. xgboost_baseline_b0.json -> models/baselines/
    move_by_candidates(["xgboost_baseline_b0.json"], ["data", "."], "models/baselines")

    # 2.5. Tất cả các file mô hình *.pt -> models/gnns/
    move_by_glob(["*.pt"], ["data", "."], "models/gnns")

    # 2.6. Tất cả các file báo cáo chỉ số *_metrics.json -> results/metrics/
    move_by_glob(["*_metrics.json"], ["data", "."], "results/metrics")

    # 2.7. Các script tiền xử lý -> src/data_prep/
    move_by_candidates(
        ["eda_check.py", "data_split.py", "preprocess_and_graph.py"],
        ["."],
        "src/data_prep",
    )

    # 2.8. Các script huấn luyện -> src/training/
    move_by_candidates(
        [
            "train_baseline_b0.py",
            "train_gnn_g0.py",
            "train_gnn_g0_full.py",
            "train_gnn_g1.py",
            "train_gnn_g1_v2.py",
        ],
        ["."],
        "src/training",
    )

    # -------------------------------------------------------------------------
    # 3. Khởi tạo các file cấu hình GitHub (.gitignore, requirements.txt, README.md)
    # -------------------------------------------------------------------------
    print("\n[3/3] Đang khởi tạo các file cấu hình GitHub...")

    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(GITIGNORE_CONTENT)
    print("   [+] Đã tạo file '.gitignore' chuẩn.")

    # Chạy pip freeze > requirements.txt (ưu tiên pip của môi trường hiện tại)
    ret = os.system("pip freeze > requirements.txt")
    if ret != 0 or not os.path.exists("requirements.txt") or os.path.getsize("requirements.txt") == 0:
        os.system(r".\env\Scripts\pip.exe freeze > requirements.txt")
    print("   [+] Đã xuất danh sách thư viện ra 'requirements.txt'.")

    if not os.path.exists("README.md"):
        with open("README.md", "w", encoding="utf-8") as f:
            f.write("")
    print("   [+] Đã tạo file 'README.md' trống.")

    print("\n" + "=" * 80)
    print("HOÀN TẤT TỔ CHỨC WORKSPACE! CẤU TRÚC HIỆN TẠI:")
    print("=" * 80)
    for d in DIRECTORIES:
        files = os.listdir(d) if os.path.exists(d) else []
        print(f" - {d:<20} : {len(files)} items -> {files}")
    print("=" * 80)


if __name__ == "__main__":
    main()
