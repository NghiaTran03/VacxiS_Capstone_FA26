"""
Script lấy tên các folder con trong SOURCE_DIRS và lọc vaccine_name_mapping.json
để tạo ra file 76_vaccine_names_mapping.json chỉ chứa các vaccine thực tế có trong kho dữ liệu.
"""
import json
from pathlib import Path

# Danh sách các thư mục nguồn theo quy định
SOURCE_DIRS = [
    "CLEANED/cleaned_nghia",
    "CLEANED/HIEU/hieu",
    "CLEANED/HUNG",
    "CLEANED/MANH/MANH(HIEU+HUNG_LAM)",
]


def get_subfolder_names(project_root: Path) -> dict[str, str]:
    """
    Quét qua các thư mục trong SOURCE_DIRS, lấy tên folder con (folder_id)
    chứa ít nhất 1 file .md.
    Trả về dict ánh xạ folder_id -> source_dir.
    """
    folder_sources: dict[str, str] = {}

    for src_rel in SOURCE_DIRS:
        src_path = project_root / src_rel
        if not src_path.exists():
            # Dự phòng trường hợp đường dẫn MANH không lồng thư mục
            alt_path = project_root / src_rel.replace("CLEANED/MANH/MANH", "CLEANED/MANH")
            if alt_path.exists():
                src_path = alt_path
            else:
                print(f"[CẢNH BÁO] Không tìm thấy thư mục: {src_path}")
                continue

        for item in sorted(src_path.iterdir()):
            if item.is_dir():
                md_files = list(item.glob("*.md"))
                if md_files:
                    folder_sources[item.name] = str(src_rel)

    return folder_sources


def filter_mapping(project_root: Path) -> Path:
    mapping_file = project_root / "vaccine_name_mapping.json"
    output_file = project_root / "76_vaccine_names_mapping.json"

    with open(mapping_file, "r", encoding="utf-8") as f:
        full_mapping: dict[str, str] = json.load(f)

    folder_sources = get_subfolder_names(project_root)
    print(f"Tổng số thư mục con tìm thấy trong SOURCE_DIRS: {len(folder_sources)}")

    filtered_mapping: dict[str, str] = {}
    missing_in_mapping = []

    for folder_id in folder_sources:
        if folder_id in full_mapping:
            # Làm sạch ký tự xuống dòng nếu có
            name = full_mapping[folder_id].replace("\n", " ").strip()
            filtered_mapping[folder_id] = name
        else:
            missing_in_mapping.append(folder_id)
            filtered_mapping[folder_id] = folder_id

    if missing_in_mapping:
        print(f"[LƯU Ý] Các folder_id không có trong vaccine_name_mapping.json (dùng folder_id làm tên): {missing_in_mapping}")

    # Ghi ra file JSON kết quả
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_mapping, f, ensure_ascii=False, indent=2)

    print(f"Đã xuất thành công {len(filtered_mapping)} mục vào: {output_file.name}")
    return output_file


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    filter_mapping(root_dir)
