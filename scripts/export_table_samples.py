"""
Trích xuất 5 chunk mẫu có has_table: True từ chunks_storage/chunks.parquet
và ghi ra tập tin ./table.md.
"""
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path


def export_table_samples():
    parquet_path = Path("chunks_storage/chunks.parquet")
    output_path = Path("table.md")

    table = pq.read_table(parquet_path)

    # Lọc các chunk có metadata.has_table == True
    has_table_array = pc.struct_field(table["metadata"], "has_table")
    filtered_table = table.filter(has_table_array)

    total_with_table = filtered_table.num_rows
    print(f"Tổng số chunk có has_table == True: {total_with_table}")

    # Chọn 5 chunk đa dạng từ các vaccine khác nhau
    rows = filtered_table.to_pylist()
    
    # Chọn đại diện các loại: thuần table và mixed từ nhiều chuyên mục
    sample_indices = [0, 8, 20, 50, 100]
    selected_samples = [rows[i] for i in sample_indices if i < len(rows)]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 5 Chunks Mẫu Có `has_table: True` Trích Xuất Từ `chunks_storage/chunks.parquet`\n\n")
        f.write(f"- **Tổng số chunk có bảng trong kho**: {total_with_table} / {table.num_rows}\n")
        f.write(f"- **Nguồn dữ liệu**: `{parquet_path}`\n\n")
        f.write("---\n\n")

        for idx, row in enumerate(selected_samples, 1):
            chunk_id = row["chunk_id"]
            doc_title = row["doc_title"]
            category = row["category"]
            header = row["header"]
            parent_hdr = row["parent_header"]
            content_type = row["content_type"]
            text = row["text"]
            words = len(text.split())

            f.write(f"## Mẫu {idx}: Chunk `{chunk_id}`\n\n")
            f.write(f"- **Tài liệu**: {doc_title}\n")
            f.write(f"- **Chuyên mục (`category`)**: `{category}`\n")
            f.write(f"- **Tiêu đề mục (`header`)**: `{header}`\n")
            f.write(f"- **Tiêu đề cha (`parent_header`)**: `{parent_hdr}`\n")
            f.write(f"- **Loại nội dung (`content_type`)**: `{content_type}`\n")
            f.write(f"- **Số lượng từ**: {words} words\n\n")
            f.write("### Nội dung chi tiết chunk:\n\n")
            f.write("````markdown\n")
            f.write(text.strip())
            f.write("\n````\n\n")
            f.write("---\n\n")

    print(f"Đã xuất 5 chunk mẫu thành công vào: {output_path.resolve()}")


if __name__ == "__main__":
    export_table_samples()
