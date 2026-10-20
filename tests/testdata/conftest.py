"""Shared fixtures for the testdata test suite."""
from __future__ import annotations

from vacxis_capstone_fa26.testdata.chunks import Chunk


def make_chunk(
    chunk_id: str,
    doc_id: str,
    category: str,
    text: str,
    vaccine: str = "Vax A",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=f"Hướng dẫn sử dụng Vaccine {vaccine}",
        category=category,
        text=text,
        source_file="",
    )


SAMPLE_INDICATION_TEXT = (
    "[CHỦ ĐỀ: Vax A] > [MỤC: CHỈ ĐỊNH]\n"
    "Liều khuyến cáo: 0,5 ml tiêm bắp.\n"
    "Vắc xin được chỉ định cho trẻ em từ 6 tuần tuổi trở lên để phòng ngừa bệnh phế cầu.\n"
    "Không dùng cho người quá mẫn với thành phần của vắc xin."
)

SAMPLE_DOSAGE_TEXT = (
    "[CHỦ ĐỀ: Vax A] > [MỤC: LIỀU LƯỢNG]\n"
    "Tiêm bắp 0,5 ml mỗi liều.\n"
    "Phác đồ 3 liều: 0, 1, 6 tháng. Liều nhắc lại sau 12 tháng nếu cần.\n"
    "Bảo quản ở 2-8°C."
)
