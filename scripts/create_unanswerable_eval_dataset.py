"""Generate unanswerable_eval_dataset.json from the first 100 unanswerable queries in test_dataset.json."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "test_dataset.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "unanswerable_eval_dataset.json"


def enrich_refusal(item: dict[str, Any]) -> str:
    """Enrich the base refusal reference answer into a comprehensive, medical-grade refusal response."""
    base_ref = (item.get("REFERENCE_ANSWER") or "").strip()
    if base_ref and not base_ref.endswith("."):
        base_ref += "."

    q_type = (item.get("QUERY_TYPE") or "").upper()

    if q_type == "ADVERSE_EFFECT":
        elaboration = (
            " Do không có dữ liệu lâm sàng hay thông tin ghi nhận về phản ứng bất lợi này trong tài liệu hướng dẫn sử dụng chính thức của vắc xin, "
            "hệ thống xin phép từ chối trả lời và không đưa ra suy đoán nhằm đảm bảo an toàn y khoa. "
            "Nếu bạn hoặc người thân đang gặp phải các triệu chứng bất thường, vui lòng liên hệ ngay với bác sĩ chuyên khoa hoặc cơ sở y tế gần nhất để được thăm khám và xử trí kịp thời."
        )
    elif q_type == "INDICATION":
        elaboration = (
            " Do tài liệu tra cứu chính thức không đề cập đến việc sử dụng vắc xin cho tình trạng hay đối tượng này, "
            "hệ thống từ chối khuyến cáo áp dụng để tránh cung cấp thông tin chưa được kiểm chứng. "
            "Bạn vui lòng tham khảo ý kiến trực tiếp của bác sĩ hoặc cán bộ y tế tại trung tâm tiêm chủng để được tư vấn phác đồ tiêm phù hợp với tình trạng sức khỏe cụ thể."
        )
    elif q_type == "CONTRAINDICATION":
        elaboration = (
            " Do tài liệu hướng dẫn sử dụng không quy định tình trạng này trong mục chống chỉ định hoặc cảnh báo thận trọng, "
            "hệ thống từ chối đưa ra kết luận về mức độ an toàn. "
            "Bạn nên thông báo chi tiết tiền sử bệnh và tình trạng hiện tại cho bác sĩ khám sàng lọc trước khi tiêm để được chỉ định chính xác."
        )
    elif q_type == "INTERACTION":
        elaboration = (
            " Do tài liệu không cung cấp dữ liệu nghiên cứu về sự tương tác giữa vắc xin và các thuốc hay tình trạng này, "
            "hệ thống từ chối tư vấn dùng kết hợp hoặc xác định khoảng thời gian cách biệt an toàn. "
            "Vui lòng tham khảo ý kiến chuyên môn của bác sĩ trước khi phối hợp điều trị hoặc tiêm chủng."
        )
    elif q_type == "DOSAGE":
        elaboration = (
            " Do tài liệu hướng dẫn sử dụng không có quy định cụ thể về liều lượng hay phác đồ tiêm cho trường hợp đặc thù này, "
            "hệ thống từ chối hướng dẫn chi tiết nhằm phòng tránh nguy cơ sai liều. "
            "Bạn cần tuân thủ nghiêm ngặt chỉ dẫn và phác đồ do bác sĩ tại cơ sở tiêm chủng chỉ định."
        )
    elif q_type == "PRECAUTION":
        elaboration = (
            " Do thông tin cảnh báo thận trọng cho tình huống này không được đề cập trong tài liệu hướng dẫn sử dụng chính thức, "
            "hệ thống từ chối đưa ra khuyến cáo để tránh suy diễn thiếu cơ sở. "
            "Bạn vui lòng trao đổi trực tiếp với bác sĩ tại phòng khám sàng lọc để được đánh giá nguy cơ và tư vấn phù hợp."
        )
    else:
        elaboration = (
            " Do thông tin này không được đề cập trong tài liệu hướng dẫn sử dụng chính thức, "
            "hệ thống xin phép từ chối trả lời nhằm tránh suy diễn thiếu căn cứ y khoa. "
            "Để đảm bảo an toàn, bạn vui lòng liên hệ bác sĩ chuyên khoa hoặc cơ sở y tế uy tín để được tư vấn chính xác."
        )

    if base_ref:
        return f"{base_ref}{elaboration}"
    return f"Tài liệu hướng dẫn sử dụng hiện tại không cung cấp thông tin cho câu hỏi này.{elaboration}"


def create_unanswerable_dataset(
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
    limit: int = 100,
) -> dict[str, Any]:
    """Extract unanswerable queries and format them according to eval_dataset_300_final_fixed.json schema."""
    with open(source_path, "r", encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    unanswerable_records = [
        r for r in records if (r.get("ANSWERABILITY") or r.get("ANSWERBILITY")) == "UNANSWERABLE"
    ][:limit]

    goldens: list[dict[str, Any]] = []
    for item in unanswerable_records:
        golden = {
            "input": item.get("QUERY", ""),
            "expected_output": enrich_refusal(item),
            "context": None,
            "additional_metadata": {
                "query_id": item.get("QUERY_ID"),
                "answerability": "UNANSWERABLE",
                "complexity": None,
                "multi_hop_type": None,
                "query_style": None,
                "source_chunk_ids": None,
                "doc_ids": None,
                "doc_titles": None,
            },
        }
        goldens.append(golden)

    payload: dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source_path.name),
            "limit": limit,
            "total_generated": len(goldens),
            "distribution": {
                "answerability": {
                    "UNANSWERABLE": len(goldens),
                },
                "complexity": None,
                "query_style": None,
            },
        },
        "goldens": goldens,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Create unanswerable evaluation dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to test_dataset.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=100, help="Number of unanswerable queries")
    args = parser.parse_args()

    payload = create_unanswerable_dataset(args.source, args.output, args.limit)
    print(f"Successfully generated {payload['metadata']['total_generated']} unanswerable queries to {args.output}")


if __name__ == "__main__":
    main()
