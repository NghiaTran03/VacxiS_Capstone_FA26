# VacxiS

## Cấu hình LLM cho query và generation

Từ thư mục gốc của repo, sao chép `.env.example` thành `.env` và điền API key của provider bạn dùng. Ví dụ với PowerShell:

```powershell
Copy-Item .env.example .env
uv run --env-file .env vacxis-generation --query "Verorab có những phản ứng phụ nào?"
```

| Biến | Công dụng |
| --- | --- |
| `VACXIS_LLM_MODEL` | Model tổng hợp câu trả lời. `--model` ghi đè giá trị này. |
| `VACXIS_QUERY_PLANNER_MODEL` | Model lập kế hoạch tìm kiếm cho chế độ một câu online. Bỏ trống để tắt planner. `--query-planner-model` ghi đè giá trị này. Chọn model hỗ trợ đầu ra JSON object. |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` | Key của provider tương ứng với model. Có thể dùng key riêng cho từng provider khi hai bước dùng model khác nhau. |
| `LLM_API_KEY` | Key chung được truyền trực tiếp cho cả hai lần gọi. Chỉ đặt khi cả hai model dùng cùng key. |

`uv run --env-file .env` nạp cấu hình vào môi trường tiến trình. Chế độ batch `--input-file` và chế độ offline `--dry-run` bỏ qua cấu hình planner từ file.
