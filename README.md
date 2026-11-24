# VaxiS

Hệ thống RAG thông tin vắc xin: Markdown → chunks Parquet → BGE-M3/Qdrant hybrid search → reranker → câu trả lời có trích dẫn. Module `testdata` tạo bộ câu hỏi và reference answer từ cùng corpus; `evaluation` đánh giá retrieval và generation theo `QUERY_ID`.

## Cấu hình

```bash
# Core dependencies only; retrieval/web stacks are opt-in.
uv sync --group dev
# Use --extra retrieval for BGE/torch/Qdrant commands.
# Use --extra web for the Gradio UI.
cp .env.example .env
```

Điền cấu hình theo `.env.example`:

| Biến | Công dụng |
| --- | --- |
| `OPENROUTER_API_KEY` | Key OpenRouter dùng chung cho mọi lời gọi LLM. |
| `VACXIS_GENERATION_MODEL` | Model trả lời. `--model` ghi đè theo CLI tương ứng. |
| `VACXIS_GENERATION_BASE_URL` | Endpoint riêng cho generation; để trống sẽ dùng `VACXIS_LLM_BASE_URL`. |
| `VACXIS_LLM_BASE_URL` | Endpoint chung cho planner và evaluation; mặc định `https://openrouter.ai/api/v1`. |
| `VACXIS_QUERY_PLANNER_MODEL` | Model planner cho backend website gọi `run_turn()`. CLI và batch không gọi planner. |

Ứng dụng chỉ đọc biến môi trường bằng `os.getenv()`, không đọc `.env` hay gọi `load_dotenv()`. `uv run --env-file .env ...` nạp biến vào môi trường tiến trình. Generation dùng endpoint riêng nếu `VACXIS_GENERATION_BASE_URL` được đặt; planner và evaluation dùng `VACXIS_LLM_BASE_URL`. Không còn backend hay alias LiteLLM.

Online thiếu key/model, timeout, API lỗi hoặc phản hồi không hợp lệ sẽ báo lỗi và dừng. Chỉ chế độ `--dry-run` mới dùng trích xuất offline. Không cần key/model LLM cho chế độ này; retrieval vẫn cần Qdrant và model BGE.

## Chạy pipeline

Chuẩn bị Qdrant tại `localhost:6333`, BGE-M3 tại `models/bge-m3` và reranker tại `models/bge-reranker`. Tài liệu đầu vào nằm trong các thư mục `CLEANED` hiện có, với mapping `75_vaccine_names_mapping.json` hoặc `vaccine_name_mapping.json`.

```bash
# Markdown → Parquet → Qdrant
uv run --extra retrieval vacxis-pipeline prepare

# Hỏi đáp, không planner
uv run --extra retrieval --env-file .env vacxis-pipeline ask --query "Verorab có phản ứng phụ nào?"

# Tạo dataset (GPU host): vLLM rewrite + answer, tách 300, mask triệu chứng phần cuối
uv run generate-test-dataset --vllm-model <model> --checkpoint ckpt.json

# Dataset → batch RAG → evaluation
uv run --extra retrieval --env-file .env vacxis-pipeline benchmark --dataset test_dataset.json

# Chạy toàn bộ: prepare → benchmark (đòi dataset đã có ở --dataset)
uv run --extra retrieval --env-file .env vacxis-pipeline run

# Toàn bộ luồng với generation offline
uv run --extra retrieval vacxis-pipeline run --dry-run
```

`generate-test-dataset` đọc `scrape_dataset/query.parquet` và `chunks_storage/chunks.parquet`: Qwen qua vLLM viết lại câu hỏi theo hướng vaccine và sinh reference answer, sau đó tách tại `--split-at` (mặc định 300): phần đầu viết vào `--out` (mặc định `test_dataset.json`), phần cuối được mask triệu chứng theo chunk bằng chứng (dữ kiện chứng minh) và viết vào `--out-masked` (mặc định `test_dataset_masked.json`). Cờ chính: `--source-queries`, `--vllm-model`, `--vllm-gpu-memory`, `--limit`, `--per-type`, `--distractors`, `--seed`, `--checkpoint`, `--answers-batch-size`, `--rewrite-batch-size`, `--verify-batch-size`, `--filter-no-han`, `--no-validate`, `--strict-validation`, `--dry-run-answers` (answer trích xuất offline thay vì vLLM), và `--relabel` (xác thực lại nhãn/trả lời/bằng chứng từ câu hỏi cuối; `--relabel-scope` `masked` (mặc định) hay `all`; loại các bản ghi vi phạm bất biến). Chạy trên GPU host vì cần vLLM.

Các lệnh điều phối nhận `--project-root`, `--collection-name`, `--qdrant-host`, `--qdrant-port`, `--model-path`, `--reranker-path`, `--top-k`, `--skip-rerank`. Đường dẫn artifact tương đối được tính từ `--project-root`. `prepare`/`run` nhận `--chunks`, `--mapping`, `--recreate`; `benchmark` nhận `--dataset`, `--results`, `--report-json`, `--report-md`, `--max-tokens` (mặc định không giới hạn số token output), `--timeout` (giây, mặc định 120; tăng lên nếu model reasoning chạy dài); `run` nhận `--dataset`, `--results`, `--report-json`, `--report-md`.

Artifact mặc định:

- `chunks_storage/chunks.parquet`
- `test_dataset.json`
- `generation_results.json`
- `evaluation_report.json`
- `evaluation_report.md`

Collection được giữ mặc định; chỉ `--recreate` mới xóa và tạo lại. Point ID dùng UUID ổn định từ chunk ID, nên ingest lại cùng corpus không nhân đôi point. Collection cũ dùng ID số hoặc corpus đã xóa/đổi chunk cần rebuild có chủ đích bằng `prepare --recreate` để loại point cũ. Pipeline không tự chạy bước phụ thuộc sau khi một bước lỗi.

Các CLI `vacxis-generation`, `vacxis-retrieval`, `vacxis-eval`, `vacxis-evaluation` và `generate-test-dataset` vẫn được giữ. `vacxis-generation` không còn các cờ planner/làm rõ; dành chức năng đó cho API Python bên dưới. Reference answer của `generate-test-dataset` dùng Qwen qua vLLM (hoặc trích xuất offline với `--dry-run-answers`); không còn chế độ OpenRouter cho dataset.

## Web UI

Web mặc định chạy kiến trúc chính **Table-aware + LightRAG & Table-aware hybrid**: `hybrid_lightrag`, LightRAG query mode `hybrid`, fusion `rrf_rerank`, collection `VacxiS_Capstone_FA26_table_dual` (bảng được lập chỉ mục bằng `summarize_dual`). Đây là nhánh kết hợp được mô tả trong `results/ablation_study.md`, khác Hybrid baseline trên collection gốc.

Chạy từ thư mục gốc với `.env`, model, collection Table-aware, các vector collection LightRAG, dữ liệu KV và Neo4j đã có sẵn. Các biến `LIGHTRAG_*`, `QDRANT_URL`, `NEO4J_*` cần trỏ đúng dịch vụ của bộ dữ liệu. Lệnh web không chạy `prepare`, không khôi phục snapshot hay tự chuyển về baseline khi thiếu dữ liệu. `--merge-table-chunks` chỉ gộp bảng lúc truy xuất, không thay thế collection đã lập chỉ mục Table-aware. Nếu môi trường đã đủ thư viện, dùng `--no-sync` để giữ nguyên dependencies:

```bash
uv run --no-sync --env-file .env vacxis-web
```

Mở [VacxiS tại localhost](http://127.0.0.1:8000). Giao diện Gradio có khung hỏi đáp và lịch sử ở thanh bên. Mục **Quá trình trả lời** có thể mở/thu gọn, cập nhật thời gian đã chạy và hiện tổng thời gian khi trả lời xong. Câu hỏi làm rõ xuất hiện như một tin nhắn ngắn. Bấm nguồn hoặc **Tài liệu vắc xin** để mở tab trình duyệt mới, xem đoạn trích dẫn và toàn bộ tài liệu Markdown nguồn có sẵn của vắc xin đó.

Trang tài liệu chia nội dung theo category có thể mở/thu gọn. Category chứa đoạn trích dẫn có dấu ✓ và mở sẵn; đoạn được trích dẫn vẫn hiển thị riêng ở đầu trang. Ô **Tìm trong tài liệu** tìm cả trong các mục đang thu gọn, hỗ trợ từ khóa có dấu hoặc không dấu, tô sáng kết quả và chuyển đến kết quả trước/tiếp theo bằng nút ↑/↓ hoặc Enter/Shift+Enter. Nhấn Escape để xóa tìm kiếm và khôi phục các mục đã mở trước đó.

Nếu một đoạn trích bao phủ nhiều category, các mục tương ứng đều được đánh dấu và mở sẵn. Mục **Giao diện** trên trang tài liệu cho phép chọn **Sáng**, **Tối** hoặc **Theo hệ thống**, với nền trung tính và lựa chọn được ghi nhớ trên trình duyệt.

Xem trước giao diện với phản hồi mẫu và tài liệu thật khi chưa chuẩn bị model hoặc dịch vụ retrieval:

```bash
uv run --extra web vacxis-web --preview
```

Chế độ này hiển thị nhãn **Bản xem trước giao diện**, mô phỏng các bước xử lý và không gọi pipeline AI. Chạy lệnh thường ở trên để dùng kiến trúc chính thực tế.

Thêm `--share` để tạo đường dẫn Gradio công khai cho cả chat và trang tài liệu:

```bash
uv run --no-sync --env-file .env vacxis-web --share --port 8000
```

Terminal sẽ in `Running on public URL: https://....gradio.live`. Gửi đường dẫn này cho người dùng; máy chủ vẫn là máy đang chạy lệnh và phải giữ máy bật, có Internet, terminal đang chạy. Link là tạm thời, có thể đổi sau khi khởi động lại. Dùng `--preview --share` để chia sẻ bản xem trước mà không gọi AI. Khi không tạo được tunnel, ứng dụng báo lỗi chia sẻ và vẫn chạy local. Nhấn Ctrl+C để dừng server và đóng tunnel. Lần đầu Gradio có thể tải chương trình tunnel từ nguồn chính thức. `--share` không nạp lại dữ liệu hay thay đổi collection Qdrant.

Dùng **Cuộc trò chuyện mới** để mở ngay một phiên trống hoặc chọn một phiên trong **Các cuộc trò chuyện** để mở lại. Mỗi mục đại diện cho toàn bộ hội thoại, được đặt tên theo câu hỏi đầu tiên; các câu hỏi tiếp theo ở cùng phiên. Lịch sử lưu trên cùng trình duyệt. Đặt `VACXIS_HISTORY_SECRET` ổn định trong `.env` và dùng cùng địa chỉ, cổng để giữ lịch sử qua các lần khởi động lại Gradio.

Dùng `--port` để đổi cổng, `--project-root` để chọn thư mục dự án và `--chunks` để chọn tệp Parquet dùng cho trích dẫn. Đường dẫn chunks tương đối được tính từ project root. Trang tài liệu hiển thị các tài liệu Markdown gốc, kể cả tài liệu chưa được lập chỉ mục; nguồn trong câu trả lời mở đúng đoạn đã lập chỉ mục. Các cờ retrieval có thể ghi đè cấu hình; riêng mặc định kiến trúc và collection của web được đặt như trên, không đổi mặc định CLI benchmark. Terminal in retriever, collection, mode LightRAG và fusion khi khởi động để kiểm tra. Máy chủ web lắng nghe trên loopback để chạy demo cục bộ.

### API Python cho luồng hỏi đáp

```python
from vacxis_capstone_fa26.generation import VacxisRAGPipeline
from vacxis_capstone_fa26.query import (
    Clarification, ClarifyPlan, ConversationMemoryConfig, ConversationMessage, QueryPlanner,
)

planner = QueryPlanner(memory_config=ConversationMemoryConfig(max_messages=12, max_characters=12_000))
history = [
    ConversationMessage("user", "Bé 14 tháng, tôi đang hỏi về Verorab."),
    ConversationMessage("assistant", "Bạn muốn biết thông tin gì về Verorab?"),
]
with VacxisRAGPipeline(query_planner=planner) as pipeline:
    result = pipeline.run_turn("Bé tiêm loại này được không?", history=history, use_memory=True)
    if isinstance(result, ClarifyPlan):
        # Retain this history snapshot and original query for the clarification reply.
        result = pipeline.run_turn(
            "Bé tiêm loại này được không?",
            history=history,
            clarifications=[Clarification(result.clarification_question, "Verorab")],
        )
    if not isinstance(result, ClarifyPlan):
        print(result.query_context)
```

`run_turn()` bật memory mặc định nhưng không lưu trạng thái hội thoại trong pipeline hoặc planner. Caller truyền `history` theo thứ tự thời gian và chỉ chứa các tin nhắn trước câu hỏi hiện tại. `ConversationMessage` bất biến, chỉ nhận role `user`/`assistant` và nội dung không rỗng. Dùng `use_memory=False` để bỏ lịch sử; các câu trả lời trong `clarifications` vẫn được sử dụng.

Giới hạn mặc định là 12 tin nhắn và 12.000 ký tự nội dung; cả hai phải là số nguyên dương. Planner chọn một đoạn cuối liên tiếp, giữ nguyên từng tin nhắn, dừng khi tin nhắn tiếp theo vượt giới hạn và loại tin assistant đầu đoạn thiếu lượt user tương ứng. Không cắt nhỏ tin nhắn hoặc bỏ qua tin lớn để lấy ngữ cảnh cũ hơn. Khi thiếu tham chiếu, planner cần hỏi làm rõ.

Một lần gọi planner giải quyết tham chiếu, làm rõ và mở rộng truy vấn. `RetrievePlan.resolved_query` là câu hỏi độc lập dùng cho cả retrieval và generation; `search_queries` gồm câu này và tối đa hai mở rộng không trùng. Constructor hai đối số cũ vẫn dùng được, mặc định `resolved_query=original_query`. Kết quả `GeneratedAnswer.query` giữ câu hỏi gốc. `query_context` chỉ xuất hiện trong `to_dict()` khi có dữ liệu, chứa câu hỏi đã giải quyết, truy vấn thực tế, trạng thái memory, số tin đã cấp/đã dùng, việc giảm lịch sử, giới hạn, model và thời gian planner; không chứa transcript hội thoại. `answer()`, `answer_batch()` và CLI một lượt vẫn giữ hành vi cũ.

Khi trả `ClarifyPlan`, pipeline không gọi retrieval/generation. Lượt tiếp theo truyền lại câu hỏi gốc, cùng snapshot lịch sử trước câu hỏi và các `Clarification(question, user_answer)` tích lũy. Web UI tự thực hiện việc này, bật memory theo cuộc trò chuyện đang chọn và lưu bằng browser storage hiện có. Hội thoại mới có ngữ cảnh rỗng. Bản ghi pending cũ thiếu snapshot dùng lịch sử rỗng và giữ nguyên câu hỏi/câu trả lời làm rõ. Lỗi tra cứu không ghi lượt gửi thất bại hay câu trả lời dở vào lịch sử; có thể gửi lại từ ô nhập.

Memory chỉ hỗ trợ hiểu câu hỏi; các câu trả lời trước không phải bằng chứng. Mỗi lượt trả lời đều lấy tài liệu mới. Phiên bản này chưa có hồ sơ dài hạn, tóm tắt, vector memory hoặc runner benchmark nhiều lượt.

Khi thực nghiệm:

- So sánh kiến trúc một lượt: tắt memory xuyên suốt bằng `use_memory=False` khi dùng `run_turn()`; CLI hiện tại không dùng planner/memory.
- So sánh kiến trúc nhiều lượt: dùng cùng cấu hình memory cho mọi kiến trúc.
- Đo riêng tác động memory: cố định hybrid baseline, so sánh memory tắt/bật.
- Giữ model planner, chính sách mở rộng, corpus và cấu hình retrieval cố định trong từng phép so sánh.

## Kiểm thử

```bash
# Mặc định: toàn bộ test offline, gồm SDK HTTP giả lập và Qdrant in-memory
uv run pytest

# Lightweight mocked memory regression checks; no retrieval/model stack required
uv run pytest tests/query/test_planner.py tests/query/test_rag_turn.py tests/query/test_memory.py
uv run --extra web pytest tests/test_web_memory.py

# Chạy riêng test dịch vụ thật; cần cấu hình/model/Qdrant tương ứng
uv run --env-file .env pytest -m integration
```

Test offline không cần key thật, tải model hoặc dịch vụ ngoài. Các test dịch vụ thật là opt-in và có thể phát sinh chi phí OpenRouter. Evaluation yêu cầu ID duy nhất, bộ ID dataset/results trùng khớp; kết quả retrieval rỗng vẫn được tính điểm.

Test memory dùng mock để kiểm tra hợp đồng và luồng gọi; chưa chứng minh chất lượng giải quyết ngữ cảnh của planner thật. Chạy các test phụ thuộc retrieval/model trên máy có đủ môi trường:

```bash
uv run --extra retrieval --extra web --env-file .env pytest
uv run --extra retrieval --extra web --env-file .env pytest -m integration
```

## Testdata Structured Outputs

Reference answer của `generate-test-dataset` dùng Qwen qua vLLM với JSON Schema strict; `--dry-run-answers` trả về answer trích xuất offline. Schema kiểm soát cấu trúc; citation và claim không được đối chiếu với corpus khi sinh, chỉ `--relabel` mới tái kiểm tra từ câu hỏi cuối. Nội dung chưa được xác minh vẫn giữ VERIFIED=false.

Trang `EVIDENCE_SPANS` (và metric `evidence_span_coverage`) đã bị xóa khỏi dataset và evaluation. Cờ `--refine-spans` đã bị xóa; parser từ chối nó như cờ lạ.
