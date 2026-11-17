# VacxiS Review

Trang riêng để đọc câu hỏi, đối chiếu đoạn nguồn, sửa câu trả lời và xác nhận. Lời mời, phân công và lịch sử duyệt được lưu trên máy chủ. Lượt thử nghiệm nội bộ có tiến độ và tệp xuất riêng.

## Chạy trên máy

Từ thư mục gốc của repo, dùng PowerShell và `uv`.

```powershell
uv run scripts/prepare_review_dataset.py evaluation/eval_dataset_300_final.json chunks_storage/chunks.parquet review-web/eval_dataset_300_native.json
Set-Location review-web
uv sync
Copy-Item .env.example .env
uv run python -c "import secrets; print(secrets.token_urlsafe(48)); print(secrets.token_urlsafe(32))"
```

Điền lần lượt hai giá trị vừa tạo vào `REVIEW_SECRET_KEY` và `REVIEW_ADMIN_PASSWORD` trong `.env`. Đặt `REVIEW_PRIVACY_EMAIL` nếu muốn công khai địa chỉ liên hệ của nhóm trên trang quyền riêng tư. Giữ `DATABASE_URL` trống để dùng SQLite cục bộ.
Nếu SQLite hiện tại đang chứa bộ 460 câu, đặt `REVIEW_DB_PATH=review-300.sqlite3` trong `.env` để nhập bộ mới vào tệp riêng.

```powershell
uv run --env-file .env store.py eval_dataset_300_native.json
uv run --env-file .env uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở [trang chủ](http://127.0.0.1:8000), bấm **Đăng nhập quản trị** và nhập mật khẩu trong `.env`. Chọn **Thử nghiệm nội bộ** để tạo lời mời thử. Mở lời mời trong hồ sơ trình duyệt khác hoặc cửa sổ riêng tư để giữ phiên quản trị đang dùng. Bí mật và SQLite cục bộ nằm trong các đường dẫn đã được Git bỏ qua.

## Deploy lên Vercel

1. Đưa thư mục `review-web` lên repo Git mà Vercel có quyền đọc. Trong Vercel, chọn **Add New → Project**, nhập repo và đặt **Root Directory** thành `review-web`. Dùng preset FastAPI và để Vercel tự cài thư viện từ `pyproject.toml` và `uv.lock`. [FastAPI trên Vercel](https://vercel.com/docs/frameworks/backend/fastapi) · [Chọn thư mục dự án](https://vercel.com/docs/monorepos#add-a-monorepo-through-the-vercel-dashboard)
2. Trong **Storage** hoặc **Marketplace**, kết nối Neon Postgres với project. Kiểm tra môi trường **Production** có `DATABASE_URL` trỏ đến cơ sở dữ liệu này. Nếu integration đặt tên biến khác, thêm `DATABASE_URL` với chuỗi kết nối có TLS của Neon. Dùng URL kết nối dạng pooled nếu Neon cung cấp. [Neon trên Vercel](https://vercel.com/marketplace/neon) · [Kết nối Postgres](https://vercel.com/docs/postgres)
3. Trong **Settings → Environment Variables**, đặt `REVIEW_SECRET_KEY` và `REVIEW_ADMIN_PASSWORD` bằng hai giá trị ngẫu nhiên riêng cho Production. Có thể thêm `REVIEW_PRIVACY_EMAIL` để hiện địa chỉ liên hệ. Bật **Enable access to System Environment Variables** để ứng dụng lấy tên miền `vercel.app` của project. Dùng `REVIEW_PUBLIC_URL=https://<tên-miền>` nếu muốn đặt tên miền phát lời mời cụ thể. Nếu bật Preview, dùng cơ sở dữ liệu và bí mật riêng cho Preview. [Biến môi trường](https://vercel.com/docs/environment-variables) · [Tên miền tự động](https://vercel.com/docs/environment-variables/system-environment-variables#vercelprojectproductionurl)
4. Chuẩn bị tệp `eval_dataset_300_native.json` bằng lệnh ở trên. Từ `review-web`, tạo tệp `.env.neon` với `DATABASE_URL` của Production rồi chạy lệnh nạp sau cho database trống. Nếu Neon đang chứa bộ 460 câu, dùng một database trống cho bộ 300 câu rồi trỏ Vercel tới database đó.

   ```powershell
   uv run --env-file .env.neon store.py eval_dataset_300_native.json
   ```

5. Deploy hoặc redeploy sau khi các biến đã được lưu. Trên tên miền Vercel cấp, mở `/admin/login`, đăng nhập, rồi tạo một lời mời **Thử nghiệm nội bộ** để kiểm tra. Sau đó mở một lời mời **Duyệt thật** để kiểm tra màn hình xin phép ngay khi bắt đầu.

Nếu Production đã nhập bộ 300 câu theo định dạng cũ, deploy mã mới trước. Trang quản trị có thể xuất JSON theo cấu trúc DeepEval ngay từ dữ liệu đó. Để khôi phục đầy đủ `metadata` của tệp gốc, dùng `.env.neon` trỏ đến chính database 300 câu hiện tại và chạy lệnh sau từ `review-web`. Lệnh kiểm tra các câu hỏi và đoạn nguồn trước khi cập nhật nguồn, đồng thời giữ lời mời, phân công và các lượt duyệt.

```powershell
uv run --env-file .env.neon store.py --upgrade-eval eval_dataset_300_native.json
```

## Vận hành đợt duyệt

- Tạo một lời mời cho mỗi người và chọn số câu cần giao. Bấm **Sao chép lời mời** để lấy tin nhắn có sẵn cả liên kết và mã 12 ký tự. Gửi riêng cho từng người.
- Người duyệt có thể mở liên kết hoặc nhập mã ngay ở trang chủ. Mã chấp nhận chữ thường và có thể bỏ dấu gạch. Hai cách dùng chung một lời mời, có hạn 7 ngày và dùng một lần.
- Để tiếp tục, người duyệt mở lại trang bằng cùng trình duyệt. Nhóm có thể cấp lời mời mới nếu người duyệt đổi trình duyệt hoặc phiên hết hạn. Giữ `REVIEW_SECRET_KEY` ổn định qua các lần deploy để mã đã cấp tiếp tục hoạt động trong thời hạn của chúng.
- Với lời mời **Duyệt thật** lần đầu, trang xin phép chia sẻ thông tin mở ngay. Người duyệt có thể điền các trường tùy chọn hoặc bấm **Bỏ qua và bắt đầu duyệt**. Cả hai cách đều mở câu hỏi đầu tiên. Người duyệt có thể xem, sửa hoặc xóa các trường đã chia sẻ tại **Bảo vệ thông tin cá nhân**. Lời mời **Thử nghiệm nội bộ** đi thẳng tới bước bắt đầu duyệt.
- Nhóm quản lý có thể xem lịch sử, mở lại câu đã xác nhận hoặc cần xem lại, và thu hồi các câu chưa bắt đầu để giao cho người khác. Khi thu hồi câu cuối cùng, tài khoản mất quyền truy cập và hồ sơ tùy chọn được xóa.
- Nút tải JSON của lượt thật xuất các câu đã xác nhận. Bộ 300 câu dùng cấu trúc DeepEval gồm `metadata` và `goldens`. Mỗi golden lưu câu trả lời đã duyệt trong `expected_output` và giữ `context` cùng `source_chunk_ids` khớp theo thứ tự đoạn nguồn được chọn. Trong `additional_metadata`, `VERIFICATION_NOTES` gồm ghi chú và câu trả lời sau chỉnh sửa nếu có. `ADJUDICATED` là `true` khi người duyệt đã sửa câu trả lời hoặc để lại ghi chú. `ANNOTATOR_IDS` ghi họ tên và vai trò khi người duyệt đồng ý đưa tên vào JSON, hoặc ghi mã người duyệt.
- Trang nguồn sử dụng Markdown, các bảng HTML đã được lọc và văn bản nguyên gốc. Nút mở PDF theo trang cần tệp PDF và bảng đối chiếu đoạn với trang PDF.

## Kiểm thử

```powershell
uv run python -m unittest discover -s tests -v
```
