# VacxiS Review

Trang riêng để đọc câu hỏi, đối chiếu đoạn nguồn điểm 3, sửa câu trả lời và xác nhận. Lời mời, phân công và lịch sử duyệt được lưu trên máy chủ. Lượt thử nghiệm nội bộ có tiến độ và tệp xuất riêng.

## Chạy trên máy

Từ thư mục gốc của repo, dùng PowerShell và `uv`.

```powershell
Set-Location review-web
uv sync
Copy-Item .env.example .env
uv run python -c "import secrets; print(secrets.token_urlsafe(48)); print(secrets.token_urlsafe(32))"
```

Điền lần lượt hai giá trị vừa tạo vào `REVIEW_SECRET_KEY` và `REVIEW_ADMIN_PASSWORD` trong `.env`. Đặt `REVIEW_PRIVACY_EMAIL` nếu muốn công khai địa chỉ liên hệ của nhóm trên trang quyền riêng tư. Giữ `DATABASE_URL` trống để dùng SQLite cục bộ.

```powershell
uv run --env-file .env store.py ../test_dataset.json
uv run --env-file .env uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở [trang chủ](http://127.0.0.1:8000), bấm **Đăng nhập quản trị** và nhập mật khẩu trong `.env`. Chọn **Thử nghiệm nội bộ** để tạo lời mời thử. Mở lời mời trong hồ sơ trình duyệt khác hoặc cửa sổ riêng tư để giữ phiên quản trị đang dùng. Bí mật và SQLite cục bộ nằm trong các đường dẫn đã được Git bỏ qua.

## Deploy lên Vercel

1. Đưa thư mục `review-web` lên repo Git mà Vercel có quyền đọc. Trong Vercel, chọn **Add New → Project**, nhập repo và đặt **Root Directory** thành `review-web`. Dùng preset FastAPI và để Vercel tự cài thư viện từ `pyproject.toml` và `uv.lock`. [FastAPI trên Vercel](https://vercel.com/docs/frameworks/backend/fastapi) · [Chọn thư mục dự án](https://vercel.com/docs/monorepos#add-a-monorepo-through-the-vercel-dashboard)
2. Trong **Storage** hoặc **Marketplace**, kết nối Neon Postgres với project. Kiểm tra môi trường **Production** có `DATABASE_URL` trỏ đến cơ sở dữ liệu này. Nếu integration đặt tên biến khác, thêm `DATABASE_URL` với chuỗi kết nối có TLS của Neon. Dùng URL kết nối dạng pooled nếu Neon cung cấp. [Neon trên Vercel](https://vercel.com/marketplace/neon) · [Kết nối Postgres](https://vercel.com/docs/postgres)
3. Trong **Settings → Environment Variables**, đặt `REVIEW_SECRET_KEY` và `REVIEW_ADMIN_PASSWORD` bằng hai giá trị ngẫu nhiên riêng cho Production. Có thể thêm `REVIEW_PRIVACY_EMAIL` để hiện địa chỉ liên hệ. Bật **Enable access to System Environment Variables** để ứng dụng lấy tên miền `vercel.app` của project. Dùng `REVIEW_PUBLIC_URL=https://<tên-miền>` nếu muốn đặt tên miền phát lời mời cụ thể. Nếu bật Preview, dùng cơ sở dữ liệu và bí mật riêng cho Preview. [Biến môi trường](https://vercel.com/docs/environment-variables) · [Tên miền tự động](https://vercel.com/docs/environment-variables/system-environment-variables#vercelprojectproductionurl)
4. Nạp dữ liệu vào Neon từ máy bạn. Tạo tệp `.env.neon` trong `review-web` với `DATABASE_URL` của Production. Chạy lệnh dưới đây. Lần chạy lại với tệp đầu vào giống hệt giữ nguyên các lượt duyệt hiện có. Với một phiên bản bộ câu hỏi khác, dùng cơ sở dữ liệu mới.

   ```powershell
   uv run --env-file .env.neon store.py ../test_dataset.json
   ```

5. Deploy hoặc redeploy sau khi các biến đã được lưu. Trên tên miền Vercel cấp, mở `/admin/login`, đăng nhập, rồi tạo một lời mời **Thử nghiệm nội bộ** để kiểm tra. Sau đó tạo lời mời **Duyệt thật** để kiểm tra màn hình xin phép trước lần xác nhận đầu tiên.

## Vận hành đợt duyệt

- Tạo một lời mời cho mỗi người và chọn số câu cần giao. Bấm **Sao chép lời mời** để lấy tin nhắn có sẵn cả liên kết và mã 12 ký tự. Gửi riêng cho từng người.
- Người duyệt có thể mở liên kết và bấm **Vào trang duyệt**, hoặc nhập mã ngay ở trang chủ. Mã chấp nhận chữ thường và có thể bỏ dấu gạch. Hai cách dùng chung một lời mời, có hạn 7 ngày và dùng một lần.
- Để tiếp tục, người duyệt mở lại trang bằng cùng trình duyệt. Nhóm có thể cấp lời mời mới nếu người duyệt đổi trình duyệt hoặc phiên hết hạn. Giữ `REVIEW_SECRET_KEY` ổn định qua các lần deploy để mã đã cấp tiếp tục hoạt động trong thời hạn của chúng.
- Người duyệt thật có thể bỏ qua biểu mẫu thông tin cá nhân và vẫn xác nhận câu trả lời. Họ có thể xem, sửa hoặc xóa các trường đã chia sẻ tại **Bảo vệ thông tin cá nhân**. Liên kết dành cho nhóm thử nghiệm luôn bỏ qua biểu mẫu.
- Nhóm quản lý có thể xem lịch sử, mở lại câu đã xác nhận hoặc cần xem lại, và thu hồi các câu chưa bắt đầu để giao cho người khác. Khi thu hồi câu cuối cùng, tài khoản mất quyền truy cập và hồ sơ tùy chọn được xóa.
- Nút tải JSON của lượt thật xuất các câu đã xác nhận. Khi câu trả lời thay đổi, web dùng đoạn nguồn được chọn để tạo trích dẫn và bỏ bản đồ mệnh đề cũ. Tệp xuất ghi mã người duyệt. Họ tên, nơi công tác, vai trò và ghi chú do người duyệt nhập được lưu riêng trong trang duyệt.
- Trang nguồn sử dụng Markdown, các bảng HTML đã được lọc và văn bản nguyên gốc. Nút mở PDF theo trang cần tệp PDF và bảng đối chiếu đoạn với trang PDF.

## Kiểm thử

```powershell
uv run python -m unittest discover -s tests -v
```
