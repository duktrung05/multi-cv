# Chuẩn bị dataset DGM4

Script `prepare_dgm4.py` đọc trực tiếp 4 ZIP `bbc.zip`, `usa_today.zip`,
`simswap.zip`, `StyleCLIP.zip` và `train.json`, `val.json`, `test.json`
trong cùng thư mục nguồn. Không cần giải nén trước. `load_Data.zip` không cần dùng.

Chạy trong PowerShell tại thư mục dự án (thư mục có `.venv`):

```powershell
.\.venv\Scripts\python.exe tools\dataset\prepare_dgm4.py --source "C:\Users\Admin\Downloads" --output "data\dgm4_binary"
```

Cần Python và Pillow (`python -m pip install Pillow` nếu dùng môi trường khác).
Thư mục đầu ra phải mới hoặc rỗng. Script không ghi đè bộ dữ liệu đã có.
Để chạy lại, chọn tên đầu ra khác bằng `--output`.

Mặc định mỗi lớp có 4.000 ảnh train, 500 val, 500 test (10.000 ảnh tổng).
Trong mỗi tập, AI_EDITED gồm một nửa SimSwap và một nửa StyleCLIP.
Thử bộ nhỏ bằng `--counts 20 10 10`. Seed mặc định 42; đổi bằng `--seed`.

Đầu ra gồm các thư mục `{train,val,test}/{real,ai_edited}`, `manifest.csv`
và `report.json`. Chỉ dùng bộ dữ liệu khi report có `complete: true`
và không còn `INCOMPLETE.txt`. Nếu chạy lỗi hoặc bị dừng, đầu ra chưa hoàn tất;
xem lỗi và chạy lại vào thư mục mới.

Script giữ các split chính thức, loại source ID/đường dẫn xuất hiện ở nhiều
split, kiểm tra CRC khi đọc ZIP, giải mã ảnh và loại trùng pixel RGB chính xác.
Ảnh chỉ sửa văn bản không được chọn vào AI_EDITED. Ảnh xuất giữ nguyên byte nguồn.
Không phát hiện được mọi ảnh gần trùng sau crop, resize hoặc nén lại;
vẫn nên kiểm tra trực quan trước khi huấn luyện. Đây là dataset chỉnh sửa
khuôn mặt, không đại diện mọi công cụ AI hay mọi kiểu chỉnh sửa ảnh.

Manifest giữ nhãn, nguồn, ID, phương pháp, bbox gốc và hash pixel để truy vết.
Dataset này chưa tự thay đổi model/cấu hình kiểm tra lỗi sản phẩm của dự án.
