# DGM4 — REAL / AI_EDITED (đang chuyển đổi)

`train.py` hiện mặc định dùng `configs/dgm4_binary.yml`: DINOv3 cố định, linear head
hai lớp, Cross-Entropy và softmax. Xem [hướng dẫn train Linux / RTX A6000](docs/DGM4_SERVER_TRAIN.md)
và `compose.train.yaml`. Đã qua 6 unit test DGM4 với dữ liệu/model giả lập; chưa
chạy thử bằng pretrained, chưa train thật và chưa chuyển API/Gradio sang DGM4.

Audit đã đọc đủ 10.000 ảnh, không lỗi hash/đọc ảnh; phát hiện 5 nhóm ứng viên gần
trùng qua các split, trong đó 4 nhóm được xác nhận cùng ảnh nguồn qua xem ảnh.
Cần xử lý chia tập/lấy bù mẫu và bổ sung pretrained trước khi train toàn bộ.

Phần dưới là tài liệu luồng sản phẩm cũ. Các lệnh train/test của luồng này phải
truyền rõ `-c configs/product_inspection.yml`.

# detect_bolt — Kiểm tra lỗi ngoại quan sản phẩm (luồng cũ)

Dự án dùng DINOv3 + phân loại đa nhãn để kiểm tra **một loại sản phẩm** từ ảnh hoặc các khung hình video. Một ảnh có thể có nhiều lỗi.

**Trạng thái:** đã chuyển mã, cấu hình, API và giao diện sang kiểm tra sản phẩm. Chưa có dữ liệu sản phẩm thật hoặc checkpoint đã huấn luyện cho bài toán này. Không dùng checkpoint kiểm duyệt cũ để dự đoán lỗi.

## Phạm vi và nhãn

| Nhãn | Ý nghĩa |
|---|---|
| DENT | Móp, biến dạng |
| SCRATCH | Trầy xước |
| CRACK | Nứt, vỡ |
| DIRT | Vết bẩn trên bề mặt |
| MISSING_PART | Thiếu thành phần, ví dụ thiếu nắp |
| MISSING_LABEL | Thiếu nhãn |
| MISALIGNED_LABEL | Nhãn lệch vị trí |

Đây là bộ nhãn khởi đầu. Chọn loại sản phẩm và chỉ giữ các lỗi áp dụng được, rồi cập nhật đồng bộ `class_list` và `num_classes` trong `configs/product_inspection.yml`.
Mô hình phân loại **toàn ảnh**, chưa trả khung bao hay mặt nạ lỗi. Chụp sản phẩm đủ lớn, ở góc và ánh sáng ổn định; lỗi nhỏ có thể mất khi resize về 384×384.

Quy tắc ban đầu theo điểm lỗi cao nhất:
- **PASS:** mọi điểm lỗi < 0.30.
- **REVIEW:** ít nhất một điểm >= 0.30 và tất cả < 0.70.
- **FAIL:** ít nhất một điểm >= 0.70.

Đây là ngưỡng khởi đầu, cần hiệu chỉnh trên tập validation thật. Điểm sigmoid chưa được hiệu chuẩn; `defect_score` không phải xác suất quyết định PASS/FAIL đúng. Ảnh ngoài loại sản phẩm đã học có thể bị phân loại sai; chưa có bộ phát hiện ngoài miền.

## Cài đặt

Chạy lệnh từ thư mục chứa README này. Dùng Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Với GPU, cài bản PyTorch tương thích GPU/CUDA của máy. Video cần lệnh `ffmpeg` trong PATH.
Các biến môi trường mẫu nằm trong `.env.example`; ứng dụng tự tải `.env` trong thư mục dự án; biến môi trường hệ điều hành được ưu tiên.

## Chuẩn bị dữ liệu

```text
data/product_inspection/
  train/
    good_001.jpg
    defect_001.jpg
  valid/
    ...
  test/
    ...
  train.csv
  valid.csv
  test.csv
```

Mỗi CSV có cột `image,labels`. Đường dẫn ảnh tương đối với thư mục split tương ứng.
Ví dụ CSV thật:

```csv
image,labels
good_001.jpg,[]
defect_001.jpg,"[""DENT"",""SCRATCH""]"
```

- `[]` chỉ dùng cho ảnh đã được kiểm tra và xác nhận không có lỗi; bỏ trống ô là lỗi dữ liệu.
- Chỉ các ảnh được liệt kê trong CSV mới tham gia huấn luyện. Không suy ra nhãn từ tên thư mục.
- Chia train/valid/test theo sản phẩm vật lý, lô hàng hoặc buổi chụp; các góc ảnh cùng một sản phẩm phải ở cùng split.
- Có ảnh đạt và ảnh lỗi trong các split, bao phủ từng loại lỗi. Không đưa ảnh test vào huấn luyện hoặc chọn ngưỡng.

Kiểm tra ảnh đọc được, nhãn hợp lệ, nhãn thiếu và ảnh trùng nội dung:

```powershell
python tools/dataset/validate_product_dataset.py
```

Nếu đổi bộ nhãn, truyền `--labels DENT SCRATCH ...` cho công cụ kiểm tra.
Công cụ không tự phát hiện ảnh gần giống hoặc cùng sản phẩm được chụp khác góc.

## Huấn luyện và đánh giá

Đặt trọng số backbone DINOv3 ViT-S/16 pretrained tại:
`ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`.
Mã dùng lại kiến trúc và backbone; phần phân loại lỗi được học mới. Trình huấn luyện báo lỗi khi thiếu pretrained, thay vì âm thầm chạy với backbone ngẫu nhiên.

```powershell
python train.py -c configs/product_inspection.yml
```

Cấu hình mặc định dùng batch 8, 40 epoch, BCE + sigmoid, AdamW. Không crop/flip ngẫu nhiên để tránh mất lỗi nhỏ hoặc làm sai nhãn lệch vị trí. Cần thử augmentation phù hợp sau khi xem dữ liệu thực tế.

Checkpoint `best.pth` và `last.pth` nằm trong thư mục chạy dưới `outputs/product_inspection/summary/` (hoặc output_dir khi không có writer). File lưu cả thứ tự nhãn và miền bài toán; API từ chối checkpoint không khớp.

Đánh giá checkpoint trên validation:

```powershell
python train.py -u test_only=true resume=outputs/product_inspection/summary/<run>/best.pth
```

Để đánh giá test, thêm các override:
`val_dataloader.dataset.root=data/product_inspection/test val_dataloader.dataset.annotations_path=data/product_inspection/test.csv`.

Báo cáo có macro-F1, precision/recall/F1 từng lỗi và độ khớp bộ nhãn ở ngưỡng 0.5. Các chỉ số này chưa thay cho kiểm định chính sách PASS/REVIEW/FAIL: hãy đo riêng bỏ sót lỗi, báo nhầm và tỷ lệ REVIEW với ngưỡng triển khai.

## Dự đoán và giao diện

Sau khi huấn luyện, copy checkpoint đã chọn thành `ckpts/product_inspection.pth`, hoặc truyền `--weights`.

```powershell
python infer_product.py --image path/to/product.jpg
python gradio_app.py
```

Giao diện mặc định: http://127.0.0.1:7877, hiển thị quyết định và điểm từng lỗi.
CLI và giao diện dùng chung loader, transform validation và quy tắc quyết định với API.

## API ảnh/video

```powershell
$env:INSPECTION_CKPT = "ckpts/product_inspection.pth"
$env:MODEL_DEVICE = "auto"
python main.py
```

Swagger: http://127.0.0.1:8126/docs.

- `POST /inspect/image`: multipart, field `file`.
- `POST /inspect/video`: JSON `{"url":"https://example.com/product.mp4","video_id":"sample","frame_step":24}`.
- `/test/classify/image` và `/test/classify/video` là alias; nội dung phản hồi đã đổi sang kiểm tra sản phẩm.
- Các endpoint job bất đồng bộ được giữ lại; kết quả job và callback có trường `inspection` chứa đầy đủ nhãn/điểm/quyết định.

Ví dụ phản hồi minh họa, không phải kết quả của model đã huấn luyện:

```json
{"ai_result":{"decision":"FAIL","defect_score":0.91,"defects":["DENT"],"suspected_defects":["SCRATCH"],"scores":{"DENT":0.91,"SCRATCH":0.4}}}
```

Phản hồi thực tế trả điểm của tất cả nhãn. Các trường cũ `coarse_label/coarse_score` trong job lần lượt mang quyết định và **điểm lỗi cao nhất**, không phải độ tin cậy của quyết định.

Video lấy mẫu mỗi `frame_step` khung hình, rồi lấy điểm cao nhất **cho từng lỗi** trên các frame đã lấy mẫu. Có thể bỏ sót lỗi giữa các frame. API đồng bộ trả `worst_frame_index` trong danh sách frame đã lấy mẫu; không trả đường dẫn tạm đã xóa.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Kiểm thử chính sách ngưỡng, dữ liệu đa nhãn, ảnh đạt, chống nhãn lạ, tổng hợp video, API ảnh và một bước học của model bằng dữ liệu giả lập. Kiểm thử kỹ thuật không chứng minh độ chính xác kiểm lỗi ngoài thực tế.

Đã xác minh trên CPU: 11 kiểm thử thành công; một epoch DINOv3 thật trên ảnh giả lập, lưu/nạp checkpoint và suy luận 64px/384px thành công; kiểm tra phụ thuộc không có xung đột. Môi trường `.venv` được chuẩn bị tại đây dùng PyTorch CPU. Chưa kiểm thử GPU, video thật qua FFmpeg hoặc độ chính xác trên dữ liệu sản phẩm thật.

Chạy lại kiểm thử tích hợp (dữ liệu/checkpoint giả lập nằm trong thư mục tạm, tự xóa):

```powershell
python tools/benchmark/smoke_product.py
```

## Mã nguồn

- `configs/product_inspection.yml`: cấu hình mặc định.
- `engine/data/dataset/product_inspection.py`: dataset CSV.
- `src/domain/inspection.py`: quy tắc và tổng hợp video.
- `src/infrastructure/ml/model_factory.py`: nạp checkpoint, suy luận.
- `infer_product.py`, `gradio_app.py`, `main.py`: CLI, UI, API.

Cấu hình và công cụ ReaS/NSFW còn lại là tài nguyên cũ, không thuộc luồng sản phẩm mặc định. README trước chuyển đổi được giữ trong `docs/LEGACY_REAS_README.md`. Các công cụ Grad-CAM/detection cũ chưa được chuyển đổi và không nằm trong MVP này.



## Docker

Đặt checkpoint đã huấn luyện tại `ckpts/product_inspection.pth` trước khi chạy.
Docker image không chứa dữ liệu, trọng số hoặc `.env`; Compose gắn checkpoint/config vào container khi chạy.

```powershell
# Chỉ cần copy nếu chưa có .env; giữ các thiết lập đang dùng nếu file đã tồn tại.
Copy-Item .env.example .env
# CPU
docker compose up --build -d
docker compose logs -f api
# Dừng dịch vụ, giữ dữ liệu output
docker compose down
```

Swagger mặc định: http://127.0.0.1:8126/docs. Compose cố định cổng trong container ở 8126;
`APP_HOST` và `APP_PORT` trong `.env` điều khiển địa chỉ/cổng công bố trên máy chủ.
Cấu hình, checkpoint và đường dẫn output trong container được Compose thiết lập riêng;
để dùng checkpoint khác, sửa đường dẫn `INSPECTION_CKPT` trong Compose tương ứng với mount `/app/ckpts`.
Output lưu trong named volume `inspection_outputs`. Không dùng `down -v` nếu cần giữ output.

GPU NVIDIA (Docker host cần hỗ trợ NVIDIA Container Toolkit/GPU passthrough):

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

Bản GPU dùng PyTorch CUDA 12.4, cần driver NVIDIA tương thích. Bản CPU là mặc định.
Container chạy bằng user không phải root và có healthcheck `/health`.
Thiếu checkpoint hoặc checkpoint sai bộ nhãn sẽ khiến ứng dụng dừng với thông báo lỗi.

## Git và tên dự án

Tên package/ứng dụng: `detect_bolt` (công cụ Python có thể chuẩn hóa thành `detect-bolt`).
Tên thư mục làm việc hiện tại được giữ nguyên để không làm hỏng đường dẫn môi trường `.venv`.
`.gitignore` bỏ qua dữ liệu, checkpoint, output, cache và `.env`; `.env.example` được đưa vào Git.
`.gitattributes` chuẩn hóa xuống dòng cho Python/Docker/Linux.

```powershell
git status
git add .
git commit -m "Describe your changes"
git push origin main
```

Repository: [duktrung05/multi-cv](https://github.com/duktrung05/multi-cv), nhánh `main`.
Dataset và trọng số cần chuyển riêng lên server vì không được lưu trong Git.
