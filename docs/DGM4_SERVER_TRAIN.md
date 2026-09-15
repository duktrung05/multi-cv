# Train REAL / AI_EDITED trên Linux / RTX A6000

SSH dùng để truy cập terminal server. Trong terminal, chọn chạy Python trong venv
hoặc Docker. Hướng dẫn này chuẩn bị cả hai cách; chưa thực thi trên server công ty.

Trạng thái: đã viết code dataset, frozen DINOv3 + linear head hai lớp, cache đặc
trưng, checkpoint và đánh giá. Chưa có pretrained trong dự án; chưa train thật.
API/Gradio đã dùng chung predictor DGM4 với CLI. Cần checkpoint DGM4 đã train để khởi động ứng dụng.

Kiểm chứng cục bộ: 6 unit test DGM4 thành công (annotation, mapping, backbone giả
lập được đóng băng, học head, lưu/nạp checkpoint, metric và ngưỡng). 9 file Python
mới/sửa đã qua kiểm tra cú pháp; Compose training qua `config --quiet`. Đây chưa
phải smoke test pretrained thật hoặc benchmark GPU A6000.

Audit cục bộ: 10.000 ảnh đọc được, pixel hash/kích thước khớp, không có ảnh trùng
pixel hoàn toàn. Tuy nhiên có 5 nhóm cùng dHash qua các split. Xem ảnh cho thấy
nhóm 1, 2, 4, 5 có cùng ảnh nền/ảnh nguồn; nhóm 3 chỉ có bố cục giống nhau.
Cần sửa việc chia tập/loại và lấy bù mẫu trước khi train toàn bộ. Luồng train đầy
đủ kiểm tra `ready_for_training` của audit; hiện chưa vượt qua điều kiện này.
Chạy thử nhỏ `limit_per_class=16` vẫn được để kiểm tra kỹ thuật.

## 1. Chuyển dự án

Chuyển mã nguồn và `data/dgm4_binary` sang cùng thư mục dự án trên server bằng
SFTP/SCP, rsync hoặc công cụ được công ty dùng. Không chuyển `.venv` Windows.
Dữ liệu và trọng số bị `.gitignore` loại trừ: clone Git không tự mang chúng sang.
Giữ nguyên manifest và các split đã tạo.

```text
project/
  train.py
  configs/dgm4_binary.yml
  engine/  src/  tools/
  data/dgm4_binary/
    train/real/  train/ai_edited/
    val/real/    val/ai_edited/
    test/real/   test/ai_edited/
    manifest.csv
    report.json
  ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
  outputs/
```

Cần pretrained chính thức DINOv3 ViT-S/16 LVD1689M. Xem
[DINOv3 của Meta](https://github.com/facebookresearch/dinov3) để lấy trọng số.
Code kiểm tra file tồn tại và load_state_dict strict; không âm thầm train bằng
backbone ngẫu nhiên khi thiếu pretrained. Có thể dùng file sẵn trên server bằng
override `pretrained=/absolute/path/to/weights.pth`.

## 2. Cách A: SSH + venv

Chạy từ thư mục dự án. Nếu công ty dùng Slurm hoặc scheduler khác, chạy các lệnh
train trong phiên đã được cấp GPU, không chạy trên login node.

```bash
nvidia-smi
free -h
df -h .
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Cặp phiên bản theo [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/).
Driver NVIDIA trên host phải hỗ trợ bản CUDA này. Không cần tự cài CUDA toolkit
chỉ để dùng wheel PyTorch, nhưng Docker cần NVIDIA Container Toolkit trên host.

Nếu server có nhiều GPU, chọn GPU được cấp bằng `CUDA_VISIBLE_DEVICES` trước khi
chạy Python. Trong tiến trình, GPU đó được gọi là `cuda:0`. Giữ giá trị do scheduler
đặt nếu đang chạy trên cụm máy.

```bash
# Ví dụ chỉ dùng khi GPU vật lý 0 đã được cấp cho bạn:
export CUDA_VISIBLE_DEVICES=0
```

### Kiểm tra dữ liệu và code

```bash
python tools/dataset/validate_dgm4.py
python -m unittest discover -s tests -p 'test_dgm4*.py' -v
```

Đọc `outputs/dgm4_binary/audit/audit.json` và xem `samples.png`. Công cụ kiểm tra
ảnh đọc được, pixel hash, kích thước, nhãn, nguồn và split. Nhóm trùng dHash giữa
split chỉ là ứng viên gần trùng, cần xem lại bằng mắt; đây chưa phải phép kiểm tra
đầy đủ mọi ảnh gần trùng hoặc cùng cảnh.

Không sửa tay `ready_for_training` để bỏ qua các nhóm đã xác nhận. Sau khi làm
sạch hoặc tạo lại dataset, giữ cùng ảnh nguồn trong một split, lấy bù để giữ
8.000/1.000/1.000 rồi audit lại. Lưu bản manifest trước/sau để truy vết.

### Chạy thử nhỏ bằng pretrained

16 ảnh mỗi lớp ở train và validation, hai epoch, không sử dụng test:

```bash
python train.py -c configs/dgm4_binary.yml -u device=cuda:0 limit_per_class=16 epochs=2 output_dir=outputs/dgm4_binary/smoke
```

Nạp checkpoint và thử inference bằng một ảnh validation:

```bash
python infer_dgm4.py --image /path/to/validation-image.jpg --weights outputs/dgm4_binary/smoke/best.pth --device cuda:0
```

Checkpoint từ lần chạy nhỏ chỉ để kiểm tra kỹ thuật, không phải model công bố chất lượng.

### Train chính thức

Sau khi chạy nhỏ thành công:

```bash
python train.py -c configs/dgm4_binary.yml -u device=cuda:0
```

Mặc định: 8.000 train, 1.000 validation; test không tham gia chọn checkpoint.
Ảnh resize 384, extraction batch=4, head batch=128; tối đa 30 epoch và early
stopping sau 5 epoch không cải thiện macro-F1. Chỉ head (1.538 tham số) được học.
DINOv3 luôn ở eval, không gradient. CLS token và trung bình patch token được cache
một lần, sau đó các epoch chỉ dùng vector đặc trưng. Chưa fine-tune backbone.

Trên RTX A6000, thử `extract_batch_size=16` trong một lần chạy nhỏ ở output_dir mới,
đo VRAM và thời gian trước khi dùng giá trị đó cho toàn bộ dữ liệu. Chưa benchmark
server, vì vậy không cam kết batch tối đa hoặc thời gian train. Head batch là batch
vector đặc trưng, không tương đương batch ảnh.

Cache được định danh theo pretrained, preprocessing và ảnh được chọn; cache của
tập nhỏ không dùng thay cho tập đầy đủ. Một run không ghi đè output_dir đã có dữ liệu.

Có thể dùng tmux nếu đã cài trên server để phiên train tiếp tục khi mất kết nối SSH:

```bash
tmux new -s dgm4
# Trong tmux: activate venv rồi chạy train; Ctrl+B, D để detach.
# Khi kết nối lại:
tmux attach -t dgm4
```

Tiếp tục từ checkpoint sau khi phiên training bị dừng:

```bash
python train.py -c configs/dgm4_binary.yml -u device=cuda:0 resume=outputs/dgm4_binary/baseline/last.pth
```

Giữ cùng output_dir, dữ liệu, preprocessing và pretrained. `last.pth` lưu optimizer,
scheduler, epoch và trạng thái xáo trộn batch. `best.pth` giữ macro-F1 validation cao
nhất. Thí nghiệm mới cần output_dir mới.

### Đánh giá test

Sau khi chốt cấu hình và ngưỡng:

```bash
python train.py -c configs/dgm4_binary.yml -u device=cuda:0 test_only=true eval_split=test resume=outputs/dgm4_binary/baseline/best.pth
```

Kết quả ở `outputs/dgm4_binary/baseline/evaluation/`: `test_metrics.json`,
`test_predictions.csv`, `test_errors.csv`. Có accuracy, macro-F1, precision/recall
từng lớp, ROC-AUC, tỷ lệ báo nhầm REAL thành AI_EDITED và bỏ sót AI_EDITED.
Không dùng test để chỉnh ngưỡng. Threshold mặc định 0.5; softmax chưa được hiệu chuẩn.
Kết quả chỉ đánh giá phạm vi DGM4 SimSwap/StyleCLIP, chưa chứng minh tổng quát cho
mọi công cụ AI. Chưa có khoanh vùng chỉnh sửa.

## 3. Cách B: Docker GPU

Host cần Docker Compose và NVIDIA Container Toolkit đã được cấu hình.
`compose.yaml`/`compose.gpu.yaml` chạy API sản phẩm; dùng riêng `compose.train.yaml`.

```bash
mkdir -p outputs ckpts
export TRAIN_UID=$(id -u)
export TRAIN_GID=$(id -g)
export TRAIN_GPU_ID=0  # GPU host được cấp cho bạn
docker compose -f compose.train.yaml build trainer
docker compose -f compose.train.yaml run --rm trainer python tools/dataset/validate_dgm4.py
docker compose -f compose.train.yaml run --rm trainer python -m unittest discover -s tests -p 'test_dgm4*.py' -v
docker compose -f compose.train.yaml run --rm trainer python train.py -c configs/dgm4_binary.yml -u device=cuda:0 limit_per_class=16 epochs=2 output_dir=outputs/dgm4_binary/smoke
docker compose -f compose.train.yaml run --rm trainer
```

`TRAIN_UID/TRAIN_GID` giúp output thuộc user hiện tại. Có thể đặt `DGM4_DATASET_DIR`
và `DGM4_CKPT_DIR` thành đường dẫn tuyệt đối nếu dữ liệu ở ngoài project. Dataset
và checkpoint mount read-only; output lưu ở `./outputs`. GPU đã chọn trên host
xuất hiện dưới tên `cuda:0` trong container. Kiểm tra có đủ quyền đọc dữ liệu và
quyền ghi outputs trước khi chạy.
