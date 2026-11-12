# Quy trình test GPU trên Kaggle

Bộ chạy này kiểm tra bộ test của repository, thực thi một phép toán CUDA thật,
rồi huấn luyện, nạp lại và chấm điểm một mô hình neural rất nhỏ trên GPU được cấp.
Bộ test bao gồm luồng batch production end-to-end trên một batch giả lập: retrain,
gate và promote, inference chia shard với nhiều process worker, và xuất bản ba
file. Bộ `real` (tùy chọn) còn chạy một lượt inference dry-run có giới hạn, không
xuất bản, trên một bộ dữ liệu MyTV private đã được phê duyệt. Các test này không
phải qualification năng lực 1M/2M profile (việc đó do `batch_cli qualify-hardware`
thực hiện trên máy chủ production), không phải chỉ số online, và không phải bằng
chứng sẵn sàng cho production.

## Chuẩn bị đầu vào private

Tạo một Kaggle dataset **private** cho source bundle và gắn dataset đó làm input
của một notebook GPU. Source bundle cần chứa code của repository và
`requirements.txt`; bootstrap loại trừ `data/` và `.git/` khi chép code
sang vùng lưu trữ ghi được. Không đặt output thô ở mức profile vào dataset public.
Chỉ cấp quyền cộng tác dataset cho người đã được phê duyệt, và ghi lại phiên bản
bất biến của dataset MyTV dùng cho lượt chạy thật.

Với bộ `real`, gắn một trong hai: snapshot đã chuẩn bị chỉ gồm canonical, có
`canonical/source_manifest.json`; hoặc bố cục nguồn thô đầy đủ kèm thư mục
canonical đã chuẩn bị đó. Trong bố cục thô, data root được mount chứa
`behavior/`, `duration/`, `mytv_vmp_content/`, `tags/`, `content_tags/`,
`month_mytv_info.parquet` và `canonical/`; Kaggle mount các thư mục này ở chế độ
chỉ đọc trong `/kaggle/input`. Bộ chạy từ chối nguồn thô không đầy đủ, yêu cầu
source manifest canonical và phiên bản dataset, rồi tạo symlink từ nguồn vào data
root ghi được của riêng nó. Bộ chạy ghi dữ liệu sinh ra, checkpoint, output
dry-run và `kaggle_test_report.json` trong `/kaggle/working`.

## Giao diện notebook: lượt chạy synthetic sạch

Chọn một GPU accelerator của Kaggle được bản PyTorch đã cài hỗ trợ, thêm source
dataset private vào mục Inputs của notebook, và dùng một phiên **Save & Run All**
mới. Lượt chạy sạch không được phụ thuộc vào các cell chạy tay theo thứ tự hay
trạng thái của notebook. Trong cell code đầu tiên, thay `recsys-source` bằng slug
thực tế của source đã mount:

```bash
!bash /kaggle/input/recsys-source/kaggle/bootstrap.sh \
  /kaggle/input/recsys-source /kaggle/working/recsys-run synthetic
```

Preflight phải báo một phép toán tensor thành công trên CUDA. Chỉ thấy GPU hoặc
`torch.cuda.is_available()` trả về true là chưa đủ; bộ chạy fail-closed và không
chuyển sang huấn luyện trên CPU. Nếu phép toán kernel thất bại, hãy chọn
accelerator/runtime Kaggle được hỗ trợ và chạy lại từ một phiên sạch. Bootstrap
dùng chung `requirements.txt` nhưng bỏ qua PyTorch và wheel index để giữ bản
Torch sẵn có của Kaggle.

## Dry-run trên dữ liệu private

Trước hết, lượt chạy `synthetic` phải đạt. Sau đó gắn dataset MyTV private đã được
phê duyệt, có đánh phiên bản, và cung cấp chính xác đường dẫn dữ liệu đã mount cùng
phiên bản đó. Ví dụ, nếu các file đã chuẩn bị nằm trong
`/kaggle/input/mytv-private/data/data`:

```bash
%env RECSYS_KAGGLE_DATASET_VERSION=mytv-private-v7
!bash /kaggle/input/recsys-source/kaggle/bootstrap.sh \
  /kaggle/input/recsys-source /kaggle/working/recsys-real real \
  /kaggle/input/mytv-private/data/data
```

Dùng phiên bản bất biến của Kaggle dataset, không dùng tên hiển thị có thể thay
đổi, làm giá trị biến môi trường. Bộ real yêu cầu nguồn gốc canonical đã chuẩn bị
và chạy `all/infer.py` với giới hạn 1,000 profile và `--dry-run`; bộ này không xuất
bản output cho khách hàng. Hãy xem lại fingerprint/hash của source manifest, phiên
bản nguồn, exit code của các lệnh, thông tin GPU, phiên bản phần mềm và artifact
dry-run trong báo cáo trước khi diễn giải kết quả.

## Phương án dùng Kaggle CLI

Với lượt chạy khởi động qua Kaggle CLI, tạo một thư mục kernel nhỏ trên máy, thêm
một script Python gọi cùng bootstrap, khởi tạo metadata, gắn source dataset và
dataset MyTV private, bật GPU, rồi push kernel. Kaggle mô tả `kaggle kernels init`
và `kaggle kernels push` trong [hướng dẫn CLI cho
kernel](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md). Giữ kernel
ở chế độ private; không đưa API credential hay nội dung dataset private vào source
của kernel.

Ví dụ, lưu nội dung sau thành `run_recsys_test.py` trong thư mục kernel đó, và chỉ
thay ba giá trị được đánh dấu rõ trong bản sao trên máy:

```python
from pathlib import Path
import os
import subprocess

SOURCE_SLUG = "replace-with-private-source-dataset-slug"
MYTV_SLUG = "replace-with-private-mytv-dataset-slug"
MYTV_VERSION = "replace-with-immutable-dataset-version"
SUITE = "synthetic"  # đổi thành "real" sau khi synthetic đạt

source = Path("/kaggle/input") / SOURCE_SLUG
args = [
    "bash", str(source / "kaggle/bootstrap.sh"), str(source),
    "/kaggle/working/recsys-run", SUITE,
]
if SUITE == "real":
    os.environ["RECSYS_KAGGLE_DATASET_VERSION"] = MYTV_VERSION
    args.append(str(Path("/kaggle/input") / MYTV_SLUG / "data/data"))
subprocess.run(args, check=True)
```

Sau đó khởi tạo và push kernel từ thư mục đó:

```bash
kaggle kernels init -p ./recsys-kernel
# Đặt run_recsys_test.py vào ./recsys-kernel và sửa kernel-metadata.json:
# đặt code_file, enable_gpu=true, is_private=true, và dataset_sources bằng slug
# của source dataset private và dataset MyTV. Ghi lại chính xác phiên bản dataset
# MyTV đã gắn vào ghi chú lượt chạy và vào MYTV_VERSION ở trên.
kaggle kernels push -p ./recsys-kernel
```

Các trường metadata chính xác được mô tả trong [hướng dẫn metadata
kernel](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
của Kaggle. Với `SUITE = "real"`, wrapper truyền data root `data/data` đã mount và
phiên bản dataset MyTV bất biến một cách tường minh cho bộ chạy fail-closed.

Trước một lượt chạy dài, hãy kiểm tra giới hạn thời gian phiên/accelerator và dung
lượng output làm việc hiện tại của tài khoản và runtime Kaggle; các giới hạn này có
thể thay đổi. CUDA smoke nhỏ và dry-run 1,000 profile là các kiểm tra có giới hạn,
không phải gate SLA 2 chu kỳ/ngày. Mọi khẳng định năng lực 1M/2M vẫn cần
qualification trọn chu kỳ, với bằng chứng về phần cứng chính xác, số worker,
RAM/VRAM đỉnh, I/O, thời gian chạy và tính nhất quán của output.

## Lấy và lưu giữ bằng chứng

Bộ chạy in một bản tóm tắt JSON và ghi atomic file
`/kaggle/working/<run-directory>/results/kaggle_test_report.json`. Tải output làm
việc từ panel Output của notebook trước khi phiên Kaggle hết hạn. Lưu giữ URL
notebook/kernel, commit nguồn hoặc fingerprint của bundle, runtime/accelerator
Kaggle, phép toán GPU thực tế và compute capability của thiết bị, bản build
PyTorch, phiên bản dataset MyTV private và fingerprint của source manifest, SHA-256
của code bundle trong báo cáo, exit code của các lệnh, JSON báo cáo, và hash của
các artifact liên quan. Trường commit có thể là null với dataset bundle chỉ chứa
code, không có `.git`; khi đó dùng SHA-256 của code bundle. Không xuất dòng dữ liệu
thô hay output sinh ra ở mức profile vào artifact public.

Thiếu input, không có GPU, hết quota thời gian, bỏ qua CUDA smoke, lệnh test thất
bại, hoặc thiếu báo cáo đều là **lượt chạy chưa xác minh/thất bại**, không bao giờ
là đạt.
