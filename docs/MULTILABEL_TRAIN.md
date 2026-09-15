# Huấn luyện kiểm tra lỗi sản phẩm

Hướng dẫn hiện hành nằm trong [README](../README.md).
Cấu hình: `configs/product_inspection.yml`.
Dataset: `ProductInspectionDataset`, mỗi ảnh có danh sách lỗi trong CSV.
Ảnh đạt chất lượng có `labels=[]`; không suy ra nhãn từ tên thư mục.

Dùng `python train.py` để huấn luyện; `python infer_product.py --image ...` để dự đoán.
Các tài liệu/cấu hình ReaS cũ chỉ là tài nguyên tham khảo, không phải luồng mặc định.
