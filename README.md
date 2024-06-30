# mushroom_classification
# Dataset
- Bao gồm 15 class và mỗi class gồm 350 ảnh

<img width="612" alt="Screenshot 2024-06-30 at 12 13 50" src="https://github.com/ThanhVViet/mushroom_classification/assets/126480817/65f50646-789a-4227-a0a2-ce54ca2adac6">

# Phân chia dữ liệu
Dữ liệu được chia thành 3 tập train, validate, test với tỉ lệ 70:20:10

<img width="650" alt="Screenshot 2024-06-30 at 12 15 39" src="https://github.com/ThanhVViet/mushroom_classification/assets/126480817/611fc2dc-0232-4801-a4c3-35132d5e32ee">

# Huấn luyện mô hình MobileNetV2
Huấn luyện mô hình trên tập dữ liệu huấn luyện (X_train, y_train) với batch size 32 và 50 epochs, đồng thời sử dụng tập validation (X_test, y_test) để đánh giá hiệu suất trong quá trình huấn luyện. Các callbacks checkpoint và early_stopping được sử dụng để lưu mô hình tốt nhất và dừng sớm nếu cần thiết

# Kết quả
<img width="550" alt="Screenshot 2024-06-30 at 12 35 47" src="https://github.com/ThanhVViet/mushroom_classification/assets/126480817/2531f7da-1c9c-45df-aa2f-429f967a14e2">

<img width="577" alt="Screenshot 2024-06-30 at 12 36 40" src="https://github.com/ThanhVViet/mushroom_classification/assets/126480817/0173f781-c1c8-4e4f-9361-0f90be78458d">

<img width="686" alt="Screenshot 2024-06-30 at 12 37 01" src="https://github.com/ThanhVViet/mushroom_classification/assets/126480817/01679f4f-9944-4532-b7e2-8cd28ac2fbdd">
