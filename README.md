# VN AIDEOM-VN - Web app cuối kỳ Các mô hình ra quyết định

Dự án xây dựng dashboard Streamlit gồm 12 menu tương ứng 12 bài tập trong đề: Cobb-Douglas, LP, MIP, TOPSIS, Pareto, tối ưu động, lao động, stochastic programming, Q-learning và đồ án tích hợp AIDEOM-VN.

## Cách chạy nhanh trên Windows

1. Giải nén file `.zip`.
2. Mở thư mục dự án.
3. Nhấp đúp `run_local.bat`.
4. Đợi cài thư viện, trình duyệt sẽ mở web app Streamlit.

## Cách chạy bằng lệnh

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# hoặc: source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Gemini API

- Trong sidebar có mục **Gemini API**.
- Có API key: dán key vào ô, bật `Dùng Gemini nếu có API`, cuối mỗi bài tác nhân AI sẽ phân tích bằng Gemini.
- Không có API key: app vẫn chạy bình thường và dùng **tác nhân AI nội bộ** để phân tích kết quả. Đây là chế độ dự phòng để giảng viên có thể mở bài không cần key.
- API key chỉ lưu trong phiên Streamlit hiện tại. Nếu muốn đặt cố định, tạo file `.streamlit/secrets.toml` hoặc đặt biến môi trường `GEMINI_API_KEY`.

## Cấu trúc thư mục

```text
aideom_vn_final/
├── app.py
├── data/
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_regions_2024.csv
│   └── vietnam_sectors_2024.csv
├── src/
│   ├── models.py
│   ├── ai_agent.py
│   └── style.py
├── reports/
├── outputs/
├── tests/
├── requirements.txt
├── run_local.bat
└── run_local.sh
```

## Ghi chú kỹ thuật

- Các mô hình lõi được đặt trong `src/models.py`, tách khỏi giao diện để dễ kiểm thử.
- Một số thư viện nâng cao trong đề như PuLP, CVXPY, Pyomo, pymoo, gymnasium được để trong requirements dạng tùy chọn. App có các thuật toán fallback bằng NumPy/SciPy để giảm lỗi cài đặt trên máy người chấm.
- Bài 4 phát hiện ràng buộc công bằng gốc với λ=0,70 và trần 12.000 có thể không khả thi; app tự báo và cho phép dùng λ khả thi gần nhất để minh họa nghiệm.


## Ghi chú bản nhẹ
Bản này dùng `requirements.txt` tối giản để cài nhanh hơn và giảm lỗi môi trường. Các mô hình trong app được tính bằng `numpy`, `pandas`, `scipy` và thuật toán nội bộ; không cần cài `cvxpy`, `pyomo`, `pymoo`, `gymnasium` hay các solver nặng. Nếu có API key Gemini, nhập trực tiếp ở sidebar; nếu không có, web vẫn phân tích bằng AI nội bộ.

Trên Windows, chạy:

```bat
run_local.bat
```


## Ghi chú bản v3
- Sửa lỗi menu Bài 10, 11, 12 mở nhầm Bài 1 bằng cách so khớp tên trang chính xác.
- Chuẩn hóa ghi chú đơn vị theo đề và dữ liệu: Bài 2 dùng nghìn tỷ VND; Bài 4/5/9/12 dùng tỷ VND; Bài 10 hiển thị nghìn tỷ VND sau quy đổi từ ràng buộc 65.000/15.000 tỷ VND.
- Cập nhật phân tích AI/Gemini để bám sát số liệu và không đoán đơn vị.
