Dự án xây dựng dashboard Streamlit gồm 12 menu tương ứng 12 bài tập theo yêu cầu: Cobb-Douglas, LP, MIP, TOPSIS, Pareto, tối ưu động, lao động, stochastic programming, Q-learning và đồ án tích hợp AIDEOM-VN.

Web app mô phỏng và phân tích 12 bài toán thuộc học phần Các mô hình ra quyết định, với chủ đề phát triển kinh tế Việt Nam trong kỷ nguyên AI.

## Tính năng chính

- Giao diện Streamlit gồm 12 menu tương ứng với 12 bài toán.
- Mỗi bài có bảng kết quả, biểu đồ trực quan và phần phân tích kết quả.
- Cho phép điều chỉnh tham số mô hình trực tiếp trên giao diện.
- Tích hợp Gemini API để hỗ trợ phân tích kết quả.
- Nếu không có API key Gemini, hệ thống vẫn có phần phân tích nội bộ.

## Dữ liệu sử dụng

Dữ liệu được đặt trong thư mục `data/`, gồm:

- `vietnam_macro_2020_2025.csv`
- `vietnam_sectors_2024.csv`
- `vietnam_regions_2024.csv`

## Cách chạy
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
