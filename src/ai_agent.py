"""Tác nhân phân tích kết quả bằng Gemini, kèm chế độ dự phòng không cần API."""
from __future__ import annotations

import os
from typing import Dict

from .models import fallback_analysis


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def call_gemini(prompt: str, api_key: str | None = None, model_name: str = "gemini-3.5-flash") -> str:
    """Gọi Gemini nếu có API key.

    Ưu tiên SDK mới `google-genai`. Nếu máy vẫn đang cài SDK cũ `google-generativeai`,
    hàm sẽ thử tiếp để tương thích ngược. Nếu thiếu key, thiếu thư viện, lỗi mạng,
    lỗi quota hoặc model không khả dụng, hàm trả về chuỗi rỗng để app dùng phân tích nội bộ.
    """
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        return ""

    # Các model mới đặt trước; gemini-2.5-flash để tương thích nếu tài khoản vẫn hỗ trợ.
    candidates = _unique([
        model_name,
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ])

    # SDK mới do Google khuyến nghị: pip install google-genai
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=key)
        for m in candidates:
            try:
                resp = client.models.generate_content(model=m, contents=prompt)
                txt = getattr(resp, "text", "") or ""
                if txt.strip():
                    return txt.strip()
            except Exception:
                continue
    except Exception:
        pass

    # Tương thích ngược với SDK cũ: google-generativeai
    try:
        import google.generativeai as old_genai  # type: ignore
        old_genai.configure(api_key=key)
        for m in candidates:
            try:
                model = old_genai.GenerativeModel(m)
                resp = model.generate_content(prompt)
                txt = getattr(resp, "text", "") or ""
                if txt.strip():
                    return txt.strip()
            except Exception:
                continue
    except Exception:
        pass

    return ""


def analyze(title: str, context: Dict, api_key: str | None = None, use_gemini: bool = True) -> str:
    """Trả về phân tích bằng Gemini nếu khả dụng; nếu không sẽ dùng tác nhân nội bộ."""
    fallback = fallback_analysis(title, context)
    if not use_gemini:
        return fallback
    prompt = f"""
Bạn là tác nhân AI hỗ trợ phân tích kết quả cho học phần Các mô hình ra quyết định.
Hãy viết bằng tiếng Việt, giọng học thuật nhưng dễ hiểu. Không bịa số liệu ngoài dữ liệu đã cho.
Tên bài: {title}
Các kết quả chính dạng dictionary: {context}
Yêu cầu trả lời 4 đoạn rõ ràng, không mở đầu kiểu "Chào bạn".
1) Diễn giải các kết quả định lượng quan trọng; phải nêu lại ít nhất 4-6 con số cụ thể có trong context nếu có, dùng đúng đơn vị ở trường unit.
2) Phân tích ý nghĩa chính sách trong bối cảnh Việt Nam, đối chiếu trực tiếp các số như GDP_gain, MAPE, VSS, EVPI, TOPSIS, Priority, NetJob hoặc reward nếu có. Không viết chung chung.
3) Chỉ ra điểm đánh đổi/ràng buộc đang chi phối kết quả; giải thích vì sao kết quả lại nghiêng về hạng mục/vùng/ngành/kịch bản đang đứng đầu.
4) Nêu hạn chế mô hình và gợi ý kiểm tra độ nhạy trước khi kết luận.
Tuyệt đối không dùng cụm "có thể là VND" hoặc đoán đơn vị. Nếu context thiếu một số liệu, nói "không có trong bảng kết quả hiện tại" thay vì bịa.
""".strip()
    txt = call_gemini(prompt, api_key=api_key)
    if txt:
        return txt
    return fallback
