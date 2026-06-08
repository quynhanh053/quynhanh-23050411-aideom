from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Để chạy trực tiếp bằng streamlit run app.py
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.style import APP_CSS
from src.ai_agent import analyze
from src.models import (
    fmt_num, load_macro, load_sectors, load_regions,
    solve_bai1, solve_bai2, solve_bai3, solve_bai4, solve_bai5, solve_bai6,
    solve_bai7, solve_bai8, solve_bai9, solve_bai10, solve_bai11, solve_bai12,
    B2_ITEMS, PROJECTS, REGION_NAMES, ITEM_NAMES, TOPSIS_LABELS, ACTIONS,
)

st.set_page_config(
    page_title="VN AIDEOM-VN | Mô hình ra quyết định",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)

MENU = [
    "Trang chủ",
    "Bài 1 — Cobb-Douglas + AI",
    "Bài 2 — LP ngân sách số",
    "Bài 3 — Priority 10 ngành",
    "Bài 4 — LP ngành-vùng",
    "Bài 5 — MIP 15 dự án",
    "Bài 6 — TOPSIS 6 vùng",
    "Bài 7 — NSGA-II Pareto",
    "Bài 8 — Động 2026-2035",
    "Bài 9 — Lao động & AI",
    "Bài 10 — Stochastic SP",
    "Bài 11 — Q-learning RL",
    "Bài 12 — AIDEOM tích hợp",
]

# ===================== Helper giao diện =====================

def hero(title: str, subtitle: str, badges: list[str] | None = None):
    badges = badges or []
    badge_html = "".join([f"<span class='badge'>{b}</span>" for b in badges])
    st.markdown(f"""
    <div class="hero">
        <div>{badge_html}</div>
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def ai_box(title: str, context: dict):
    with st.expander("Phân tích kết quả", expanded=True):
        api_key = st.session_state.get("gemini_api_key", "")
        use_gemini = st.session_state.get("use_gemini", True)
        text = analyze(title, context, api_key=api_key, use_gemini=use_gemini)
        st.markdown(text)


def kpi(label: str, value: str, help_text: str = ""):
    st.markdown(f"<div class='kpi'><small>{label}</small><b>{value}</b><br><small>{help_text}</small></div>", unsafe_allow_html=True)


def styled_formula(text: str):
    st.markdown(f"<div class='formula'>{text}</div>", unsafe_allow_html=True)


def unit_note(text: str):
    st.caption(f"Đơn vị sử dụng trong bài: {text}")


def metric_cols(items: list[tuple[str, str, str]]):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def show_df(df: pd.DataFrame, height: int = 360):
    st.dataframe(df, width="stretch", height=height)

# ===================== Sidebar =====================

with st.sidebar:
    st.markdown("### VN AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI")
    page = st.radio("Menu bài làm", MENU, index=0, label_visibility="collapsed", key="main_page_selector")
    st.divider()
    st.markdown("#### Chế độ phân tích")
    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = os.getenv("GEMINI_API_KEY", "")
    ai_mode = st.radio(
        "Lựa chọn AI",
        ["Dùng API key Gemini", "Không có API - dùng AI nội bộ"],
        index=0 if st.session_state.get("gemini_api_key", "") else 1,
        label_visibility="collapsed",
    )
    st.session_state["use_gemini"] = ai_mode == "Dùng API key Gemini"
    if st.session_state["use_gemini"]:
        st.session_state["gemini_api_key"] = st.text_input(
            "API key Gemini", value=st.session_state["gemini_api_key"], type="password"
        )
    else:
        st.session_state["gemini_api_key"] = ""
    st.divider()
    st.caption("Dữ liệu: 3 file CSV Việt Nam 2020-2025, sectors 2024, regions 2024.")

# ===================== Pages =====================

if page == "Trang chủ":
    hero("VN AIDEOM-VN", "Web app giải 12 bài toán mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI.", ["Streamlit", "Tối ưu hóa", "AI Agent", "Dữ liệu Việt Nam"])
    macro, sectors, regions = load_macro(), load_sectors(), load_regions()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("GDP 2025", "514,0 tỷ USD", "↑ 8,02%")
    with c2:
        kpi("Kinh tế số/GDP", "≈19,5%", "↑ 1,2 điểm")
    with c3:
        kpi("FDI giải ngân 2025", "27,6 tỷ USD", "↑ 8,9%")
    with c4:
        kpi("GDP/người 2025", "5,026 USD", "↑ 6,9%")

    st.markdown("## 12 bài toán theo 4 cấp độ")
    st.caption("Cách phân nhóm bám theo đề bài: từ mô hình cơ bản đến tối ưu đa mục tiêu, tối ưu động, quy hoạch ngẫu nhiên và học tăng cường.")

    levels = {
        "Cấp độ DỄ — Làm quen mô hình": [
            ("Bài 1", "Hàm sản xuất Cobb-Douglas mở rộng + AI — growth accounting, dự báo GDP 2030"),
            ("Bài 2", "LP phân bổ ngân sách số 4 hạng mục — scipy.optimize, shadow price"),
            ("Bài 3", "Chỉ số ưu tiên 10 ngành — min-max norm, weighted scoring, sensitivity"),
        ],
        "Cấp độ TRUNG BÌNH — Tối ưu có điều kiện": [
            ("Bài 4", "LP phân bổ ngân sách số ngành-vùng — 24 biến, ràng buộc công bằng vùng"),
            ("Bài 5", "MIP lựa chọn 15 dự án chuyển đổi số — knapsack + ràng buộc tiên quyết"),
            ("Bài 6", "TOPSIS xếp hạng 6 vùng — trọng số chuyên gia + Entropy"),
        ],
        "Cấp độ KHÁ KHÓ — Đa mục tiêu và động": [
            ("Bài 7", "NSGA-II/Pareto — cân bằng tăng trưởng, bao trùm, môi trường và rủi ro dữ liệu"),
            ("Bài 8", "Tối ưu động 2026-2035 — quỹ đạo K, D, AI, H và welfare"),
            ("Bài 9", "Tác động AI tới lao động — NetJob, đào tạo lại và kiểm soát dịch chuyển việc làm"),
        ],
        "Cấp độ KHÓ — Bất định, học tăng cường và tích hợp": [
            ("Bài 10", "Stochastic programming hai giai đoạn — VSS, EVPI, recourse theo kịch bản"),
            ("Bài 11", "Q-learning chính sách kinh tế thích nghi — MDP, reward và policy"),
            ("Bài 12", "Dashboard AIDEOM-VN tích hợp — M1-M6, 5 kịch bản chính sách"),
        ],
    }
    for level, rows in levels.items():
        with st.expander(level, expanded=level.startswith("Cấp độ DỄ")):
            show_df(pd.DataFrame(rows, columns=["Bài", "Nội dung chính"]),
                height=150)

    with st.expander("Phạm vi phân tích của web app", expanded=False):
        st.markdown("""
        Web app AIDEOM-VN được xây dựng để mô phỏng và phân tích 12 bài toán mô hình ra quyết định trong bối cảnh phát triển kinh tế Việt Nam trong kỷ nguyên AI.

        Dữ liệu sử dụng gồm 3 nhóm chính:

        - **Dữ liệu vĩ mô Việt Nam 2020–2025**: GDP, vốn tích lũy, lao động, kinh tế số, năng lực AI và nhân lực số.
        - **Dữ liệu 10 ngành kinh tế năm 2024**: tăng trưởng, năng suất, xuất khẩu, việc làm, AI readiness và rủi ro tự động hóa.
        - **Dữ liệu 6 vùng kinh tế - xã hội năm 2024**: GRDP/người, FDI, Digital Index, AI readiness, lao động qua đào tạo, R&D, Internet và Gini.

        Mỗi bài toán bao gồm mô hình định lượng, bảng kết quả, biểu đồ trực quan và phần phân tích kết quả bằng Gemini API hoặc AI nội bộ.
        """)

elif page.startswith("Bài 1 —"):
    hero("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng", "Ước lượng TFP, so sánh GDP thực tế - dự báo, phân rã tăng trưởng và mô phỏng GDP 2030.", ["Cấp độ dễ", "Growth accounting", "numpy/pandas"])
    styled_formula("<b>Mô hình:</b> Yₜ = Aₜ · Kₜ^α · Lₜ^β · Dₜ^γ · AIₜ^δ · Hₜ^θ, với α + β + γ + δ + θ = 1")
    unit_note("Y, K tính theo nghìn tỷ VND; L là triệu lao động; D là % kinh tế số/GDP; AI là nghìn doanh nghiệp/năng lực số; H là % lao động qua đào tạo.")
    styled_formula("<b>Phân rã tăng trưởng:</b> ΔlnYₜ = ΔlnAₜ + αΔlnKₜ + βΔlnLₜ + γΔlnDₜ + δΔlnAIₜ + θΔlnHₜ")
    with st.expander("Thiết lập tham số Bài 1", expanded=True):
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        alpha = col_a.slider("α (K)", 0.10, 0.50, 0.31, 0.01)
        beta = col_b.slider("β (L)", 0.10, 0.60, 0.42, 0.01)
        gamma = col_c.slider("γ (D)", 0.01, 0.25, 0.10, 0.01)
        delta = col_d.slider("δ (AI)", 0.01, 0.25, 0.08, 0.01)
        theta = round(1.0 - alpha - beta - gamma - delta, 4)
        col_e.metric("θ (H) tự động", f"{theta:.2f}")
        if theta <= 0:
            st.warning("Tổng α+β+γ+δ đang vượt 1. Hãy giảm một hệ số để θ dương.")
            theta = 0.01
        col1, col2, col3 = st.columns(3)
        d2030 = col1.slider("Kinh tế số/GDP 2030 (%)", 20.0, 40.0, 30.0, 0.5)
        ai2030 = col2.slider("AI 2030 (nghìn DN/năng lực)", 80.0, 140.0, 100.0, 1.0)
        h2030 = col3.slider("Nhân lực số 2030 (%)", 28.0, 45.0, 35.0, 0.5)
    res = solve_bai1(alpha, beta, gamma, delta, theta, d2030=d2030, ai2030=ai2030, h2030=h2030)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE", f"{res['mape']:.2f}%")
    c2.metric("TFP TB", f"{res['A_bar']:.2f}")
    c3.metric("GDP 2030", f"{res['Y2030']:.1f} nghìn tỷ")
    c4.metric("Tổng hệ số", f"{sum(res['params']):.2f}")
    tab1, tab2, tab3 = st.tabs(["Bảng kết quả", "Biểu đồ", "Phân rã tăng trưởng"])
    with tab1:
        show_df(res["detail"].round(4), 340)
    with tab2:
        fig = px.line(res["detail"], x="Năm", y=["Y thực tế", "Y dự báo"], markers=True, title="GDP thực tế và GDP dự báo")
        st.plotly_chart(fig, width="stretch")
        fig2 = px.line(res["detail"], x="Năm", y="TFP A_t", markers=True, title="Xu hướng TFP A_t")
        st.plotly_chart(fig2, width="stretch")
    with tab3:
        show_df(res["contrib_summary"].round(4), 300)
        fig3 = px.bar(res["contrib_summary"], x="Yếu tố", y="Tỷ trọng đóng góp (%)", title="Tỷ trọng đóng góp vào tăng trưởng GDP log bình quân")
        st.plotly_chart(fig3, width="stretch")
    top_contrib = res["contrib_summary"].sort_values("Tỷ trọng đóng góp (%)", ascending=False).iloc[0]
    ai_box("Bài 1 Cobb-Douglas", {
        "mape": res["mape"], "Y2030": res["Y2030"], "A_bar": res["A_bar"],
        "top": f"{top_contrib['Yếu tố']} ({top_contrib['Tỷ trọng đóng góp (%)']:.1f}%)",
        "tfp_2020": float(res["detail"].iloc[0]["TFP A_t"]),
        "tfp_2025": float(res["detail"].iloc[-1]["TFP A_t"]),
    })

elif page.startswith("Bài 2 —"):
    hero("Bài 2 — LP phân bổ ngân sách số", "Tối đa hóa GDP kỳ vọng với 4 biến quyết định và các ràng buộc ngân sách, sàn đầu tư, tỷ trọng công nghệ chiến lược.", ["LP", "scipy.optimize.linprog", "shadow price"])
    styled_formula("max Z = 0,85x₁ + 1,20x₂ + 0,95x₃ + 1,35x₄")
    unit_note("x₁..x₄ là nghìn tỷ VND; Z là nghìn tỷ VND tăng GDP kỳ vọng.")
    with st.expander("Thiết lập tham số Bài 2", expanded=True):
        B = st.slider("Ngân sách tổng (nghìn tỷ)", 80.0, 160.0, 100.0, 5.0)
        min_i = st.slider("Sàn hạ tầng số", 0.0, 60.0, 25.0, 1.0)
        min_ai = st.slider("Sàn AI và dữ liệu", 0.0, 60.0, 15.0, 1.0)
        min_h = st.slider("Sàn nhân lực số", 0.0, 60.0, 20.0, 1.0)
        min_rd = st.slider("Sàn R&D", 0.0, 60.0, 10.0, 1.0)
        tech_ratio = st.slider("Tỷ trọng AI + R&D tối thiểu", 0.10, 0.60, 0.35, 0.01)
    res = solve_bai2(B, min_i, min_ai, min_h, min_rd, tech_ratio)
    if not res["success"]:
        st.error(res["message"])
    else:
        metric_cols([("Z*", f"{res['z']:.2f}", "nghìn tỷ VND GDP kỳ vọng"), ("Ngân sách", f"{B:.0f}", "nghìn tỷ VND"), ("Hệ số cao nhất", "R&D 1,35", "tác động biên"), ("Công nghệ chiến lược", f"{tech_ratio:.0%}", "AI+R&D")])
        col1, col2 = st.columns([1.1, .9])
        with col1:
            show_df(res["allocation"].round(3), 260)
        with col2:
            fig = px.pie(res["allocation"], values="Phân bổ tối ưu", names="Hạng mục", hole=.45, title="Cơ cấu phân bổ tối ưu")
            st.plotly_chart(fig, width="stretch")
        tab1, tab2 = st.tabs(["Shadow price", "Độ nhạy ngân sách"])
        with tab1:
            show_df(res["duals"].round(4), 260)
        with tab2:
            show_df(res["sensitivity"].round(3), 180)
            st.plotly_chart(px.line(res["sensitivity"], x="Ngân sách", y="Z*", markers=True, title="Đường cong Z*(B)"), width="stretch")
            hh = res["high_h"]
            if hh.get("success"):
                st.info(f"Khi tăng sàn nhân lực số lên x₃ ≥ 30, bài toán vẫn khả thi; Z* = {hh['z']:.2f}.")
        ai_box("Bài 2 LP ngân sách số", {"z": res["z"], "unit": "nghìn tỷ VND GDP kỳ vọng", "budget": B, "tech_ratio": tech_ratio, "top": res["allocation"].sort_values("Phân bổ tối ưu", ascending=False).iloc[0]["Hạng mục"], "allocation": res["allocation"].round(2).to_dict("records")})

elif page.startswith("Bài 3 —"):
    hero("Bài 3 — Chỉ số ưu tiên ngành Priorityᵢ", "Chuẩn hóa min-max 7 tiêu chí, tính điểm Priority và kiểm tra độ nhạy trọng số AI readiness.", ["MCDM", "Min-max", "Sensitivity"])
    styled_formula("Priorityᵢ = a₁Growthᵢ + a₂Productivityᵢ + a₃Spilloverᵢ + a₄Exportᵢ + a₅Employmentᵢ + a₆AIReadinessᵢ − a₇Riskᵢ. Trong app, Risk đã đảo chiều thành điểm an toàn nên được cộng vào tổng điểm.")
    unit_note("Tăng trưởng %, năng suất triệu VND/lao động, xuất khẩu tỷ USD, việc làm triệu lao động, AI readiness 0-100, rủi ro tự động hóa %; Priority là điểm chuẩn hóa không có đơn vị.")
    res = solve_bai3()
    top3 = ", ".join(res["result"].head(3)["sector_name_vi"].tolist())
    metric_cols([("Top 1", res["result"].iloc[0]["sector_name_vi"], "điểm cao nhất"), ("Số ngành", "10", "CSV sectors"), ("Tiêu chí", "7", "đã chuẩn hóa"), ("Top 3", "ổn định/nhạy", "xem heatmap")])
    tab1, tab2, tab3 = st.tabs(["Xếp hạng", "Ma trận chuẩn hóa", "Độ nhạy"])
    with tab1:
        show_df(res["result"].round(4), 420)
        st.plotly_chart(px.bar(res["result"], x="Priority", y="sector_name_vi", orientation="h", title="Xếp hạng Priority", text="Priority"), width="stretch")
    with tab2:
        show_df(res["normalized"].round(3), 360)
    with tab3:
        pivot = res["sensitivity"].pivot(index="Ngành", columns="w_AI", values="Xếp hạng")
        fig = px.imshow(pivot, text_auto=True, aspect="auto", title="Heatmap xếp hạng khi thay đổi trọng số AI readiness")
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown("**Định hướng tăng trưởng**")
        c1.dataframe(res["growth_rank"].round(4), width="stretch")
        c2.markdown("**Định hướng bao trùm**")
        c2.dataframe(res["inclusive_rank"].round(4), width="stretch")
    ai_box("Bài 3 Priority ngành", {"top": top3})

elif page.startswith("Bài 4 —"):
    hero("Bài 4 — LP phân bổ ngân sách số theo vùng", "Phân bổ 50.000 tỷ VND cho 6 vùng và 4 hạng mục, có sàn/trần vùng và ràng buộc công bằng số.", ["LP 24 biến", "Fairness", "Regional policy"])
    styled_formula("max Z = ΣᵣΣⱼ βⱼᵣxⱼᵣ; C5: Dᵣ + γxᴰᵣ ≥ λ·maxᵣ(Dᵣ + γxᴰᵣ)")
    unit_note("xⱼᵣ là tỷ VND; Z là tỷ VND GDP gain kỳ vọng; Dᵣ là chỉ số số hóa 0-100.")
    with st.expander("Thiết lập tham số Bài 4", expanded=True):
        B = st.slider("Ngân sách tổng", 30000.0, 70000.0, 50000.0, 1000.0)
        lam = st.slider("λ công bằng", 0.50, 0.75, 0.60, 0.01)
        fairness = st.checkbox("Bật ràng buộc công bằng C5", True)
    res = solve_bai4(total_budget=B, lam=lam, fairness=fairness, auto_relax=True)
    nofair = solve_bai4(total_budget=B, fairness=False)
    if not res["success"]:
        st.error(res["message"])
    else:
        if res["relaxed"]:
            st.warning(f"Với λ={lam:.2f}, mô hình gốc không khả thi do trần vùng và D ban đầu thấp ở Tây Nguyên. App tự dùng λ gần nhất khả thi: {res['lambda_used']:.3f} để vẫn có nghiệm minh họa.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z*", f"{res['z']:.1f} tỷ VND")
        c2.metric("λ dùng", f"{res['lambda_used']:.3f}")
        c3.metric("Chi phí công bằng", f"{(nofair['z']-res['z']):.1f} tỷ VND" if nofair.get("success") else "NA")
        c4.metric("Hạng mục lớn nhất", res["item_total"].sort_values("Tổng theo hạng mục", ascending=False).iloc[0]["Hạng mục"])
        tab1, tab2, tab3 = st.tabs(["Ma trận phân bổ", "Heatmap", "So sánh công bằng"])
        with tab1:
            show_df(res["allocation"].round(2), 300)
        with tab2:
            fig = px.imshow(res["allocation"], text_auto=".0f", aspect="auto", title="Heatmap phân bổ ngân sách tối ưu")
            st.plotly_chart(fig, width="stretch")
        with tab3:
            show_df(res["digital_after"].round(3), 260)
            if nofair.get("success"):
                comp = pd.DataFrame({"Mô hình": ["Có công bằng", "Không công bằng"], "Z*": [res["z"], nofair["z"]]})
                st.plotly_chart(px.bar(comp, x="Mô hình", y="Z*", title="Chi phí kinh tế của công bằng vùng miền"), width="stretch")
        ai_box("Bài 4 LP ngành-vùng", {"z": res["z"], "unit": "tỷ VND GDP gain kỳ vọng", "lambda": res["lambda_used"], "fairness_cost": (nofair["z"]-res["z"]) if nofair.get("success") else None, "top": res["item_total"].sort_values("Tổng theo hạng mục", ascending=False).iloc[0]["Hạng mục"], "warning": "Ràng buộc công bằng cần kiểm tra khả thi với λ, γ và trần ngân sách vùng."})

elif page.startswith("Bài 5 —"):
    hero("Bài 5 — MIP lựa chọn 15 dự án chuyển đổi số", "Chọn tập dự án tối ưu với biến nhị phân, ràng buộc ngân sách đa năm, loại trừ và tiên quyết.", ["MIP", "Knapsack", "Binary decision"])
    styled_formula("max Σᵢ Bᵢyᵢ; ΣCᵢyᵢ ≤ 80.000; y₁+y₂≤1; y₈≤y₁₂; y₁₃≤y₁₂; y₁₄=1; 7≤Σyᵢ≤11")
    unit_note("Chi phí, lợi ích NPV, ngân sách năm 1-2 và năm 3-5 đều là tỷ VND; yᵢ là biến nhị phân chọn/không chọn dự án.")
    with st.expander("Thiết lập tham số Bài 5", expanded=True):
        budget = st.slider("Ngân sách tổng 5 năm", 60000.0, 110000.0, 80000.0, 1000.0)
        b12 = st.slider("Ngân sách năm 1-2", 30000.0, 60000.0, 40000.0, 1000.0)
        force = st.checkbox("Bắt buộc chọn cả P1 và P2", False)
        risk = st.checkbox("Tối đa hóa lợi ích kỳ vọng theo xác suất đúng tiến độ", False)
    res = solve_bai5(budget, b12, force, risk)
    if not res["success"]:
        st.error(res["message"])
    else:
        metric_cols([("Lợi ích mục tiêu", f"{res['z']:.0f}", "tỷ VND"), ("Tổng chi phí", f"{res['total_cost']:.0f}", "tỷ VND"), ("Số dự án", str(res["n_projects"]), "được chọn"), ("NPV/chi phí", f"{res['benefit_cost']:.2f}", "hiệu quả biên")])
        tab1, tab2 = st.tabs(["Dự án được chọn", "Danh mục đầy đủ"])
        with tab1:
            show_df(res["selected"], 420)
            fig = px.bar(res["selected"], x="id", y=["Chi phí", "Lợi ích NPV"], barmode="group", title="Chi phí và lợi ích NPV dự án được chọn")
            st.plotly_chart(fig, width="stretch")
        with tab2:
            show_df(PROJECTS, 480)
        alt100 = solve_bai5(100000, b12, False, risk)
        if alt100.get("success"):
            st.info(f"Khi nới ngân sách lên 100.000 tỷ, lợi ích mục tiêu tăng lên {alt100['z']:.0f} với {alt100['n_projects']} dự án được chọn.")
        ai_box("Bài 5 MIP dự án", {"z": res["z"], "unit": "tỷ VND lợi ích NPV", "total_cost": res["total_cost"], "benefit_cost": res["benefit_cost"], "n_projects": res["n_projects"], "top": ", ".join("P" + res["selected"]["id"].astype(str).head(4))})

elif page.startswith("Bài 6 —"):
    hero("Bài 6 — TOPSIS xếp hạng 6 vùng ưu tiên đầu tư AI", "Tính hệ số gần gũi C* theo trọng số chuyên gia và trọng số Entropy khách quan.", ["TOPSIS", "Entropy", "MCDM"])
    styled_formula("Cᵢ* = Sᵢ⁻ / (Sᵢ* + Sᵢ⁻). Cᵢ* càng cao, vùng càng gần lời giải lý tưởng dương.")
    unit_note("GRDP/người triệu VND, FDI tỷ USD, Digital/AI index 0-100, lao động đào tạo %, R&D/GRDP %, Internet %, Gini là tiêu chí chi phí; TOPSIS C* không có đơn vị.")
    res = solve_bai6()
    top3 = ", ".join(res["result"].head(3)["region_name_vi"].tolist())
    metric_cols([("Top 1", res["result"].iloc[0]["region_name_vi"], "TOPSIS chuyên gia"), ("Top 3", top3, "vùng ưu tiên"), ("Tiêu chí", "8", "7 lợi ích + Gini chi phí"), ("Entropy", "tự động", "trọng số khách quan")])
    tab1, tab2, tab3 = st.tabs(["Xếp hạng TOPSIS", "Trọng số Entropy", "Độ nhạy w_AI"])
    with tab1:
        show_df(res["result"].round(4), 360)
        st.plotly_chart(px.bar(res["result"], x="TOPSIS chuyên gia", y="region_name_vi", orientation="h", title="Xếp hạng vùng theo TOPSIS"), width="stretch")
    with tab2:
        show_df(res["entropy_weights"].round(4), 320)
    with tab3:
        pivot = res["sensitivity"].pivot(index="Vùng", columns="w_AI", values="Xếp hạng")
        st.plotly_chart(px.imshow(pivot, text_auto=True, aspect="auto", title="Độ nhạy xếp hạng theo trọng số AI readiness"), width="stretch")
    ai_box("Bài 6 TOPSIS vùng", {"top": top3})

elif page.startswith("Bài 7 —"):
    hero("Bài 7 — Tối ưu đa mục tiêu Pareto", "Tìm tập phương án Pareto giữa tăng trưởng, bao trùm, môi trường và an ninh dữ liệu; chọn nghiệm thỏa hiệp bằng TOPSIS.", ["Pareto", "NSGA-II mô phỏng", "TOPSIS compromise"])
    styled_formula("max f₁ tăng trưởng; min f₂ bất bình đẳng; min f₃ phát thải; min f₄ rủi ro dữ liệu ròng")
    unit_note("Phân bổ ngân sách tính bằng tỷ VND; tăng trưởng/GDP gain, phát thải và rủi ro là chỉ số mô phỏng theo hệ số trong mô hình; điểm TOPSIS thỏa hiệp không có đơn vị.")
    with st.expander("Thiết lập tham số Bài 7", expanded=True):
        n = st.slider("Số nghiệm mô phỏng", 200, 1500, 600, 100)
        seed = st.number_input("Seed", 1, 9999, 42)
    res = solve_bai7(n_samples=n, seed=int(seed))
    metric_cols([("Số nghiệm Pareto", str(res["n_pareto"]), "lọc không bị trội"), ("GDP thỏa hiệp", f"{res['best_objectives'][0]:.0f}", "f1"), ("Bất bình đẳng", f"{res['best_objectives'][1]:.0f}", "f2"), ("Rủi ro dữ liệu", f"{res['best_objectives'][3]:.0f}", "f4")])
    tab1, tab2, tab3 = st.tabs(["Tập Pareto", "Biểu đồ 3D", "Nghiệm thỏa hiệp"])
    with tab1:
        show_df(res["pareto"].round(3), 420)
    with tab2:
        fig = px.scatter_3d(res["pareto"], x="Tăng trưởng GDP", y="Bất bình đẳng", z="Phát thải", color="Điểm TOPSIS thỏa hiệp", title="Biên Pareto 3 chiều")
        st.plotly_chart(fig, width="stretch")
    with tab3:
        show_df(res["best_allocation"].round(2), 300)
        st.plotly_chart(px.imshow(res["best_allocation"], text_auto=".0f", aspect="auto", title="Phân bổ nghiệm thỏa hiệp"), width="stretch")
    # Bổ sung bảng đánh đổi: so sánh nghiệm thỏa hiệp với nghiệm tăng trưởng cao nhất
    growth_best = res["pareto"].sort_values("Tăng trưởng GDP", ascending=False).iloc[0]
    compromise = res["pareto"].sort_values("Điểm TOPSIS thỏa hiệp", ascending=False).iloc[0]
    tradeoff = pd.DataFrame([
        {"Phương án": "Tăng trưởng cao nhất", "Tăng trưởng GDP (tỷ VND)": growth_best["Tăng trưởng GDP"], "Bất bình đẳng": growth_best["Bất bình đẳng"], "Phát thải": growth_best["Phát thải"], "Rủi ro dữ liệu": growth_best["Rủi ro dữ liệu"]},
        {"Phương án": "Thỏa hiệp TOPSIS", "Tăng trưởng GDP (tỷ VND)": compromise["Tăng trưởng GDP"], "Bất bình đẳng": compromise["Bất bình đẳng"], "Phát thải": compromise["Phát thải"], "Rủi ro dữ liệu": compromise["Rủi ro dữ liệu"]},
    ])
    with st.expander("Bổ sung: chi phí cơ hội giữa tăng trưởng và các mục tiêu khác", expanded=False):
        show_df(tradeoff.round(3), 140)
        st.plotly_chart(px.bar(tradeoff, x="Phương án", y=["Tăng trưởng GDP (tỷ VND)", "Bất bình đẳng", "Phát thải", "Rủi ro dữ liệu"], barmode="group", title="Đánh đổi giữa nghiệm tăng trưởng cao nhất và nghiệm thỏa hiệp"), width="stretch")
    ai_box("Bài 7 Pareto đa mục tiêu", {"top": "nghiệm thỏa hiệp TOPSIS", "z": float(res["best_objectives"][0]), "unit": "tỷ VND GDP gain cho f1; f2-f4 là chỉ số mô phỏng", "n_pareto": res["n_pareto"], "growth_best": tradeoff.round(3).to_dict("records")})

elif page.startswith("Bài 8 —"):
    hero("Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035", "Tối đa hóa phúc lợi chiết khấu với động học K, D, AI, H và hàm sản xuất Cobb-Douglas.", ["Dynamic optimization", "SLSQP", "Welfare"])
    styled_formula("max ΣρᵗU(Cₜ); Kₜ₊₁=(1−δK)Kₜ+Iᴷₜ; D, AI, H cập nhật theo phương trình động học.")
    unit_note("Y, C, K và các khoản đầu tư I_K/I_D/I_AI/I_H hiển thị theo nghìn tỷ VND; D là chỉ số/%, AI là nghìn DN/năng lực, H là % lao động qua đào tạo; welfare là điểm hữu dụng chiết khấu, không phải tiền tệ.")
    with st.expander("Thiết lập tham số Bài 8", expanded=True):
        rho = st.slider("ρ chiết khấu", 0.85, 0.995, 0.97, 0.005)
        utility = st.selectbox("Hàm thỏa dụng", ["log", "crra"])
        shock = st.checkbox("Cú sốc 2028: Y giảm 8%", False)
    with st.spinner("Đang tối ưu SLSQP..."):
        res = solve_bai8(rho=rho, utility=utility, shock_2028=shock, maxiter=80)
    metric_cols([("Welfare tối ưu", f"{res['welfare']:.2f}", utility), ("Trải đều", f"{res['even_welfare']:.2f}", "so sánh"), ("Front-load", f"{res['front_welfare']:.2f}", "so sánh"), ("Trạng thái", "OK" if res["success"] else "Cần xem", res["message"][:32])])
    tab1, tab2 = st.tabs(["Quỹ đạo tối ưu", "So sánh chiến lược"])
    with tab1:
        show_df(res["path"].round(3), 420)
        fig = px.line(res["path"], x="Năm", y=["Y", "C"], markers=True, title="Quỹ đạo Y và C")
        st.plotly_chart(fig, width="stretch")
        fig2 = px.line(res["path"], x="Năm", y=["K", "D", "AI", "H"], markers=True, title="Quỹ đạo trạng thái K, D, AI, H")
        st.plotly_chart(fig2, width="stretch")
    with tab2:
        comp = pd.DataFrame({"Chiến lược": ["Tối ưu", "Trải đều", "Front-load"], "Welfare": [res["welfare"], res["even_welfare"], res["front_welfare"]]})
        st.plotly_chart(px.bar(comp, x="Chiến lược", y="Welfare", title="So sánh welfare tổng"), width="stretch")
    invest_cols = ["I_K", "I_D", "I_AI", "I_H"]
    inv_summary = res["path"][invest_cols].sum().reset_index()
    inv_summary.columns = ["Hạng mục đầu tư", "Tổng đầu tư 2026-2035 (nghìn tỷ VND)"]
    with st.expander("Bổ sung: tổng đầu tư theo hạng mục 2026-2035", expanded=False):
        show_df(inv_summary.round(2), 180)
        st.plotly_chart(px.pie(inv_summary, names="Hạng mục đầu tư", values="Tổng đầu tư 2026-2035 (nghìn tỷ VND)", hole=.45, title="Cơ cấu đầu tư tích lũy 2026-2035"), width="stretch")
    ai_box("Bài 8 tối ưu động", {"z": res["welfare"], "unit": "điểm welfare; Y/C/đầu tư là nghìn tỷ VND", "Y2026": float(res["path"].iloc[0]["Y"]), "Y2035": float(res["path"].iloc[-1]["Y"]), "C2026": float(res["path"].iloc[0]["C"]), "C2035": float(res["path"].iloc[-1]["C"]), "investment_summary": inv_summary.round(2).to_dict("records"), "warning": "Kết quả phụ thuộc hệ số quy đổi đầu tư thành D, AI, H; cần phân tích độ nhạy."})

elif page.startswith("Bài 9 —"):
    hero("Bài 9 — Tác động AI tới thị trường lao động", "Tối ưu phân bổ ngân sách AI và đào tạo lại để NetJob ròng không âm và tốc độ tự động hóa không vượt năng lực đào tạo lại.", ["Labor simulation", "LP", "NetJob"])
    styled_formula("NetJobᵢ = NewJobᵢ + UpgradeJobᵢ − DisplacedJobᵢ; DisplacedJobᵢ ≤ RetrainingCapacityᵢ")
    unit_note("Ngân sách x_AI, x_H tính bằng tỷ VND; NewJob/UpgradeJob/DisplacedJob/NetJob là số việc làm mô phỏng theo hệ số của bài.")
    with st.expander("Thiết lập tham số Bài 9", expanded=True):
        budget = st.slider("Ngân sách lao động", 10000.0, 50000.0, 30000.0, 1000.0)
        cap = st.slider("Trần mỗi ngành", 3000.0, 15000.0, 8000.0, 500.0)
        disp5 = st.checkbox("Thêm ràng buộc mất việc ≤ 5% lao động", False)
    res = solve_bai9(budget=budget, max_sector_budget=cap, max_displaced_pct=0.05 if disp5 else None)
    if not res["success"]:
        st.error(res["message"])
    else:
        metric_cols([("Tổng NetJob", f"{res['total_netjob']:.0f}", "việc làm"), ("Ngưỡng ngành 2", f"xH ≥ {res['threshold_sector2_ratio']:.2f}·xAI", "đào tạo tối thiểu"), ("Ngân sách", f"{budget:.0f}", "tỷ VND"), ("Ràng buộc 5%", "Có" if disp5 else "Không", "mở rộng")])
        tab1, tab2 = st.tabs(["Bảng lao động", "Biểu đồ"])
        with tab1:
            show_df(res["result"].round(3), 430)
        st.caption("x_AI và x_H: tỷ VND; các cột NewJob/UpgradeJob/DisplacedJob/RetrainingCapacity/NetJob: số việc làm mô phỏng.")
        with tab2:
            st.plotly_chart(px.bar(res["result"], x="Ngành", y=["NewJob", "UpgradeJob", "DisplacedJob", "NetJob"], barmode="group", title="Tạo việc làm, nâng cấp việc làm và dịch chuyển"), width="stretch")
            st.plotly_chart(px.bar(res["result"], x="Ngành", y=["x_AI", "x_H"], barmode="stack", title="Phân bổ ngân sách AI và đào tạo"), width="stretch")
        top_labor = res["result"].sort_values("NetJob", ascending=False).head(3)
        with st.expander("Bổ sung: Top 3 ngành theo NetJob", expanded=False):
            show_df(top_labor[["Ngành", "x_AI", "x_H", "DisplacedJob", "RetrainingCapacity", "NetJob"]].round(2), 160)
        ai_box("Bài 9 Lao động và AI", {"z": res["total_netjob"], "unit": "việc làm mô phỏng; ngân sách là tỷ VND", "budget": budget, "top": top_labor.iloc[0]["Ngành"], "top3": top_labor[["Ngành", "x_AI", "x_H", "DisplacedJob", "RetrainingCapacity", "NetJob"]].round(2).to_dict("records")})

elif page.startswith("Bài 10 —"):
    hero("Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn", "Ra quyết định here-and-now và recourse dưới 4 kịch bản bất định; tính VSS và EVPI.", ["Stochastic programming", "VSS", "EVPI"])
    styled_formula("max Σβⱼxⱼ + ΣₛpₛΣβˢⱼyˢⱼ; Σx≤65.000; Σyˢ≤15.000; y_AIˢ≤0,5x_H")
    unit_note("x và y tính bằng tỷ VND theo đề; Z, EV, VSS và EVPI là tỷ VND GDP gain kỳ vọng. App có thêm cột quy đổi nghìn tỷ VND để đọc nhanh, nhưng đơn vị chính vẫn là tỷ VND.")
    res = solve_bai10()
    metric_cols([("SP value", f"{res['sp_value']:,.0f}".replace(",", "."), "tỷ VND"), ("EV value", f"{res['ev_value']:,.0f}".replace(",", "."), "tỷ VND"), ("VSS", f"{res['vss']:,.0f}".replace(",", "."), "tỷ VND"), ("EVPI", f"{res['evpi']:,.0f}".replace(",", "."), "tỷ VND")])
    tab0, tab1, tab2, tab3 = st.tabs(["Kịch bản & hệ số", "First-stage", "Recourse theo kịch bản", "Giá trị bất định"])
    with tab0:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cây kịch bản**")
            show_df(res["scenario_tree"].round(3), 220)
        with c2:
            st.markdown("**Hệ số β theo kịch bản**")
            show_df(res["beta_table"].round(3), 220)
        st.plotly_chart(px.imshow(res["beta_table"].set_index("Hạng mục"), text_auto=".2f", aspect="auto", title="Heatmap hệ số hiệu quả β"), width="stretch")
    with tab1:
        show_df(res["first_stage"].round(3), 220)
        st.plotly_chart(px.bar(res["first_stage"], x="Hạng mục", y="x first-stage (tỷ VND)", title="Quyết định here-and-now (tỷ VND)"), width="stretch")
    with tab2:
        show_df(res["recourse"].round(3), 340)
        st.plotly_chart(px.bar(res["recourse"], x="Kịch bản", y="y recourse (tỷ VND)", color="Hạng mục", barmode="stack", title="Điều chỉnh ngân sách theo kịch bản (tỷ VND)"), width="stretch")
    with tab3:
        show_df(res["value_compare"].round(3), 220)
        st.plotly_chart(px.bar(res["value_compare"], x="Chỉ tiêu", y="Giá trị (tỷ VND)", text="Giá trị (tỷ VND)", title="SP, EV, VSS và EVPI (tỷ VND)"), width="stretch")
    ai_box("Bài 10 Stochastic SP", {"z": res["sp_value"], "unit": "tỷ VND GDP gain kỳ vọng", "ev_value": res["ev_value"], "vss": res["vss"], "evpi": res["evpi"], "first_stage": res["first_stage"].round(2).to_dict("records"), "recourse_top": res["recourse"].sort_values("y recourse (tỷ VND)", ascending=False).head(4).round(2).to_dict("records"), "top": res["first_stage"].sort_values("x first-stage (tỷ VND)", ascending=False).iloc[0]["Hạng mục"], "warning": f"VSS={res['vss']:,.0f} tỷ VND, EVPI={res['evpi']:,.0f} tỷ VND."})

elif page.startswith("Bài 11 —"):
    hero("Bài 11 — Q-learning cho chính sách kinh tế thích nghi", "Mô phỏng MDP 81 trạng thái, 5 hành động ngân sách, huấn luyện chính sách π* bằng tabular Q-learning.", ["Reinforcement Learning", "Q-learning", "MDP"])
    styled_formula("Rₜ = w₁ΔGDP − w₂Δunemploy − w₃CyberRisk − w₄Emission, với w=(0,40;0,25;0,20;0,15)")
    unit_note("Trạng thái rời rạc 3 mức; hành động là tỷ trọng phân bổ; reward là điểm welfare mô phỏng, không phải tiền tệ.")
    with st.expander("Thiết lập tham số Bài 11", expanded=True):
        episodes = st.slider("Số episode huấn luyện", 500, 10000, 3000, 500)
        seed = st.number_input("Seed RL", 1, 9999, 42)
    with st.spinner("Đang huấn luyện Q-learning..."):
        res = solve_bai11(episodes=episodes, seed=int(seed))
    metric_cols([("Episode", str(episodes), "huấn luyện"), ("Policy 2026", res["policy"].iloc[0]["Hành động khuyến nghị"], "trạng thái thực tế"), ("Best rule", res["rule_compare"].sort_values("Reward TB", ascending=False).iloc[0]["Chính sách"], "so sánh"), ("Số hành động", "5", "a0-a4")])
    tab1, tab2, tab3 = st.tabs(["Learning curve", "Policy trích xuất", "So sánh rule-based"])
    with tab1:
        st.plotly_chart(px.line(res["curve"], x="Episode", y="Reward", title="Learning curve Q-learning"), width="stretch")
    with tab2:
        show_df(res["policy"], 280)
    with tab3:
        show_df(res["rule_compare"].round(3), 240)
        st.plotly_chart(px.bar(res["rule_compare"], x="Chính sách", y="Reward TB", title="Reward tích lũy trung bình"), width="stretch")
        st.markdown("**Tần suất hành động trong chính sách trích xuất**")
        st.plotly_chart(px.bar(res["action_distribution"], x="Hành động", y="Số trạng thái mẫu", text="Số trạng thái mẫu", title="Tần suất hành động trên 5 trạng thái kiểm tra"), width="stretch")
    best_rule = res["rule_compare"].sort_values("Reward TB", ascending=False).iloc[0]
    with st.expander("Bổ sung: mô tả 5 hành động ngân sách", expanded=False):
        action_table = pd.DataFrame([{"Mã": f"a{k}", "Tên": v[0], "K": v[1][0], "D": v[1][1], "AI": v[1][2], "H": v[1][3]} for k, v in ACTIONS.items()])
        show_df(action_table, 210)
    ai_box("Bài 11 Q-learning RL", {"top": res["policy"].iloc[0]["Hành động khuyến nghị"], "unit": "điểm reward/welfare mô phỏng, không phải tiền tệ", "episodes": episodes, "eval_reward_mean": res["eval_reward_mean"], "eval_reward_std": res["eval_reward_std"], "best_rule": best_rule["Chính sách"], "best_reward": float(best_rule["Reward TB"]), "policy": res["policy"].to_dict("records"), "action_distribution": res["action_distribution"].to_dict("records"), "warning": "RL chỉ hỗ trợ ra quyết định, không thay thế thảo luận chính trị - xã hội."})

elif page.startswith("Bài 12 —"):
    hero("Bài 12 — AIDEOM-VN Dashboard tích hợp", "Tích hợp 6 module: Dự báo (M1), Sẵn sàng số (M2), Phân bổ (M3), Lao động (M4), Đánh giá rủi ro (M5) và Dashboard ra quyết định (M6).", ["Tích hợp", "6 module", "5 kịch bản chính sách"])
    styled_formula("Mô hình AIDEOM-VN tích hợp: M1 + M2 → M3 → M4/M5 → M6. Kết quả dùng để so sánh 5 kịch bản chính sách trong Mục 15.")
    unit_note("M1: GDP/Y là nghìn tỷ VND. M3 và GDP_gain kịch bản: tỷ VND. Phát_thải, Rủi_ro_rộng và các điểm chuẩn hóa là chỉ số mô phỏng không có đơn vị tiền tệ. NetJob là số việc làm mô phỏng.")

    with st.expander("Thiết lập tham số Bài 12", expanded=True):
        budget = st.slider("Tổng ngân sách kịch bản (tỷ VND)", 50000.0, 100000.0, 80000.0, 5000.0)
        st.caption("Mặc định 80.000 tỷ VND để khớp bảng demo của thầy.")
    res = solve_bai12(total_budget_2026_2030=budget)
    scenarios = res["scenario_table"]
    best = scenarios.sort_values("GDP_gain", ascending=False).iloc[0]
    risk_row = scenarios.sort_values("Phát_thải").iloc[0]

    metric_cols([
        ("Kịch bản tốt nhất", best["Kịch_bản"], "theo GDP_gain"),
        ("GDP_gain cao nhất", f"{best['GDP_gain']:,.0f}".replace(",", "."), "tỷ VND"),
        ("Phát thải thấp nhất", risk_row["Kịch_bản"], f"{risk_row['Phát_thải']:.3f}"),
        ("Top vùng AI", res["top_regions"].iloc[0]["region_name_vi"], "TOPSIS")
    ])

    tab0, tab1, tab2, tab3 = st.tabs(["Tổng quan (M1-M2)", "Phân bổ (M3)", "5 kịch bản chính", "Cảnh báo rủi ro (M4-M5)"])

    with tab0:
        st.markdown("### M1 — Dự báo kinh tế")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE Cobb-Douglas", f"{res['m1']['mape']:.2f}%")
        c2.metric("A hiệu chỉnh", f"{res['m1']['A']:.4f}")
        c3.metric("Y2030 dự báo", f"{res['m1']['Y2030']:,.0f} ng.tỷ".replace(",", "."))
        st.plotly_chart(px.bar(res["m1"]["growth_accounting"], x="Yếu tố", y="Đóng góp", title="Phân rã đóng góp tăng trưởng 2020-2025"), width="stretch")

        st.markdown("### M2 — Đánh giá sẵn sàng số (TOPSIS)")
        c1, c2 = st.columns(2)
        with c1:
            show_df(res["top_regions"].round(4), 260)
        with c2:
            st.plotly_chart(px.bar(res["top_regions"], x="region_name_vi", y="TOPSIS chuyên gia", title="TOPSIS theo vùng"), width="stretch")

    with tab1:
        st.markdown("### M3 — Tối ưu phân bổ ngân sách số (LP 6 vùng × 4 hạng mục)")
        fair = st.toggle("Bắt ràng buộc công bằng vùng", value=True)
        lam_m3 = 0.60 if fair else 0.50
        m3 = solve_bai4(total_budget=50000.0, lam=lam_m3, fairness=fair, auto_relax=True)
        if not m3["success"]:
            st.error(m3["message"])
        else:
            c1, c2 = st.columns(2)
            c1.metric("Z GDP_gain", f"{m3['z']:,.0f} tỷ VND".replace(",", "."))
            c2.metric("Tổng vốn", "50.000")
            st.caption("Với λ = 0,60, mô hình phân bổ bảo đảm công bằng vùng và cho Z GDP_gain xấp xỉ 60.760 tỷ VND.")
            st.plotly_chart(px.imshow(m3["allocation"], text_auto=".0f", aspect="auto", title="Phân bổ ngân sách tối ưu (tỷ VND)"), width="stretch")
            c1, c2 = st.columns(2)
            c1.markdown("**Tổng theo vùng**")
            c1.dataframe(m3["region_total"].round(2), width="stretch")
            c2.markdown("**Tổng theo hạng mục**")
            c2.dataframe(m3["item_total"].round(2), width="stretch")

    with tab2:
        st.markdown("### So sánh 5 kịch bản chính sách (Mục 15)")
        show_df(scenarios.round(3), 300)
        st.plotly_chart(px.bar(scenarios, x="Kịch_bản", y="GDP_gain", title="GDP gain theo kịch bản", text="GDP_gain"), width="stretch")
        st.plotly_chart(px.scatter(scenarios, x="Phát_thải", y="Rủi_ro_rộng", size="GDP_gain", color="Kịch_bản", title="Bubble: GDP × Phát thải × Rủi ro"), width="stretch")
        radar = scenarios[["Kịch_bản", "GDP_gain_chuẩn_hóa", "Phát_thải_chuẩn_hóa", "Rủi_ro_chuẩn_hóa"]].melt(id_vars="Kịch_bản", var_name="KPI", value_name="Điểm")
        st.plotly_chart(px.line_polar(radar, r="Điểm", theta="KPI", color="Kịch_bản", line_close=True, title="Radar so sánh KPI đã chuẩn hóa"), width="stretch")

    with tab3:
        st.markdown("### Cảnh báo rủi ro các mục tiêu")
        warning = res["warnings"]
        c1, c2, c3 = st.columns(3)
        c1.metric("NetJob mô phỏng", f"{warning['NetJob']:,.0f}".replace(",", "."))
        c2.metric("Rủi ro dữ liệu", f"{warning['Rủi_ro_dữ_liệu']:+.1f}%")
        c3.metric("Phát thải tương đối", f"{warning['Phát_thải_tương_đối']:+.1f}%")
        with st.expander("Giải thích cảnh báo chính sách", expanded=True):
            st.markdown(warning["Khuyến_nghị"])
        st.plotly_chart(px.bar(scenarios, x="Kịch_bản", y=["Phát_thải", "Rủi_ro_rộng"], barmode="group", title="Cảnh báo phát thải và rủi ro theo kịch bản"), width="stretch")
        st.markdown("### Ngành ưu tiên bổ trợ cho S5")
        show_df(res["top_sectors"].round(4), 180)
        st.plotly_chart(px.bar(res["top_sectors"], x="Priority", y="sector_name_vi", orientation="h", title="Top ngành ưu tiên để triển khai mô hình tích hợp"), width="stretch")

    ai_box("Bài 12 AIDEOM-VN tích hợp", {
        "best": best["Kịch_bản"],
        "GDP_gain": float(best["GDP_gain"]),
        "unit": "tỷ VND GDP gain kỳ vọng",
        "emission_best": float(best["Phát_thải"]),
        "risk_best": float(best["Rủi_ro_rộng"]),
        "risk_min": risk_row["Kịch_bản"],
        "scenario_table": scenarios[["Kịch_bản", "GDP_gain", "Phát_thải", "Rủi_ro_rộng"]].round(3).to_dict("records"),
        "warning": warning["Khuyến_nghị"]
    })
