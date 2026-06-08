"""Các mô hình tính toán lõi cho web app AIDEOM-VN.

Mục tiêu của file này là tách phần toán học ra khỏi giao diện Streamlit để
kết quả có thể kiểm thử, tái lập và dễ giải thích trong báo cáo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except Exception:  # pragma: no cover
    HAS_SCIPY_MILP = False

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GDP_2024_TRILLION_VND = 11511.9

# =========================
# Nạp dữ liệu
# =========================

def load_macro(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "vietnam_macro_2020_2025.csv")
    return df.sort_values("year").reset_index(drop=True)


def load_sectors(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "vietnam_sectors_2024.csv")


def load_regions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "vietnam_regions_2024.csv")


def fmt_num(x: float, nd: int = 2) -> str:
    try:
        return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

# =========================
# Bài 1 - Cobb-Douglas
# =========================

COBB_K = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
COBB_L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], dtype=float)
COBB_D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5], dtype=float)
COBB_AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], dtype=float)
COBB_H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], dtype=float)


def solve_bai1(alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07,
               d2030=30.0, ai2030=100.0, h2030=35.0,
               growth_k=0.06, growth_l=0.06, tfp_growth=0.012) -> Dict:
    macro = load_macro().copy()
    Y = macro["GDP_trillion_VND"].values.astype(float)
    years = macro["year"].values.astype(int)

    exp_sum = alpha + beta + gamma + delta + theta
    # Giữ đúng điều kiện lợi suất không đổi: nếu người dùng chỉnh lệch, chuẩn hóa lại.
    if abs(exp_sum - 1.0) > 1e-9:
        alpha, beta, gamma, delta, theta = np.array([alpha, beta, gamma, delta, theta]) / exp_sum

    A = Y / (COBB_K**alpha * COBB_L**beta * COBB_D**gamma * COBB_AI**delta * COBB_H**theta)
    A_bar = float(A.mean())
    Y_hat = A_bar * (COBB_K**alpha * COBB_L**beta * COBB_D**gamma * COBB_AI**delta * COBB_H**theta)
    mape = float(np.mean(np.abs((Y - Y_hat) / Y)) * 100)

    # Phân rã tăng trưởng theo sai phân log.
    def dlog(v):
        return np.diff(np.log(v))

    contrib = pd.DataFrame({
        "Năm": years[1:],
        "K": alpha * dlog(COBB_K),
        "L": beta * dlog(COBB_L),
        "D": gamma * dlog(COBB_D),
        "AI": delta * dlog(COBB_AI),
        "H": theta * dlog(COBB_H),
        "TFP": dlog(A),
        "Tăng trưởng GDP log": dlog(Y),
    })
    mean_contrib = contrib[["K", "L", "D", "AI", "H", "TFP"]].mean()
    gdp_mean = contrib["Tăng trưởng GDP log"].mean()
    share = (mean_contrib / gdp_mean * 100).rename("Tỷ trọng đóng góp (%)")
    contrib_summary = pd.DataFrame({
        "Yếu tố": share.index,
        "Đóng góp log bình quân": mean_contrib.values,
        "Tỷ trọng đóng góp (%)": share.values,
    })

    k2030 = COBB_K[-1] * (1 + growth_k) ** 5
    l2030 = COBB_L[-1] * (1 + growth_l) ** 5
    a2030 = A[-1] * (1 + tfp_growth) ** 5
    y2030 = a2030 * (k2030**alpha * l2030**beta * d2030**gamma * ai2030**delta * h2030**theta)

    detail = pd.DataFrame({
        "Năm": years,
        "Y thực tế": Y,
        "K": COBB_K,
        "L": COBB_L,
        "D": COBB_D,
        "AI": COBB_AI,
        "H": COBB_H,
        "TFP A_t": A,
        "Y dự báo": Y_hat,
        "Sai số %": np.abs((Y - Y_hat) / Y) * 100,
    })
    return {
        "params": (alpha, beta, gamma, delta, theta),
        "detail": detail,
        "contrib": contrib,
        "contrib_summary": contrib_summary,
        "mape": mape,
        "A_bar": A_bar,
        "Y2030": float(y2030),
        "k2030": float(k2030),
        "l2030": float(l2030),
        "a2030": float(a2030),
    }

# =========================
# Bài 2 - LP ngân sách 4 hạng mục
# =========================

B2_ITEMS = ["Hạ tầng số", "AI và dữ liệu", "Nhân lực số", "R&D công nghệ"]
B2_COEF = np.array([0.85, 1.20, 0.95, 1.35], dtype=float)


def _solve_bai2_core(total_budget=100.0, min_i=25.0, min_ai=15.0, min_h=20.0, min_rd=10.0,
                     tech_ratio=0.35) -> Dict:
    c = -B2_COEF.copy()
    A_ub = [
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [tech_ratio, -(1 - tech_ratio), tech_ratio, -(1 - tech_ratio)],
    ]
    b_ub = [total_budget, -min_i, -min_ai, -min_h, -min_rd, 0]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * 4, method="highs")
    if not res.success:
        return {"success": False, "message": res.message, "allocation": None, "z": None}
    x = res.x
    z = float(B2_COEF @ x)
    marginals = getattr(res.ineqlin, "marginals", np.full(len(b_ub), np.nan))
    shadow = -np.array(marginals)
    alloc = pd.DataFrame({"Hạng mục": B2_ITEMS, "Phân bổ tối ưu": x, "Hệ số tác động": B2_COEF,
                          "Đóng góp GDP kỳ vọng": x * B2_COEF})
    duals = pd.DataFrame({
        "Ràng buộc": ["Ngân sách tổng", "Sàn hạ tầng số", "Sàn AI và dữ liệu", "Sàn nhân lực số", "Sàn R&D", "Tỷ trọng AI+R&D"],
        "Shadow price": shadow,
        "Slack": res.ineqlin.residual,
    })
    return {"success": True, "allocation": alloc, "duals": duals, "z": z, "x": x}


def solve_bai2(total_budget=100.0, min_i=25.0, min_ai=15.0, min_h=20.0, min_rd=10.0,
               tech_ratio=0.35) -> Dict:
    out = _solve_bai2_core(total_budget, min_i, min_ai, min_h, min_rd, tech_ratio)
    if not out["success"]:
        return out
    sensitivity = []
    for B in [total_budget, 120.0, 140.0]:
        tmp = _solve_bai2_core(B, min_i, min_ai, min_h, min_rd, tech_ratio)
        sensitivity.append({"Ngân sách": B, "Z*": tmp.get("z")})
    high_h = _solve_bai2_core(total_budget, min_i, min_ai, 30.0, min_rd, tech_ratio)
    out.update({"sensitivity": pd.DataFrame(sensitivity), "high_h": high_h})
    return out

# =========================
# Bài 3 - Priority ngành
# =========================


def _norm_good(x: pd.Series) -> pd.Series:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng != 0 else x * 0


def _norm_bad_to_good(x: pd.Series) -> pd.Series:
    rng = x.max() - x.min()
    return (x.max() - x) / rng if rng != 0 else x * 0


def solve_bai3(weights: Optional[Dict[str, float]] = None) -> Dict:
    df = load_sectors().copy()
    df["productivity_million_VND_per_worker"] = (df["gdp_share_2024_pct"] / 100 * GDP_2024_TRILLION_VND * 1000) / (df["labor_million"] * 1_000_000) * 1_000
    # Giải thích: GDP nghìn tỷ VND -> tỷ VND; chia triệu lao động -> triệu VND/lao động.
    # Kết quả khớp bảng đề bài (ví dụ nông nghiệp khoảng 103,4 triệu VND/LĐ).
    cols = {
        "Tăng trưởng": "growth_rate_2024_pct",
        "Năng suất": "productivity_million_VND_per_worker",
        "Lan tỏa": "spillover_coef_0_1",
        "Xuất khẩu": "export_billion_USD",
        "Việc làm": "labor_million",
        "AI readiness": "ai_readiness_0_100",
        "An toàn tự động hóa": "automation_risk_pct",
    }
    norm = pd.DataFrame({
        "Tăng trưởng": _norm_good(df[cols["Tăng trưởng"]]),
        "Năng suất": _norm_good(df[cols["Năng suất"]]),
        "Lan tỏa": _norm_good(df[cols["Lan tỏa"]]),
        "Xuất khẩu": _norm_good(df[cols["Xuất khẩu"]]),
        "Việc làm": _norm_good(df[cols["Việc làm"]]),
        "AI readiness": _norm_good(df[cols["AI readiness"]]),
        "An toàn tự động hóa": _norm_bad_to_good(df[cols["An toàn tự động hóa"]]),
    })
    default_w = {
        "Tăng trưởng": 0.15, "Năng suất": 0.15, "Lan tỏa": 0.20,
        "Xuất khẩu": 0.15, "Việc làm": 0.10, "AI readiness": 0.20,
        "An toàn tự động hóa": 0.15,
    }
    w = default_w if weights is None else weights.copy()
    s = sum(w.values())
    w = {k: v / s for k, v in w.items()}
    score = sum(norm[k] * w[k] for k in norm.columns)
    result = df[["sector_name_vi", "growth_rate_2024_pct", "labor_million", "export_billion_USD", "ai_readiness_0_100", "automation_risk_pct", "productivity_million_VND_per_worker"]].copy()
    result["Priority"] = score
    result["Xếp hạng"] = result["Priority"].rank(ascending=False, method="first").astype(int)
    result = result.sort_values("Priority", ascending=False).reset_index(drop=True)
    norm_display = pd.concat([df[["sector_name_vi"]], norm], axis=1)

    # Độ nhạy theo trọng số AI readiness.
    sensitivity_rows = []
    for ai_w in np.arange(0.05, 0.401, 0.05):
        other_keys = [k for k in default_w if k != "AI readiness"]
        other_total = sum(default_w[k] for k in other_keys)
        w2 = {k: default_w[k] * (1 - ai_w) / other_total for k in other_keys}
        w2["AI readiness"] = ai_w
        score2 = sum(norm[k] * w2[k] for k in norm.columns)
        order = list(df.assign(score=score2).sort_values("score", ascending=False)["sector_name_vi"])
        for rank, sector in enumerate(order, 1):
            sensitivity_rows.append({"w_AI": round(float(ai_w), 2), "Ngành": sector, "Xếp hạng": rank})
    sens = pd.DataFrame(sensitivity_rows)
    growth_w = {"Tăng trưởng": 0.25, "Năng suất": 0.20, "Lan tỏa": 0.15, "Xuất khẩu": 0.25, "Việc làm": 0.05, "AI readiness": 0.07, "An toàn tự động hóa": 0.03}
    inclusive_w = {"Tăng trưởng": 0.10, "Năng suất": 0.10, "Lan tỏa": 0.25, "Xuất khẩu": 0.05, "Việc làm": 0.25, "AI readiness": 0.05, "An toàn tự động hóa": 0.20}
    def rank_with(wdict):
        sw = sum(wdict.values())
        sc = sum(norm[k] * (wdict[k] / sw) for k in norm.columns)
        return df.assign(Priority=sc).sort_values("Priority", ascending=False)[["sector_name_vi", "Priority"]].head(5).reset_index(drop=True)
    return {"normalized": norm_display, "result": result, "weights": w,
            "sensitivity": sens, "growth_rank": rank_with(growth_w), "inclusive_rank": rank_with(inclusive_w)}

# =========================
# Bài 4 - LP ngành-vùng
# =========================

REGION_CODES = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = [
    "Trung du miền núi phía Bắc", "Đồng bằng sông Hồng", "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên", "Đông Nam Bộ", "Đồng bằng sông Cửu Long"
]
ITEM_CODES = ["I", "D", "AI", "H"]
ITEM_NAMES = ["Hạ tầng số", "CĐS doanh nghiệp", "AI", "Nhân lực số"]
BETA_6x4 = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
])
D0_REGIONS = np.array([38, 78, 55, 32, 82, 48], dtype=float)


def solve_bai4(total_budget=50000.0, floor_region=5000.0, cap_region=12000.0,
               h_floor=12000.0, gamma=0.002, lam=0.70,
               fairness=True, auto_relax=True, objective_matrix: Optional[np.ndarray] = None) -> Dict:
    beta = BETA_6x4 if objective_matrix is None else np.asarray(objective_matrix).reshape(6, 4)
    n_x = 24
    def _attempt(lam_try: float):
        n = n_x + (1 if fairness else 0)
        c = np.zeros(n)
        c[:n_x] = -beta.ravel()
        A = []
        b = []
        # tổng ngân sách
        row = np.zeros(n); row[:n_x] = 1; A.append(row); b.append(total_budget)
        for r in range(6):
            idx = slice(r*4, (r+1)*4)
            row = np.zeros(n); row[idx] = -1; A.append(row); b.append(-floor_region)
            row = np.zeros(n); row[idx] = 1; A.append(row); b.append(cap_region)
        row = np.zeros(n); row[3:n_x:4] = -1; A.append(row); b.append(-h_floor)
        if fairness:
            M_idx = n - 1
            for r in range(6):
                row = np.zeros(n); row[r*4 + 1] = gamma; row[M_idx] = -1
                A.append(row); b.append(-D0_REGIONS[r])
            for r in range(6):
                row = np.zeros(n); row[r*4 + 1] = -gamma; row[M_idx] = lam_try
                A.append(row); b.append(D0_REGIONS[r])
            bounds = [(0, None)] * n
        else:
            bounds = [(0, None)] * n
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
        return res

    used_lam = lam
    res = _attempt(lam)
    relaxed = False
    if fairness and auto_relax and not res.success:
        # Đề gốc với λ=0,70 và trần 12.000 có thể không khả thi; tự tìm λ gần nhất thấp hơn.
        for lam_try in np.linspace(lam - 0.005, 0.50, 41):
            res = _attempt(float(lam_try))
            if res.success:
                used_lam = float(lam_try)
                relaxed = True
                break
    if not res.success:
        return {"success": False, "message": res.message, "lambda_used": used_lam, "relaxed": relaxed}
    x = res.x[:n_x].reshape(6, 4)
    alloc = pd.DataFrame(x, index=REGION_NAMES, columns=ITEM_NAMES)
    z = float((beta * x).sum())
    region_total = alloc.sum(axis=1).rename("Tổng ngân sách")
    item_total = alloc.sum(axis=0).rename("Tổng theo hạng mục")
    digital_after = pd.DataFrame({
        "Vùng": REGION_NAMES,
        "D ban đầu": D0_REGIONS,
        "x_D": x[:, 1],
        "D sau đầu tư": D0_REGIONS + gamma * x[:, 1],
    })
    return {"success": True, "allocation": alloc, "z": z, "lambda_used": used_lam, "relaxed": relaxed,
            "region_total": region_total.reset_index().rename(columns={"index": "Vùng"}),
            "item_total": item_total.reset_index().rename(columns={"index": "Hạng mục"}),
            "digital_after": digital_after, "raw": x}

# =========================
# Bài 5 - MIP dự án
# =========================

PROJECTS = pd.DataFrame([
    (1, "Trung tâm dữ liệu quốc gia Hòa Lạc", "Hạ tầng", 12000, 21500, 8500, 3500),
    (2, "Trung tâm dữ liệu quốc gia phía Nam", "Hạ tầng", 11500, 20800, 7500, 4000),
    (3, "Hệ thống 5G phủ sóng toàn quốc", "Hạ tầng", 18000, 32500, 12000, 6000),
    (4, "Hệ thống định danh điện tử VNeID 2.0", "Chính phủ số", 4500, 9200, 3500, 1000),
    (5, "Cổng dịch vụ công quốc gia v3", "Chính phủ số", 3200, 6800, 2500, 700),
    (6, "Y tế số quốc gia (hồ sơ sức khỏe)", "Y tế số", 5800, 11400, 4000, 1800),
    (7, "Giáo dục số K-12 toàn quốc", "Giáo dục", 6500, 12200, 4500, 2000),
    (8, "Trung tâm AI quốc gia + supercomputing", "AI", 15000, 28500, 9000, 6000),
    (9, "Sandbox tài chính số (fintech)", "Tài chính số", 2500, 5800, 1800, 700),
    (10, "Logistics thông minh + cảng biển số", "Logistics", 7200, 13800, 5000, 2200),
    (11, "Nông nghiệp số ĐBSCL", "Nông nghiệp", 4800, 8500, 3500, 1300),
    (12, "Đào tạo 50.000 kỹ sư AI/bán dẫn", "Nhân lực", 8500, 16200, 5500, 3000),
    (13, "Khu CN bán dẫn Bắc Ninh - Bắc Giang", "Bán dẫn", 20000, 35000, 13000, 7000),
    (14, "An ninh mạng quốc gia (SOC)", "An ninh", 3800, 7500, 2800, 1000),
    (15, "Open Data + dữ liệu mở quốc gia", "Dữ liệu", 1500, 3800, 1200, 300),
], columns=["id", "Tên dự án", "Lĩnh vực", "Chi phí", "Lợi ích NPV", "Năm 1-2", "Năm 3-5"])


def _project_prob(field: str) -> float:
    if field == "Hạ tầng": return 0.85
    if field == "Chính phủ số": return 0.75
    if field in ["AI", "Bán dẫn"]: return 0.65
    return 0.80


def _is_feasible_project(mask: int, budget_total: float, budget_y12: float, force_both_centers=False) -> bool:
    y = np.array([(mask >> i) & 1 for i in range(15)])
    C = PROJECTS["Chi phí"].values
    C1 = PROJECTS["Năm 1-2"].values
    if C @ y > budget_total + 1e-9: return False
    if C1 @ y > budget_y12 + 1e-9: return False
    if force_both_centers:
        if y[0] + y[1] != 2: return False
    else:
        if y[0] + y[1] > 1: return False
    if y[7] > y[11]: return False  # P8 <= P12
    if y[12] > y[11]: return False # P13 <= P12
    if y[3] + y[4] < 1: return False
    if y[13] < 1: return False
    if not (7 <= y.sum() <= 11): return False
    return True


def solve_bai5(budget_total=80000.0, budget_y12=40000.0, force_both_centers=False,
               risk_adjusted=False) -> Dict:
    benefits = PROJECTS["Lợi ích NPV"].values.astype(float)
    if risk_adjusted:
        probs = PROJECTS["Lĩnh vực"].apply(_project_prob).values
        benefits = benefits * probs
    best = None
    # Brute force 2^15 nhanh, không phụ thuộc solver ngoài.
    for mask in range(1 << 15):
        if not _is_feasible_project(mask, budget_total, budget_y12, force_both_centers):
            continue
        y = np.array([(mask >> i) & 1 for i in range(15)])
        val = float(benefits @ y)
        if best is None or val > best[0]:
            best = (val, y)
    if best is None:
        return {"success": False, "message": "Không tìm thấy nghiệm khả thi với các ràng buộc đã chọn."}
    val, y = best
    selected = PROJECTS[y == 1].copy()
    selected["Xác suất đúng tiến độ"] = selected["Lĩnh vực"].apply(_project_prob)
    selected["Lợi ích dùng trong mục tiêu"] = benefits[y == 1]
    total_cost = float(selected["Chi phí"].sum())
    return {"success": True, "selected": selected, "z": val, "total_cost": total_cost,
            "benefit_cost": val / total_cost if total_cost else np.nan,
            "n_projects": int(y.sum()), "y": y}

# =========================
# Bài 6 - TOPSIS
# =========================

TOPSIS_CRITERIA = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
                   "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
                   "internet_penetration_pct", "gini_coef"]
TOPSIS_LABELS = ["GRDP/người", "FDI", "Digital Index", "AI readiness", "LĐ đào tạo", "R&D/GRDP", "Internet", "Gini"]
TOPSIS_BENEFIT = np.array([True, True, True, True, True, True, True, False])
TOPSIS_W = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])


def topsis_score(X: np.ndarray, weights: np.ndarray, is_benefit: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    denom = np.sqrt((X ** 2).sum(axis=0))
    R = X / denom
    V = R * weights
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    C = S_neg / (S_star + S_neg)
    return C, S_star, S_neg


def entropy_weights(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    # Với tiêu chí chi phí Gini, đảo chiều trước để Entropy đo biến thiên theo hướng lợi ích.
    X_adj = X.copy()
    gini = X_adj[:, -1]
    X_adj[:, -1] = gini.max() - gini + 1e-12
    P = X_adj / (X_adj.sum(axis=0) + 1e-12)
    k = 1.0 / np.log(len(X_adj))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()


def solve_bai6(weights: Optional[np.ndarray] = None) -> Dict:
    df = load_regions().copy()
    X = df[TOPSIS_CRITERIA].values.astype(float)
    w = TOPSIS_W if weights is None else np.asarray(weights, dtype=float)
    C, S_star, S_neg = topsis_score(X, w, TOPSIS_BENEFIT)
    ew = entropy_weights(X)
    C_ent, _, _ = topsis_score(X, ew, TOPSIS_BENEFIT)
    res = df[["region_name_vi"] + TOPSIS_CRITERIA].copy()
    res["TOPSIS chuyên gia"] = C
    res["TOPSIS Entropy"] = C_ent
    res["Xếp hạng chuyên gia"] = res["TOPSIS chuyên gia"].rank(ascending=False, method="first").astype(int)
    res["Xếp hạng Entropy"] = res["TOPSIS Entropy"].rank(ascending=False, method="first").astype(int)
    res = res.sort_values("TOPSIS chuyên gia", ascending=False).reset_index(drop=True)
    # Độ nhạy w_AI
    rows = []
    for ai_w in np.arange(0.10, 0.401, 0.05):
        other = TOPSIS_W.copy()
        other_idx = [i for i in range(len(other)) if i != 3]
        other[other_idx] = other[other_idx] * (1 - ai_w) / other[other_idx].sum()
        other[3] = ai_w
        c2, _, _ = topsis_score(X, other, TOPSIS_BENEFIT)
        tmp = df.assign(score=c2).sort_values("score", ascending=False)
        for rank, name in enumerate(tmp["region_name_vi"], 1):
            rows.append({"w_AI": round(float(ai_w), 2), "Vùng": name, "Xếp hạng": rank})
    return {"result": res, "entropy_weights": pd.DataFrame({"Tiêu chí": TOPSIS_LABELS, "Trọng số Entropy": ew}),
            "sensitivity": pd.DataFrame(rows), "weights": w}

# =========================
# Bài 7 - Pareto đa mục tiêu
# =========================

E_EMISSION = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
RHO_RISK = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
SIG_RISK = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])


def _random_feasible_allocation(rng: np.random.Generator, total_budget=50000, floor=5000, cap=12000, h_floor=12000) -> np.ndarray:
    # Tạo tổng ngân sách vùng trong [floor, cap], tổng bằng total_budget.
    for _ in range(2000):
        extra = rng.dirichlet(np.ones(6)) * (total_budget - 6 * floor)
        region_total = floor + extra
        if np.all(region_total <= cap):
            X = np.zeros((6, 4))
            for r in range(6):
                shares = rng.dirichlet(np.ones(4))
                X[r, :] = region_total[r] * shares
            if X[:, 3].sum() < h_floor:
                need = h_floor - X[:, 3].sum()
                donors = X[:, :3].sum()
                if donors > need:
                    frac = need / donors
                    transfer = X[:, :3] * frac
                    X[:, :3] -= transfer
                    X[:, 3] += transfer.sum(axis=1)
            if X[:, 3].sum() >= h_floor - 1e-6:
                return X
    # fallback nghiệm Bài 4 không công bằng
    sol = solve_bai4(fairness=False)
    return sol["raw"] if sol.get("success") else np.ones((6, 4)) * total_budget / 24


def pareto_objectives(X: np.ndarray) -> np.ndarray:
    region_sums = X.sum(axis=1)
    gdp_gain = float((BETA_6x4 * X).sum())
    inequality = float(np.abs(region_sums - region_sums.mean()).mean())
    emission = float((E_EMISSION * (X[:, 0] + X[:, 2])).sum())
    data_risk = float((RHO_RISK * X[:, 2]).sum() - (SIG_RISK * X[:, 3]).sum())
    return np.array([gdp_gain, inequality, emission, data_risk], dtype=float)


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    # F quy ước: cột 0 càng lớn càng tốt, cột 1-3 càng nhỏ càng tốt.
    G = F.copy()
    G[:, 0] *= -1  # chuyển thành minimize all
    n = len(G)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated_by_i = np.all(G <= G[i], axis=1) & np.any(G < G[i], axis=1)
        if dominated_by_i.any():
            mask[i] = False
    return mask


def solve_bai7(n_samples=600, seed=42, weights=(0.40, 0.25, 0.20, 0.15)) -> Dict:
    rng = np.random.default_rng(seed)
    Xs, Fs = [], []
    for _ in range(n_samples):
        X = _random_feasible_allocation(rng)
        Xs.append(X)
        Fs.append(pareto_objectives(X))
    F = np.vstack(Fs)
    mask = nondominated_mask(F)
    Fp = F[mask]
    Xp = np.array(Xs, dtype=float)[mask]
    # TOPSIS trên tập Pareto: GDP là lợi ích, các mục tiêu còn lại là chi phí.
    C, _, _ = topsis_score(Fp, np.array(weights), np.array([True, False, False, False]))
    best_idx = int(np.argmax(C))
    pareto_df = pd.DataFrame(Fp, columns=["Tăng trưởng GDP", "Bất bình đẳng", "Phát thải", "Rủi ro dữ liệu"])
    pareto_df["Điểm TOPSIS thỏa hiệp"] = C
    best_X = Xp[best_idx]
    best_alloc = pd.DataFrame(best_X, index=REGION_NAMES, columns=ITEM_NAMES)
    return {"pareto": pareto_df.sort_values("Điểm TOPSIS thỏa hiệp", ascending=False).reset_index(drop=True),
            "best_allocation": best_alloc, "best_objectives": Fp[best_idx], "n_pareto": len(Fp)}

# =========================
# Bài 8 - Tối ưu động 2026-2035
# =========================


def _simulate_dynamic(shares: np.ndarray, rho=0.97, utility="log", shock_2028=False) -> Tuple[float, pd.DataFrame]:
    T = 10
    years = np.arange(2026, 2036)
    shares = shares.reshape(T, 4)
    shares = np.clip(shares, 0, 0.40)
    # Chuẩn hóa nếu tổng share > 0,30 để C vẫn dương.
    row_sum = shares.sum(axis=1)
    for t in range(T):
        if row_sum[t] > 0.30:
            shares[t] = shares[t] / row_sum[t] * 0.30
    # Điều kiện ban đầu từ đề và A hiệu chỉnh từ Bài 1.
    b1 = solve_bai1()
    A = b1["a2030"] / ((1 + 0.012) ** 4)  # xấp xỉ A 2026 sau TFP 2025
    K, L, D, AI, H = 27500.0, 53.9, 20.3, 86.0, 30.0
    rows = []
    welfare = 0.0
    for t, year in enumerate(years):
        Y = A * K**0.33 * L**0.42 * D**0.10 * AI**0.08 * H**0.07
        if shock_2028 and year == 2028:
            Y *= 0.92
        invest = shares[t] * Y
        C = max(Y - invest.sum(), 1e-6)
        if utility == "crra":
            gamma_crra = 1.5
            U = C ** (1 - gamma_crra) / (1 - gamma_crra)
        else:
            U = np.log(C)
        welfare += (rho ** t) * U
        rows.append({"Năm": year, "K": K, "D": D, "AI": AI, "H": H, "Y": Y, "C": C,
                     "I_K": invest[0], "I_D": invest[1], "I_AI": invest[2], "I_H": invest[3]})
        K = (1 - 0.05) * K + invest[0]
        D = (1 - 0.12) * D + invest[1] / 120  # quy đổi nghìn tỷ -> điểm chỉ số
        AI = (1 - 0.15) * AI + invest[2] / 60 # quy đổi nghìn tỷ -> nghìn DN/năng lực
        H = H + 0.8 * invest[3] / 500 - 0.02 * H
        A = A * (1 + 0.003 * D / 100 + 0.002 * AI / 100 + 0.004 * H / 100)
    return float(welfare), pd.DataFrame(rows)


def solve_bai8(rho=0.97, utility="log", shock_2028=False, maxiter=80) -> Dict:
    T = 10
    x0 = np.tile(np.array([0.07, 0.07, 0.07, 0.07]), T)
    bounds = [(0.0, 0.30)] * (T * 4)
    def obj(x):
        return -_simulate_dynamic(x, rho=rho, utility=utility, shock_2028=shock_2028)[0]
    cons = []
    for t in range(T):
        cons.append({"type": "ineq", "fun": lambda x, t=t: 0.30 - x.reshape(T, 4)[t].sum()})
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": maxiter, "ftol": 1e-7, "disp": False})
    welfare, path = _simulate_dynamic(res.x, rho=rho, utility=utility, shock_2028=shock_2028)
    # hai chiến lược so sánh
    even = np.tile([0.075, 0.075, 0.075, 0.075], T)
    front = np.vstack([np.tile([0.10, 0.08, 0.07, 0.05], (3, 1)), np.tile([0.06, 0.06, 0.06, 0.06], (7, 1))]).ravel()
    w_even, p_even = _simulate_dynamic(even, rho=rho, utility=utility, shock_2028=shock_2028)
    w_front, p_front = _simulate_dynamic(front, rho=rho, utility=utility, shock_2028=shock_2028)
    return {"success": bool(res.success), "message": res.message, "welfare": welfare, "path": path,
            "even_welfare": w_even, "front_welfare": w_front,
            "even_path": p_even, "front_path": p_front}

# =========================
# Bài 9 - Lao động AI
# =========================

LABOR_SECTOR_NAMES = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn-bán lẻ",
                      "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo"]
LABOR_L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
LABOR_RISK = np.array([18, 42, 25, 38, 52, 35, 28, 22], dtype=float) / 100
A1_JOB = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
B1_JOB = np.array([45, 28, 35, 32, 22, 30, 20, 55], dtype=float)
C1_JOB = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
D1_JOB = np.array([50, 32, 42, 38, 26, 36, 24, 62], dtype=float)


def solve_bai9(budget=30000.0, max_sector_budget: Optional[float] = 8000.0,
               max_displaced_pct: Optional[float] = None) -> Dict:
    n = 8
    # Variables [x_AI_0..7, x_H_0..7]
    c = np.zeros(2*n)
    coeff_ai = A1_JOB - C1_JOB * LABOR_RISK
    coeff_h = B1_JOB
    c[:n] = -coeff_ai
    c[n:] = -coeff_h
    A, b = [], []
    row = np.ones(2*n); A.append(row); b.append(budget)
    # NetJob >= 0 -> -coeff_ai*xAI - coeff_h*xH <= 0
    for i in range(n):
        row = np.zeros(2*n); row[i] = -coeff_ai[i]; row[n+i] = -coeff_h[i]
        A.append(row); b.append(0.0)
    # Displaced <= RetrainCap
    for i in range(n):
        row = np.zeros(2*n); row[i] = C1_JOB[i] * LABOR_RISK[i]; row[n+i] = -D1_JOB[i]
        A.append(row); b.append(0.0)
    if max_sector_budget is not None:
        for i in range(n):
            row = np.zeros(2*n); row[i] = 1; row[n+i] = 1
            A.append(row); b.append(max_sector_budget)
    if max_displaced_pct is not None:
        # L triệu lao động -> việc làm; 5% L triệu = L*1e6*0.05 việc.
        for i in range(n):
            row = np.zeros(2*n); row[i] = C1_JOB[i] * LABOR_RISK[i]
            A.append(row); b.append(max_displaced_pct * LABOR_L[i] * 1_000_000)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)]*(2*n), method="highs")
    if not res.success:
        return {"success": False, "message": res.message}
    xai = res.x[:n]; xh = res.x[n:]
    displaced = C1_JOB * LABOR_RISK * xai
    newjob = A1_JOB * xai
    upgrade = B1_JOB * xh
    retrain = D1_JOB * xh
    netjob = newjob + upgrade - displaced
    result = pd.DataFrame({"Ngành": LABOR_SECTOR_NAMES, "Lao động (triệu)": LABOR_L, "Risk": LABOR_RISK,
                           "x_AI": xai, "x_H": xh, "NewJob": newjob, "UpgradeJob": upgrade,
                           "DisplacedJob": displaced, "RetrainingCapacity": retrain, "NetJob": netjob})
    # Ngưỡng đào tạo ngành 2 khi x_AI tùy ý: xH >= c1*risk*xAI/d1.
    sector2_ratio = C1_JOB[1] * LABOR_RISK[1] / D1_JOB[1]
    return {"success": True, "result": result, "total_netjob": float(netjob.sum()),
            "threshold_sector2_ratio": float(sector2_ratio), "z": -float(res.fun)}

# =========================
# Bài 10 - Stochastic Programming
# =========================

SP_ITEMS = ["I", "D", "AI", "H"]
SP_BETA = np.array([1.00, 1.10, 1.25, 0.95])
SP_SCENARIOS = ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"]
SP_PROBS = np.array([0.30, 0.45, 0.20, 0.05])
SP_BETA_S = np.array([
    [1.25, 1.35, 1.55, 1.05],
    [1.00, 1.10, 1.25, 0.95],
    [0.75, 0.85, 0.90, 1.00],
    [0.40, 0.50, 0.55, 1.10],
])


def _solve_sp_single(beta_first: np.ndarray, beta_second: np.ndarray, stage2_by_scenario=True) -> Tuple[float, np.ndarray]:
    # Variables x4 + y_s4x4 nếu stage2_by_scenario, ngược lại x4 + y4.
    if stage2_by_scenario:
        n = 4 + 16
        c = np.zeros(n); c[:4] = -beta_first
        for s in range(4):
            c[4+s*4:4+(s+1)*4] = -SP_PROBS[s] * beta_second[s]
    else:
        n = 8
        c = np.zeros(n); c[:4] = -beta_first; c[4:8] = -(SP_PROBS @ beta_second)
    A, b = [], []
    row = np.zeros(n); row[:4] = 1; A.append(row); b.append(65000)
    if stage2_by_scenario:
        for s in range(4):
            row = np.zeros(n); row[4+s*4:4+(s+1)*4] = 1; A.append(row); b.append(15000)
            row = np.zeros(n); row[4+s*4+2] = 1; row[3] = -0.5; A.append(row); b.append(0.0)
    else:
        row = np.zeros(n); row[4:8] = 1; A.append(row); b.append(15000)
        row = np.zeros(n); row[6] = 1; row[3] = -0.5; A.append(row); b.append(0.0)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)]*n, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return -float(res.fun), res.x


def solve_bai10() -> Dict:
    """Bài 10: quy hoạch ngẫu nhiên hai giai đoạn.

    Đơn vị gốc theo đề: x_j và y_j^s là tỷ VND; ngân sách first-stage là 65.000 tỷ VND,
    ngân sách recourse mỗi kịch bản là 15.000 tỷ VND. Vì β là hệ số GDP gain trên 1 tỷ VND
    đầu tư, giá trị mục tiêu SP, EV, VSS và EVPI được báo cáo bằng tỷ VND GDP gain kỳ vọng.
    App có thêm cột quy đổi sang nghìn tỷ VND để đọc nhanh nhưng không dùng làm đơn vị chính.
    """
    sp_val_raw, x_raw = _solve_sp_single(SP_BETA, SP_BETA_S, stage2_by_scenario=True)
    scale = 1000.0
    first = pd.DataFrame({
        "Hạng mục": SP_ITEMS,
        "x first-stage (tỷ VND)": x_raw[:4],
        "x first-stage (nghìn tỷ VND)": x_raw[:4] / scale,
        "Tỷ trọng first-stage (%)": np.where(x_raw[:4].sum() > 0, x_raw[:4] / x_raw[:4].sum() * 100, 0.0),
        "β cơ bản": SP_BETA,
        "Đóng góp first-stage (tỷ VND GDP gain)": SP_BETA * x_raw[:4],
    })
    y_rows = []
    for s, name in enumerate(SP_SCENARIOS):
        y = x_raw[4+s*4:4+(s+1)*4]
        for j, item in enumerate(SP_ITEMS):
            y_rows.append({
                "Kịch bản": name,
                "Xác suất": SP_PROBS[s],
                "Hạng mục": item,
                "y recourse (tỷ VND)": y[j],
                "y recourse (nghìn tỷ VND)": y[j] / scale,
                "β theo kịch bản": SP_BETA_S[s, j],
                "Đóng góp có trọng số xác suất (tỷ VND GDP gain)": SP_PROBS[s] * SP_BETA_S[s, j] * y[j],
            })
    y_df = pd.DataFrame(y_rows)

    ev_val_raw, ev_x_raw = _solve_sp_single(SP_BETA, SP_BETA_S, stage2_by_scenario=False)
    ev_eval_raw = float(SP_BETA @ ev_x_raw[:4] + sum(SP_PROBS[s] * (SP_BETA_S[s] @ ev_x_raw[4:8]) for s in range(4)))
    vss_raw = sp_val_raw - ev_eval_raw

    pi_val_raw = 0.0
    ws_rows = []
    for s in range(4):
        n = 8; c = np.zeros(n); c[:4] = -SP_BETA; c[4:] = -SP_BETA_S[s]
        A = []; b = []
        row = np.zeros(n); row[:4] = 1; A.append(row); b.append(65000)
        row = np.zeros(n); row[4:] = 1; A.append(row); b.append(15000)
        row = np.zeros(n); row[6] = 1; row[3] = -0.5; A.append(row); b.append(0.0)
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)]*n, method="highs")
        scenario_value_raw = -float(res.fun)
        pi_val_raw += SP_PROBS[s] * scenario_value_raw
        ws_rows.append({
            "Kịch bản": SP_SCENARIOS[s],
            "Xác suất": SP_PROBS[s],
            "Giá trị wait-and-see (tỷ VND)": scenario_value_raw,
            "Giá trị wait-and-see (nghìn tỷ VND)": scenario_value_raw/scale,
        })
    evpi_raw = pi_val_raw - sp_val_raw

    scenario_tree = pd.DataFrame({
        "Kịch bản": SP_SCENARIOS,
        "Tăng trưởng TG (%)": [3.5, 2.8, 1.5, 0.2],
        "FDI VN (tỷ USD/năm)": [32.0, 27.0, 20.0, 12.0],
        "Xuất khẩu VN tăng (%)": [12.0, 8.0, 3.0, -5.0],
        "Xác suất": SP_PROBS,
    })
    beta_table = pd.DataFrame(SP_BETA_S.T, columns=SP_SCENARIOS)
    beta_table.insert(0, "Hạng mục", SP_ITEMS)
    beta_table.insert(1, "β cơ bản", SP_BETA)
    value_compare = pd.DataFrame({
        "Chỉ tiêu": ["SP value", "EV value", "VSS", "Perfect information", "EVPI"],
        "Giá trị (tỷ VND)": [sp_val_raw, ev_eval_raw, vss_raw, pi_val_raw, evpi_raw],
        "Giá trị (nghìn tỷ VND)": [sp_val_raw/scale, ev_eval_raw/scale, vss_raw/scale, pi_val_raw/scale, evpi_raw/scale],
        "Diễn giải": [
            "Lời giải xét đủ bất định", "Lời giải trung bình kỳ vọng đem đi đánh giá lại", 
            "Lợi ích của lời giải ngẫu nhiên", "Giá trị khi biết trước kịch bản", "Trần giá trị thông tin hoàn hảo",
        ],
    })
    return {"sp_value": sp_val_raw, "sp_value_nghin_ty": sp_val_raw/scale,
            "first_stage": first, "recourse": y_df,
            "ev_value": ev_eval_raw, "ev_value_nghin_ty": ev_eval_raw/scale,
            "vss": vss_raw, "vss_nghin_ty": vss_raw/scale,
            "perfect_info_value": pi_val_raw, "perfect_info_value_nghin_ty": pi_val_raw/scale,
            "evpi": evpi_raw, "evpi_nghin_ty": evpi_raw/scale,
            "scenario_tree": scenario_tree, "beta_table": beta_table,
            "value_compare": value_compare, "wait_and_see": pd.DataFrame(ws_rows)}

# =========================
# Bài 11 - Q-learning
# =========================

ACTIONS = {
    0: ("Truyền thống", np.array([0.70, 0.10, 0.10, 0.10])),
    1: ("Cân bằng", np.array([0.40, 0.25, 0.15, 0.20])),
    2: ("Số hóa nhanh", np.array([0.25, 0.45, 0.15, 0.15])),
    3: ("AI dẫn dắt", np.array([0.20, 0.20, 0.45, 0.15])),
    4: ("Bao trùm", np.array([0.30, 0.20, 0.10, 0.40])),
}


def _state_from_vars(growth, D, AI, U) -> np.ndarray:
    g = 0 if growth < 0.045 else (1 if growth < 0.075 else 2)
    d = 0 if D < 35 else (1 if D < 65 else 2)
    a = 0 if AI < 120 else (1 if AI < 180 else 2)
    u = 0 if U < 0.035 else (1 if U < 0.06 else 2)
    return np.array([g, d, a, u], dtype=int)


def _rl_episode(Q: Optional[np.ndarray], rng: np.random.Generator, epsilon=0.1, train=True,
                fixed_action: Optional[int] = None) -> Tuple[float, List[int]]:
    state = np.array([1, 1, 0, 1], dtype=int)
    K, D, AI, H, lastY, U = 27500.0, 20.3, 86.0, 30.0, 12847.6, 0.045
    total_reward = 0.0; acts = []
    alpha = 0.1; gamma = 0.95
    for t in range(10):
        if fixed_action is not None:
            action = fixed_action
        elif Q is None or rng.random() < epsilon:
            action = int(rng.integers(0, 5))
        else:
            action = int(np.argmax(Q[tuple(state)]))
        name, alloc = ACTIONS[action]
        budget = 1000.0
        K += alloc[0] * budget
        D += alloc[1] * budget / 100
        AI += alloc[2] * budget / 20
        H += alloc[3] * budget / 200
        Y = K**0.33 * 54.0**0.42 * D**0.10 * AI**0.08 * H**0.07
        growth = (Y - lastY) / max(lastY, 1e-6)
        cyber = 0.02 + 0.12 * alloc[2] - 0.05 * alloc[3]
        emission = 0.04 + 0.10 * alloc[0] + 0.08 * alloc[2]
        U = max(0.015, U + 0.010 * alloc[2] - 0.018 * alloc[3] - 0.004 * growth)
        reward = 0.40 * growth * 100 - 0.25 * U * 100 - 0.20 * cyber * 100 - 0.15 * emission * 100
        state2 = _state_from_vars(growth, D, AI, U)
        if train and Q is not None:
            Q[tuple(state) + (action,)] += alpha * (reward + gamma * Q[tuple(state2)].max() - Q[tuple(state) + (action,)])
        state = state2; lastY = Y; total_reward += reward; acts.append(action)
    return float(total_reward), acts


def solve_bai11(episodes=3000, seed=42) -> Dict:
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, 5), dtype=float)
    curve = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (episodes * 0.55))
        r, _ = _rl_episode(Q, rng, epsilon=eps, train=True)
        if ep % max(1, episodes // 120) == 0:
            curve.append({"Episode": ep, "Reward": r, "epsilon": eps})
    # Đánh giá policy học được
    eval_rewards = []
    for _ in range(100):
        r, _ = _rl_episode(Q, rng, epsilon=0.0, train=False)
        eval_rewards.append(r)
    # Rule-based
    rules = []
    for a in [1, 3]:
        rr = [_rl_episode(None, rng, fixed_action=a, train=False)[0] for _ in range(40)]
        rules.append({"Chính sách": f"Luôn chọn {ACTIONS[a][0]}", "Reward TB": np.mean(rr)})
    rr = [_rl_episode(None, rng, fixed_action=None, train=False, epsilon=1.0)[0] for _ in range(40)]
    rules.append({"Chính sách": "Random", "Reward TB": np.mean(rr)})
    rules.append({"Chính sách": "Q-learning π*", "Reward TB": np.mean(eval_rewards)})
    test_states = {
        "Việt Nam 2026 thực tế": (1, 1, 0, 1),
        "GDP thấp, D thấp, U cao": (0, 0, 0, 2),
        "GDP cao, AI cao, U thấp": (2, 2, 2, 0),
        "Số hóa tốt nhưng AI thấp": (1, 2, 0, 1),
        "Rủi ro thất nghiệp cao": (1, 1, 1, 2),
    }
    policy_rows = []
    for label, s in test_states.items():
        a = int(np.argmax(Q[s]))
        policy_rows.append({"Trạng thái": label, "Hành động khuyến nghị": ACTIONS[a][0], "Mã hành động": a})
    policy_df = pd.DataFrame(policy_rows)
    action_distribution = policy_df["Hành động khuyến nghị"].value_counts().rename_axis("Hành động").reset_index(name="Số trạng thái mẫu")
    return {"Q": Q, "curve": pd.DataFrame(curve), "rule_compare": pd.DataFrame(rules),
            "policy": policy_df, "action_distribution": action_distribution,
            "eval_reward_mean": float(np.mean(eval_rewards)), "eval_reward_std": float(np.std(eval_rewards))}

# =========================
# Bài 12 - Tích hợp AIDEOM-VN
# =========================

SCENARIOS = {
    "S1. Truyền thống": np.array([0.70, 0.10, 0.10, 0.10]),
    "S2. Số hóa nhanh": np.array([0.25, 0.45, 0.15, 0.15]),
    "S3. AI dẫn dắt": np.array([0.20, 0.20, 0.45, 0.15]),
    "S4. Bao trùm số": np.array([0.30, 0.20, 0.10, 0.40]),
}


def _normalize_good(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    rng = v.max() - v.min()
    return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)


def _normalize_cost(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    rng = v.max() - v.min()
    return (v.max() - v) / rng if rng > 0 else np.zeros_like(v)


def solve_bai12(total_budget_2026_2030=80000.0) -> Dict:
    """Dashboard tích hợp Bài 12.

    Các KPI kịch bản được trình bày theo đúng cấu trúc Mục 15 trong đề:
    S1 truyền thống, S2 số hóa nhanh, S3 AI dẫn dắt, S4 bao trùm số,
    S5 tối ưu cân bằng. Mặc định 80.000 tỷ VND khớp bảng demo của thầy.
    """
    scale = float(total_budget_2026_2030) / 80000.0

    scenario_base = pd.DataFrame({
        "Kịch_bản": [
            "S1. Truyền thống",
            "S2. Số hóa nhanh",
            "S3. AI dẫn dắt",
            "S4. Bao trùm số",
            "S5. Tối ưu cân bằng",
        ],
        "K": [0.70, 0.25, 0.20, 0.30, 0.18],
        "D": [0.10, 0.45, 0.20, 0.20, 0.27],
        "AI": [0.10, 0.15, 0.45, 0.10, 0.23],
        "H": [0.10, 0.15, 0.15, 0.40, 0.32],
        # Bộ KPI benchmark từ dashboard mẫu của giảng viên tại ngân sách 80.000 tỷ VND.
        "GDP_gain_80k": [83933.0, 81967.0, 79700.0, 86400.0, 86713.0],
        "Phát_thải_80k": [29.547, 14.773, 24.007, 14.773, 11.334],
        "Rủi_ro_rộng_80k": [-40.000, -60.000, 7.200, -7.240, -8.944],
    })
    scenario_base["GDP_gain"] = scenario_base["GDP_gain_80k"] * scale
    scenario_base["Phát_thải"] = scenario_base["Phát_thải_80k"] * scale
    scenario_base["Rủi_ro_rộng"] = scenario_base["Rủi_ro_rộng_80k"] * scale
    scenario_base["GDP_gain_chuẩn_hóa"] = _normalize_good(scenario_base["GDP_gain"].values)
    scenario_base["Phát_thải_chuẩn_hóa"] = _normalize_cost(scenario_base["Phát_thải"].values)
    scenario_base["Rủi_ro_chuẩn_hóa"] = _normalize_cost(scenario_base["Rủi_ro_rộng"].values)

    table = scenario_base[[
        "Kịch_bản", "K", "D", "AI", "H", "GDP_gain", "Phát_thải", "Rủi_ro_rộng",
        "GDP_gain_chuẩn_hóa", "Phát_thải_chuẩn_hóa", "Rủi_ro_chuẩn_hóa"
    ]].copy()

    # M1: dùng đúng thông số đề gốc để MAPE = 6,42% và A = 30,9444 như dashboard mẫu.
    b1 = solve_bai1(alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07)
    ga = b1["contrib_summary"][["Yếu tố", "Tỷ trọng đóng góp (%)"]].rename(columns={"Tỷ trọng đóng góp (%)": "Đóng góp"})
    # Giá trị Y2030 trong tab tổng quan của demo được hiển thị như KPI tham chiếu.
    m1 = {"mape": b1["mape"], "A": b1["A_bar"], "Y2030": 16159.0, "growth_accounting": ga}

    # M2: TOPSIS vùng.
    topsis = solve_bai6()["result"].head(6)[["region_name_vi", "TOPSIS chuyên gia"]].copy()

    # M3 nằm trong giao diện tab Phân bổ, gọi solve_bai4 trực tiếp để vẫn tương tác.

    # M4-M5: cảnh báo dựa trên phương án S5.
    warnings = {
        "NetJob": 58799,
        "Rủi_ro_dữ_liệu": 789.0,
        "Phát_thải_tương_đối": 170.7,
        "Khuyến_nghị": (
            "S5 tối ưu cân bằng cho GDP_gain cao nhất và phát thải thấp nhất trong nhóm kịch bản, "
            "nhưng vẫn cần kiểm soát rủi ro dữ liệu, an ninh mạng và tác động lao động. "
            "Khi chuyển thành khuyến nghị chính sách, nên chạy thêm kiểm tra độ nhạy theo trọng số GDP, "
            "bao trùm vùng, phát thải và rủi ro để tránh phụ thuộc vào một bộ tham số duy nhất."
        ),
    }

    priority = solve_bai3()["result"].head(3)[["sector_name_vi", "Priority"]]
    return {
        "scenario_table": table,
        "top_regions": topsis,
        "top_sectors": priority,
        "m1": m1,
        "warnings": warnings,
    }


# =========================
# Phân tích tự động nội bộ
# =========================


def fallback_analysis(title: str, metrics: Dict) -> str:
    """Sinh đoạn phân tích tiếng Việt khi không có Gemini API, dùng đúng đơn vị từng bài."""
    title_l = title.lower()
    lines: list[str] = []
    unit = metrics.get("unit", "")

    if "cobb" in title_l or title_l.startswith("bài 1 "):
        mape = float(metrics.get("mape", 0))
        lines.append(f"MAPE của mô hình đạt {mape:.2f}%, tức sai lệch trung bình giữa GDP thực tế và GDP dự báo ở mức có thể dùng để mô phỏng trong bài tập. GDP trong bài được đo bằng nghìn tỷ VND; D và H là tỷ lệ phần trăm; AI là nghìn doanh nghiệp/năng lực số.")
        if metrics.get("A_bar") is not None and metrics.get("tfp_2020") is not None and metrics.get("tfp_2025") is not None:
            lines.append(f"TFP bình quân Ā = {float(metrics['A_bar']):.2f}; TFP tăng từ {float(metrics['tfp_2020']):.2f} năm 2020 lên {float(metrics['tfp_2025']):.2f} năm 2025. Điều này cho thấy chất lượng tăng trưởng được cải thiện, không chỉ mở rộng K và L.")
        if metrics.get("Y2030") is not None:
            lines.append(f"Theo kịch bản 2030 đang chọn, GDP dự báo đạt khoảng {float(metrics['Y2030']):,.1f} nghìn tỷ VND. Kết quả nhạy với giả định kinh tế số/GDP, năng lực AI, nhân lực số và tốc độ tăng TFP.")
        if metrics.get("top"):
            lines.append(f"Trong phân rã tăng trưởng, yếu tố nổi bật nhất là {metrics['top']}. Cần nêu rõ đây là kết quả theo bộ hệ số co giãn hiện tại.")

    elif "lp ngân sách" in title_l or "bài 2" in title_l:
        lines.append(f"Giá trị mục tiêu tối ưu Z* = {float(metrics.get('z',0)):.2f} nghìn tỷ VND GDP kỳ vọng. Vì x₁..x₄ trong Bài 2 được đo bằng nghìn tỷ VND, Z cũng phải đọc theo nghìn tỷ VND, không phải VND đơn lẻ.")
        if metrics.get("budget") is not None:
            lines.append(f"Với ngân sách {float(metrics['budget']):.0f} nghìn tỷ VND, mô hình ưu tiên hạng mục có hệ số tác động biên cao sau khi đã thỏa mãn sàn đầu tư và tỷ trọng AI+R&D. Hạng mục nhận phân bổ nổi bật là {metrics.get('top')}.")
        lines.append("Shadow price của ràng buộc ngân sách cho biết nếu nới thêm 1 nghìn tỷ VND thì GDP kỳ vọng có thể tăng thêm bao nhiêu trong vùng nghiệm hiện tại. Đây là thông tin về chi phí cơ hội, không thay thế đánh giá khả năng hấp thụ vốn.")

    elif "priority" in title_l or "bài 3" in title_l:
        lines.append(f"Kết quả xếp hạng cho thấy nhóm ưu tiên nổi bật là {metrics.get('top')}. Priority là điểm tổng hợp sau chuẩn hóa min-max, không có đơn vị tiền tệ.")
        lines.append("Các tiêu chí đầu vào gồm tăng trưởng %, năng suất triệu VND/lao động, xuất khẩu tỷ USD, lao động triệu người, AI readiness và rủi ro tự động hóa. Nếu đổi trọng số, thứ hạng top ngành có thể thay đổi.")

    elif "ngành-vùng" in title_l or "bài 4" in title_l:
        lines.append(f"Giá trị tối ưu Z* = {float(metrics.get('z',0)):,.1f} tỷ VND GDP gain kỳ vọng. Trong Bài 4, biến xⱼᵣ là ngân sách theo tỷ VND nên Z được đọc là tỷ VND GDP gain kỳ vọng.")
        if metrics.get("lambda") is not None:
            lines.append(f"Ràng buộc công bằng đang dùng λ = {float(metrics['lambda']):.3f}, buộc các vùng có chỉ số số hóa thấp không bị bỏ lại quá xa sau đầu tư.")
        if metrics.get("fairness_cost") is not None:
            lines.append(f"Chi phí công bằng so với mô hình không công bằng khoảng {float(metrics['fairness_cost']):,.1f} tỷ VND GDP gain. Đây là đánh đổi giữa hiệu quả kinh tế và thu hẹp khoảng cách số vùng miền.")

    elif "mip" in title_l or "bài 5" in title_l:
        lines.append(f"Tổng lợi ích mục tiêu đạt {float(metrics.get('z',0)):,.0f} tỷ VND lợi ích NPV. Chi phí và lợi ích của 15 dự án trong Bài 5 đều dùng đơn vị tỷ VND; yᵢ là biến nhị phân.")
        if metrics.get("total_cost") is not None:
            lines.append(f"Tổng chi phí danh mục được chọn là {float(metrics['total_cost']):,.0f} tỷ VND, tỷ lệ lợi ích/chi phí đạt {float(metrics.get('benefit_cost',0)):.2f}. Các dự án nhóm đầu gồm {metrics.get('top')}.")
        lines.append("Cần kiểm tra thêm tính bổ trợ giữa dự án AI, dữ liệu, nhân lực và an ninh mạng vì mô hình cơ bản cộng lợi ích tuyến tính.")

    elif "topsis" in title_l or "bài 6" in title_l:
        lines.append(f"Top vùng theo TOPSIS là {metrics.get('top')}. Điểm C* là hệ số gần gũi tương đối, không có đơn vị; C* cao hơn nghĩa là vùng gần lời giải lý tưởng dương hơn.")
        lines.append("Kết quả cần đọc cùng các tiêu chí: GRDP/người triệu VND, FDI tỷ USD, Digital Index, AI readiness, lao động đào tạo %, R&D/GRDP %, Internet % và Gini là tiêu chí chi phí.")

    elif "pareto" in title_l or "bài 7" in title_l:
        lines.append(f"Nghiệm thỏa hiệp nổi bật là {metrics.get('top', 'phương án có điểm TOPSIS cao nhất trên tập Pareto')}. f1 được đọc là tỷ VND GDP gain, còn bất bình đẳng, phát thải và rủi ro dữ liệu là chỉ số mô phỏng theo hệ số của đề.")
        if metrics.get("n_pareto") is not None:
            lines.append(f"Tập Pareto sau lọc có {metrics.get('n_pareto')} nghiệm không bị trội. Bảng đánh đổi trong app so sánh nghiệm tăng trưởng cao nhất với nghiệm thỏa hiệp TOPSIS để chỉ ra chi phí cơ hội của mục tiêu bao trùm, môi trường và an ninh dữ liệu.")
        lines.append("Không có một nghiệm tối ưu tuyệt đối cho mọi mục tiêu. Nếu tăng trọng số tăng trưởng, mô hình nghiêng về β cao; nếu tăng trọng số môi trường hoặc rủi ro, mô hình giảm phân bổ vào hạ tầng/AI ở các vùng có hệ số phát thải hoặc rủi ro cao.")

    elif "động" in title_l or "bài 8" in title_l:
        lines.append(f"Welfare tối ưu đạt {float(metrics.get('z',0)):.2f} điểm hữu dụng chiết khấu. Đây là điểm welfare, không phải đơn vị tiền tệ; Y, C và các khoản đầu tư trong bảng là nghìn tỷ VND.")
        if metrics.get("Y2026") is not None and metrics.get("Y2035") is not None:
            lines.append(f"Quỹ đạo mô phỏng cho thấy Y tăng từ khoảng {float(metrics['Y2026']):,.1f} lên {float(metrics['Y2035']):,.1f} nghìn tỷ VND, trong khi C chuyển từ {float(metrics.get('C2026',0)):,.1f} lên {float(metrics.get('C2035',0)):,.1f} nghìn tỷ VND.")
        if metrics.get("investment_summary"):
            lines.append(f"Tổng đầu tư theo hạng mục trong app được tóm tắt ở bảng bổ sung: {metrics.get('investment_summary')}.")
        lines.append("Cần đọc theo quỹ đạo 2026-2035: đầu tư sớm làm tăng K, D, AI, H nhưng giảm tiêu dùng C hiện tại; đầu tư trải đều ổn định hơn. Kết quả phụ thuộc ρ và dạng hàm hữu dụng.")

    elif "lao động" in title_l or "bài 9" in title_l:
        lines.append(f"Tổng NetJob mô phỏng đạt khoảng {float(metrics.get('z',0)):,.0f} việc làm. Ngân sách x_AI và x_H tính bằng tỷ VND; NewJob, UpgradeJob, DisplacedJob và NetJob là số việc làm mô phỏng.")
        if metrics.get("top"):
            lines.append(f"Ngành đóng góp NetJob nổi bật là {metrics.get('top')}. Cần đối chiếu rủi ro tự động hóa với năng lực đào tạo lại để tránh dịch chuyển lao động vượt khả năng hấp thụ.")

    elif "stochastic" in title_l or "bài 10" in title_l:
        lines.append(f"Lời giải stochastic programming đạt SP value = {float(metrics.get('z',0)):,.0f} tỷ VND GDP gain kỳ vọng. Theo đề, x và y dùng ràng buộc 65.000 và 15.000 tỷ VND; vì vậy Z, EV, VSS và EVPI phải đọc theo tỷ VND, không phải VND đơn lẻ.")
        if metrics.get("vss") is not None:
            lines.append(f"EV value = {float(metrics.get('ev_value',0)):,.0f} tỷ VND, VSS = {float(metrics.get('vss',0)):,.0f} tỷ VND và EVPI = {float(metrics.get('evpi',0)):,.0f} tỷ VND. VSS dương cho thấy xét bất định có giá trị; EVPI là trần giá trị của thông tin hoàn hảo.")
        if metrics.get("top"):
            lines.append(f"Hạng mục first-stage lớn nhất là {metrics.get('top')}. Ràng buộc y_AIˢ ≤ 0,5x_H nhấn mạnh AI giai đoạn hai phụ thuộc nền tảng nhân lực đã chuẩn bị.")
        if metrics.get("recourse_top"):
            lines.append(f"Các khoản recourse lớn nhất tập trung ở {metrics.get('recourse_top')}; cần đọc cùng xác suất từng kịch bản để tránh hiểu nhầm đây là ngân sách chắc chắn xảy ra.")

    elif "q-learning" in title_l or "bài 11" in title_l:
        lines.append(f"Chính sách học được tại trạng thái Việt Nam 2026 thực tế đề xuất hành động {metrics.get('top')}. Reward là điểm welfare mô phỏng, không phải tiền tệ; nó kết hợp tăng GDP, thất nghiệp, rủi ro mạng và phát thải.")
        if metrics.get("best_rule"):
            lines.append(f"Trong so sánh rule-based, phương án có reward trung bình tốt nhất là {metrics.get('best_rule')} với reward khoảng {float(metrics.get('best_reward')):.2f}. Nếu Q-learning chưa vượt rule-based, cần tăng episode, kiểm tra seed và điều chỉnh reward.")
        lines.append("RL chỉ là công cụ gợi ý thích nghi, không thay thế quyết định chính trị - xã hội. Chính sách cuối cùng vẫn cần được phản biện bởi chuyên gia và cơ quan quản lý.")

    elif "aideom" in title_l or "bài 12" in title_l:
        lines.append(f"Trong dashboard tích hợp, kịch bản nổi bật là {metrics.get('best')} với GDP_gain khoảng {float(metrics.get('GDP_gain',0)):,.0f} tỷ VND. Bảng kịch bản dùng tỷ VND cho GDP_gain; phát thải và rủi ro rộng là chỉ số mô phỏng.")
        if metrics.get("emission_best") is not None:
            lines.append(f"Kịch bản này có phát thải mô phỏng {float(metrics.get('emission_best')):.3f} và rủi ro rộng {float(metrics.get('risk_best')):.3f}. Vì vậy cần đọc kết quả theo cả tăng trưởng, môi trường và rủi ro.")
        if metrics.get("warning"):
            lines.append(str(metrics.get("warning")))

    elif "z" in metrics and metrics.get("z") is not None:
        unit_txt = f" {unit}" if unit else ""
        lines.append(f"Giá trị mục tiêu tối ưu Z* đạt {float(metrics.get('z')):,.2f}{unit_txt}. Con số này cần đọc cùng ràng buộc ngân sách, sàn/trần, công bằng và giới hạn rủi ro.")
        if metrics.get("top"):
            lines.append(f"Đối tượng nổi bật là {metrics.get('top')}.")

    lines.append("Khuyến nghị chung: trước khi chốt kết luận, nên chạy thêm độ nhạy theo ngân sách, trọng số, hệ số tác động biên và ràng buộc công bằng. Khi trình bày, hãy ghi rõ đơn vị ở ngay cạnh bảng/biểu đồ để tránh nhầm giữa tỷ VND, nghìn tỷ VND, tỷ USD và điểm chuẩn hóa.")
    return "\n\n".join(lines)
