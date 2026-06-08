APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400;600;700&display=swap');
:root{
    --bg:#f4f9fb;
    --card:#ffffff;
    --ink:#17324d;
    --muted:#5b6c7d;
    --teal:#1aa6a6;
    --blue:#2b6cb0;
    --soft:#e9f7f7;
    --line:#dce8ef;
}
html, body, [class*="css"]  {
    font-family: "Times New Roman", "Noto Serif", Georgia, serif;
}
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef8f8 58%, #f7fbff 100%);
    color: var(--ink);
}
.block-container {
    padding-top: 2rem;
    max-width: 1320px;
}
section[data-testid="stSidebar"] {
    background: #eef6f8;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * {
    color: #17324d;
}
h1, h2, h3 {
    color: #17324d;
    letter-spacing: .2px;
}
h1 {
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    margin-bottom: .2rem !important;
}
h2 {
    font-size: 1.55rem !important;
    border-left: 6px solid var(--teal);
    padding-left: 12px;
    margin-top: 1.2rem;
}
.hero {
    border-radius: 24px;
    padding: 30px 34px;
    background: linear-gradient(135deg, #ffffff 0%, #eaf8fb 100%);
    border: 1px solid #d7edf2;
    box-shadow: 0 10px 32px rgba(23,50,77,.08);
    margin-bottom: 18px;
}
.hero .subtitle {font-size: 1.06rem; color: var(--muted); margin-top: 8px;}
.badge {
    display: inline-block;
    padding: 5px 12px;
    background: #dff5f2;
    border: 1px solid #bce7e3;
    color: #0b6868;
    border-radius: 999px;
    font-size: .88rem;
    font-weight: 700;
    margin-right: 6px;
}
.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 8px 24px rgba(23,50,77,.06);
    margin: 12px 0;
}
.kpi-row {display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin: 14px 0 22px 0;}
.kpi {
    background: #ffffff;
    border: 1px solid #dce8ef;
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 8px 20px rgba(23,50,77,.05);
}
.kpi small {color: #6b7c8f; display:block; font-size:.86rem;}
.kpi b {font-size:1.75rem; color:#17324d; white-space:nowrap;}
.ai-box {
    background: #f8fcff;
    border: 1px solid #cfe8f3;
    border-left: 6px solid #2b6cb0;
    border-radius: 16px;
    padding: 14px 18px;
    margin: 16px 0;
}
.formula {
    background: #f7fbfd;
    border: 1px solid #dce8ef;
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 1.02rem;
    color: #23384e;
}
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 12px 14px;
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(23,50,77,.05);
}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    background: #ffffff;
    border: 1px solid #dce8ef;
    border-radius: 999px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #dff5f2 !important;
    color: #0b6868 !important;
    font-weight: 700;
}
hr {border: none; border-top: 1px solid var(--line); margin: 22px 0;}
</style>
"""
