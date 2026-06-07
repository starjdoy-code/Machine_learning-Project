import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import date

st.set_page_config(page_title="Rossman Forecast", layout="wide")

# ── Minimal CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');
*, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #f7f6f3; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e8e4de; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #1c1c1a !important; }
[data-testid="stForm"] { border:none !important; background:transparent !important; box-shadow:none !important; padding:0 !important; }
div[data-testid="stFormSubmitButton"] button {
    width:100%; background:#1c1c1a; color:#ffffff !important; border:none; border-radius:4px;
    font-size:.85rem; font-weight:500; letter-spacing:.06em; text-transform:uppercase; padding:14px;
}
div[data-testid="stFormSubmitButton"] button:hover { background:#3a3a36; color:#ffffff !important; }
div[data-testid="stFormSubmitButton"] button p { color:#ffffff !important; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e8e4de; border-radius:6px; padding:1.1rem 1.3rem; }
[data-testid="stMetricLabel"] { font-size:.72rem !important; text-transform:uppercase !important; letter-spacing:.08em !important; color:#999 !important; }
[data-testid="stMetricValue"] { font-family:'DM Mono',monospace !important; font-size:1.5rem !important; color:#1c1c1a !important; }
[data-testid="stMetricDelta"] { font-size:.72rem !important; color:#999 !important; }
hr { border:none; border-top:1px solid #e8e4de; margin:1.2rem 0; }
h1 { font-size:1.5rem !important; font-weight:600 !important; color:#1c1c1a !important; }
h2 { font-size:1rem !important; font-weight:500 !important; color:#3a3a36 !important; }
.pill { display:inline-block; background:#fff; border:1px solid #e8e4de; border-radius:4px; padding:3px 9px; font-size:.76rem; color:#3a3a36; margin:2px; }
.banner { padding:11px 15px; border-radius:0 4px 4px 0; font-size:.84rem; margin-bottom:4px; }
.high { background:#f0f5f0; border-left:3px solid #5a8a5a; color:#2d4a2d; }
.mid  { background:#f3f5f8; border-left:3px solid #5a78a0; color:#2d3d52; }
.low  { background:#fdf5ec; border-left:3px solid #c4894a; color:#5a3a1a; }
.trow { display:flex; align-items:center; gap:8px; padding:7px 0; border-bottom:1px solid #f0ede8; font-size:.81rem; color:#3a3a36; }
.dp { width:7px; height:7px; border-radius:50%; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ── Load model once, never reload ─────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    m  = joblib.load('model_rossman.pkl')
    f  = joblib.load('model_features.pkl')
    ev = joblib.load('eval_data.pkl')
    fi = joblib.load('feature_importance.pkl')
    return m, f, ev, fi

model, FEATURES, ev, fi = load_model()

# ── Pre-build static charts once ──────────────────────────────────
@st.cache_data(show_spinner=False)
def static_charts():
    # Scatter
    fig_s = go.Figure()
    fig_s.add_trace(go.Scattergl(          # Scattergl = WebGL, much faster
        x=ev['y_aktual'], y=ev['y_prediksi'],
        mode='markers', name='Stores',
        marker=dict(color='#1c1c1a', size=3, opacity=0.3)
    ))
    lo = float(min(ev['y_aktual'].min(), ev['y_prediksi'].min()))
    hi = float(max(ev['y_aktual'].max(), ev['y_prediksi'].max()))
    fig_s.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode='lines', name='Perfect',
                               line=dict(color='#c4894a', dash='dash', width=1.5)))
    fig_s.update_layout(paper_bgcolor='#fff', plot_bgcolor='#fff',
        xaxis=dict(title='Actual (€)', gridcolor='#f0ede8', tickfont=dict(family='DM Mono', size=11)),
        yaxis=dict(title='Predicted (€)', gridcolor='#f0ede8', tickfont=dict(family='DM Mono', size=11)),
        title=dict(text=f"R² = {ev['r2']:.4f}", font=dict(size=12, color='#999')),
        legend=dict(bgcolor='#fff', bordercolor='#e8e4de', borderwidth=1),
        height=420, margin=dict(t=45,b=40,l=55,r=15), font=dict(family='DM Sans'))

    # Feature importance
    fig_f = None
    if fi:
        df_fi = pd.DataFrame({'F': list(fi.keys()), 'I': list(fi.values())}).sort_values('I')
        fig_f = go.Figure(go.Bar(x=df_fi['I'], y=df_fi['F'], orientation='h',
            marker=dict(color=df_fi['I'], colorscale=[[0,'#e8e4de'],[1,'#1c1c1a']], showscale=False)))
        fig_f.update_layout(paper_bgcolor='#fff', plot_bgcolor='#fff',
            xaxis=dict(title='Importance', gridcolor='#f0ede8', tickfont=dict(family='DM Mono', size=11)),
            yaxis=dict(tickfont=dict(family='DM Sans', size=11)),
            height=460, margin=dict(t=15,b=40,l=155,r=15), font=dict(family='DM Sans'))
    return fig_s, fig_f

fig_scatter, fig_fi_chart = static_charts()

# ── Prediction function — pure numpy, no Python loop ──────────────
@st.cache_data(show_spinner=False)
def predict(_model, _features, store_id, day_of_week, promo, state_holiday,
            school_holiday, store_type, assortment, comp_dist,
            comp_month, comp_year, promo2, p2_week, p2_year,
            promo_interval, year, month, day, week):
    row = pd.DataFrame([{
        'Store': store_id, 'DayOfWeek': day_of_week, 'Promo': promo,
        'StateHoliday': state_holiday, 'SchoolHoliday': school_holiday,
        'StoreType': store_type, 'Assortment': assortment,
        'CompetitionDistance': comp_dist,
        'CompetitionOpenSinceMonth': comp_month, 'CompetitionOpenSinceYear': comp_year,
        'Promo2': promo2, 'Promo2SinceWeek': p2_week, 'Promo2SinceYear': p2_year,
        'PromoInterval': promo_interval,
        'Year': year, 'Month': month, 'Day': day, 'WeekOfYear': week
    }])[_features]

    pred = np.expm1(_model.predict(row)[0])

    # Confidence via numpy stack — avoids Python-level loop overhead
    try:
        all_preds = np.expm1(
            np.column_stack([t.predict(row.values) for t in _model.estimators_]).ravel()
        )
        std = np.std(all_preds)
        conf = float(np.clip(100.0 - (std / pred * 100), 40, 99.9)) if pred > 0 else 60.0
        lo   = float(max(0, pred - 1.96 * std))
        hi   = float(pred + 1.96 * std)
    except Exception:
        conf = ev['r2'] * 100
        lo   = pred - ev['mae']
        hi   = pred + ev['mae']

    return float(pred), conf, lo, hi

# ── Sidebar form ───────────────────────────────────────────────────
with st.sidebar.form("inputs"):
    st.markdown("**Store**")
    store_id = st.number_input("Store ID", 1, 1115, 1)
    st_type  = st.selectbox("Store Type",
        [("a — Standard",0),("b — Large Format",1),("c — Specialty",2),("d — Extra Large",3)],
        format_func=lambda x: x[0])
    asst = st.selectbox("Assortment",
        [("a — Basic",0),("b — Extra",1),("c — Extended",2)],
        format_func=lambda x: x[0])

    st.markdown("**Date**")
    sel_date = st.date_input("Transaction Date", value=date.today())

    st.markdown("**Promotions**")
    promo  = st.radio("Active Promotion",  [("Yes",1),("No",0)], format_func=lambda x:x[0], horizontal=True)
    promo2 = st.radio("Promo2",            [("Yes",1),("No",0)], format_func=lambda x:x[0], horizontal=True)
    p_int  = st.selectbox("Promo2 Interval",
        [("None",0),("Jan/Apr/Jul/Oct",1),("Feb/May/Aug/Nov",2),("Mar/Jun/Sep/Dec",3)],
        format_func=lambda x: x[0])
    p2w = st.number_input("Promo2 Start Week",  0, 52,   0)
    p2y = st.number_input("Promo2 Start Year",  0, 2025, 0)

    st.markdown("**Holidays**")
    s_hol  = st.selectbox("State Holiday",
        [("None",0),("Public Holiday",1),("Easter",2),("Christmas",3)],
        format_func=lambda x: x[0])
    sc_hol = st.radio("School Holiday", [("Yes",1),("No",0)], format_func=lambda x:x[0], horizontal=True)

    st.markdown("**Competitor**")
    comp_dist  = st.number_input("Distance to Competitor (m)", 0, 100000, 1000, 100)
    comp_month = st.selectbox("Competitor Open Month", list(range(0,13)), index=0)
    comp_year  = st.number_input("Competitor Open Year", 0, 2025, 0)

    run = st.form_submit_button("Run Prediction", use_container_width=True)

# ── On submit: compute and store in session_state ─────────────────
if run:
    dow  = sel_date.weekday() + 1
    woy  = sel_date.isocalendar()[1]
    pred, conf, lo, hi = predict(
        model, FEATURES,
        store_id, dow, promo[1], s_hol[1], sc_hol[1],
        st_type[1], asst[1], comp_dist, comp_month, comp_year,
        promo2[1], p2w, p2y, p_int[1],
        sel_date.year, sel_date.month, sel_date.day, woy
    )
    st.session_state.res = dict(
        pred=pred, conf=conf, lo=lo, hi=hi,
        store_id=store_id, date=sel_date, dow=dow,
        promo=promo, st_type=st_type, asst=asst,
        s_hol=s_hol, comp_dist=comp_dist
    )

# ── Main display ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.title("Rossman Sales Forecast")

if 'res' not in st.session_state:
    st.markdown('<p style="color:#999;font-size:.9rem;">Set parameters and click <b>Run Prediction</b>.</p>', unsafe_allow_html=True)
else:
    r = st.session_state.res
    st.markdown(f'<p style="color:#999;font-size:.88rem;margin-top:-6px;">Store #{r["store_id"]} · {r["date"].strftime("%d %b %Y")}</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Pills
    st.markdown(
        f'<span class="pill">{r["st_type"][0].split(" — ")[0]}</span>'
        f'<span class="pill">Assortment {r["asst"][0].split(" — ")[0]}</span>'
        f'<span class="pill">Day {r["dow"]}</span>'
        f'<span class="pill">Promo {"On" if r["promo"][1] else "Off"}</span>'
        f'<span class="pill">{"Holiday" if r["s_hol"][1] else "No Holiday"}</span>'
        f'<span class="pill">{r["comp_dist"]:,} m to competitor</span>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Predicted Sales",   f"€{r['pred']:,.0f}", ev['model_name'])
    c2.metric("Confidence",        f"{r['conf']:.1f}%")
    c3.metric("Lower Bound (95%)", f"€{r['lo']:,.0f}")
    c4.metric("Upper Bound (95%)", f"€{r['hi']:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("### Assessment")
        if r['pred'] >= 10000:
            st.markdown('<div class="banner high"><b>High Sales Day</b><br>Strong forecast. Ensure adequate stock and staffing.</div>', unsafe_allow_html=True)
        elif r['pred'] >= 5000:
            st.markdown('<div class="banner mid"><b>Moderate Sales Day</b><br>Normal performance. Consider activating promotions.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner low"><b>Low Sales Day</b><br>Below average. Review promotions and staffing.</div>', unsafe_allow_html=True)

    with a2:
        st.markdown("### Key Drivers")
        if fi:
            max_score = max(fi.values())
            for feat in list(fi.keys())[:4]:
                score = fi[feat]
                w = int(score / max_score * 100)
                st.markdown(
                    f'<div style="margin-bottom:9px">'
                    f'<div style="display:flex;justify-content:space-between;font-size:.79rem;margin-bottom:3px">'
                    f'<span style="color:#3a3a36;font-weight:500">{feat}</span>'
                    f'<span style="color:#999;font-family:DM Mono,monospace">{score:.4f}</span></div>'
                    f'<div style="background:#e8e4de;border-radius:2px;height:3px">'
                    f'<div style="background:#1c1c1a;width:{w}%;height:3px;border-radius:2px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

    # Gauge
    st.markdown("<br>", unsafe_allow_html=True)
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=r['pred'],
        number={'prefix':'€','font':{'family':'DM Mono','size':26,'color':'#1c1c1a'}},
        title={'text':'Predicted Daily Sales','font':{'family':'DM Sans','size':13,'color':'#999'}},
        gauge={
            'axis':{'range':[0,30000],'tickfont':{'family':'DM Mono','size':10,'color':'#aaa'},'tickcolor':'#e8e4de'},
            'bar':{'color':'#1c1c1a','thickness':0.22},
            'bgcolor':'#f7f6f3','bordercolor':'#e8e4de',
            'steps':[{'range':[0,5000],'color':'#f5f2ee'},{'range':[5000,10000],'color':'#edeae4'},
                     {'range':[10000,20000],'color':'#e3e0d8'},{'range':[20000,30000],'color':'#d8d4ca'}],
            'threshold':{'line':{'color':'#c4894a','width':2},'thickness':0.75,'value':10000}
        }
    ))
    fig_g.update_layout(height=260, paper_bgcolor='#fff', margin=dict(t=35,b=5,l=15,r=15))
    st.plotly_chart(fig_g, use_container_width=True)

# ── Evaluation (static, pre-built) ────────────────────────────────
st.markdown("---")
st.markdown("## Model Evaluation")

e1,e2,e3,e4 = st.columns(4)
e1.metric("Model", ev['model_name'])
e2.metric("MAE",   f"€{ev['mae']:,.0f}",  f"{ev['mae_pct']:.2f}% of mean")
e3.metric("RMSE",  f"€{ev['rmse']:,.0f}", f"{ev['rmse_pct']:.2f}% of mean")
e4.metric("R²",    f"{ev['r2']:.4f}")

st.markdown("<br>", unsafe_allow_html=True)
t1,t2,t3 = st.columns(3)
for col, label, val, ok in [
    (t1, "MAE < 15%",  f"{ev['mae_pct']:.2f}%",  ev['mae_pct']  < 15),
    (t2, "RMSE < 15%", f"{ev['rmse_pct']:.2f}%", ev['rmse_pct'] < 15),
    (t3, "R² > 0.85",  f"{ev['r2']:.4f}",        ev['r2'] > 0.85),
]:
    clr = '#5a8a5a' if ok else '#c4894a'
    col.markdown(
        f'<div class="trow"><span class="dp" style="background:{clr}"></span>'
        f'<span>{label}</span>'
        f'<span style="margin-left:auto;font-family:DM Mono,monospace;color:#1c1c1a">{val}</span></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Actual vs Predicted")
st.plotly_chart(fig_scatter, use_container_width=True)

if fig_fi_chart:
    st.markdown("### Feature Importance")
    st.plotly_chart(fig_fi_chart, use_container_width=True)

st.markdown("---")
st.markdown(
    '<p style="font-size:.76rem;color:#bbb;">Kelompok 1 — LM01 &nbsp;·&nbsp; '
    'Louis Huang &nbsp;·&nbsp; Gilbert Tjandra Adanarianto &nbsp;·&nbsp; Dava Rabbani Adrian Widyatmoko<br>'
    'Dataset: <a href="https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales" style="color:#bbb">Rossman Store Sales — Kaggle</a></p>',
    unsafe_allow_html=True
)