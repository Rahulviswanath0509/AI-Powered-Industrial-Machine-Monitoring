"""
dashboard.py
Professional Streamlit dashboard for the AI Maintenance System.
Run with:  streamlit run dashboard.py
"""

import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Industrial Maintenance Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  .main { background-color: #0e1117; }
  .block-container { padding-top: 1rem; }

  /* Status cards */
  .card {
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
    border-left: 5px solid;
  }
  .card-running  { background:#0d2137; border-color:#00b4d8; }
  .card-warning  { background:#1e1a00; border-color:#ffd60a; }
  .card-critical { background:#1e0000; border-color:#ff4444; }
  .card-failed   { background:#120010; border-color:#9b59b6; }
  .card-title    { font-size:0.75rem; color:#8899aa; text-transform:uppercase; letter-spacing:1px; margin:0; }
  .card-value    { font-size:1.7rem; font-weight:700; margin:4px 0 0; }
  .card-sub      { font-size:0.78rem; color:#8899aa; margin:0; }
  .running-color  { color:#00b4d8; }
  .warning-color  { color:#ffd60a; }
  .critical-color { color:#ff4444; }
  .failed-color   { color:#9b59b6; }

  /* Section headers */
  .section-header {
    font-size:1.1rem; font-weight:600; color:#cdd6f4;
    border-bottom:1px solid #2a2d3e; padding-bottom:6px;
    margin-bottom:14px; margin-top:8px;
  }

  /* Alert badge */
  .badge-critical { background:#ff4444; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; }
  .badge-warning  { background:#ffd60a; color:#000; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; }
  .badge-info     { background:#00b4d8; color:#000; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=4)
def load_recent_logs(hours: int = 2) -> pd.DataFrame:
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = sqlite3.connect(config.DB_PATH)
        df   = pd.read_sql(
            "SELECT * FROM machine_logs WHERE timestamp >= ? ORDER BY timestamp",
            conn, params=(cutoff,)
        )
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=4)
def load_alerts(limit: int = 200) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(config.DB_PATH)
        df   = pd.read_sql(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
            conn, params=(limit,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=4)
def latest_status() -> pd.DataFrame:
    """Most recent row per machine."""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        df   = pd.read_sql("""
            SELECT ml.*
            FROM machine_logs ml
            INNER JOIN (
                SELECT machine_id, MAX(timestamp) AS max_ts
                FROM machine_logs
                GROUP BY machine_id
            ) sub ON ml.machine_id = sub.machine_id AND ml.timestamp = sub.max_ts
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ── Colour helpers ────────────────────────────────────────────────────────────
STATUS_COLOR = {
    "Running":  "#00b4d8",
    "Warning":  "#ffd60a",
    "Critical": "#ff4444",
    "Failed":   "#9b59b6",
}


def severity_badge(sev: str) -> str:
    cls = {"CRITICAL": "badge-critical", "WARNING": "badge-warning"}.get(sev, "badge-info")
    return f'<span class="{cls}">{sev}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏭 AI Maintenance")
    st.markdown("**Industrial Machine Monitor**")
    st.divider()

    time_window = st.selectbox("Time window", [1, 2, 6, 12, 24], index=1,
                               format_func=lambda h: f"Last {h} hour{'s' if h > 1 else ''}")
    selected_machines = st.multiselect(
        "Filter machines",
        options=list(config.MACHINES.keys()),
        default=list(config.MACHINES.keys()),
        format_func=lambda k: f"{k} – {config.MACHINES[k]['name']}"
    )
    alert_severity = st.multiselect("Alert severity", ["CRITICAL", "WARNING"], default=["CRITICAL", "WARNING"])

    st.divider()
    auto_refresh = st.toggle("Auto-refresh (5 s)", value=True)
    st.markdown(f"🕐 **{datetime.now().strftime('%H:%M:%S')}**")
    st.caption("Data refreshes every 5 seconds when active.")
    st.divider()
    st.caption("Powered by Isolation Forest · Scikit-learn · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

df_all   = load_recent_logs(hours=time_window)
df       = df_all[df_all["machine_id"].isin(selected_machines)] if not df_all.empty else df_all
df_alerts_all = load_alerts()
df_alerts = (
    df_alerts_all[
        df_alerts_all["machine_id"].isin(selected_machines) &
        df_alerts_all["severity"].isin(alert_severity)
    ]
    if not df_alerts_all.empty else df_alerts_all
)
df_latest = latest_status()
if not df_latest.empty and selected_machines:
    df_latest = df_latest[df_latest["machine_id"].isin(selected_machines)]

no_data = df.empty or df_latest.empty


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<h1 style="color: #00b4d8;">🏭 AI-Powered Industrial Machine Monitoring</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color: #cdd6f4;">Predictive Maintenance System · Real-Time Dashboard</h3>', unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERALL MACHINE STATUS CARDS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">📊 Overall Machine Status</p>', unsafe_allow_html=True)

if no_data:
    st.warning("⚠️ No data yet. Run `setup.py` first, then start `monitor_service.py`.")
else:
    status_counts = df_latest["status"].value_counts().to_dict()
    n_running  = status_counts.get("Running",  0)
    n_warning  = status_counts.get("Warning",  0)
    n_critical = status_counts.get("Critical", 0)
    n_failed   = status_counts.get("Failed",   0)
    n_total    = len(df_latest)
    n_alerts   = len(df_alerts)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class="card card-running">
            <p class="card-title">Total Machines</p>
            <p class="card-value running-color">{n_total}</p>
            <p class="card-sub">Monitored</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card card-running">
            <p class="card-title">Running</p>
            <p class="card-value running-color">{n_running}</p>
            <p class="card-sub">Healthy</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card card-warning">
            <p class="card-title">Warning</p>
            <p class="card-value warning-color">{n_warning}</p>
            <p class="card-sub">Needs attention</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="card card-critical">
            <p class="card-title">Critical</p>
            <p class="card-value critical-color">{n_critical}</p>
            <p class="card-sub">Immediate action</p></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="card card-failed">
            <p class="card-title">Failed</p>
            <p class="card-value failed-color">{n_failed}</p>
            <p class="card-sub">Offline</p></div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="card card-critical">
            <p class="card-title">Active Alerts</p>
            <p class="card-value critical-color">{n_alerts}</p>
            <p class="card-sub">Last {time_window}h</p></div>""", unsafe_allow_html=True)

    st.divider()

    # ── Per-machine status cards ──────────────────────────────────────────────
    st.markdown('<p class="section-header">🖥️ Individual Machine Status</p>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (_, row_m) in enumerate(df_latest.iterrows()):
        col_cls = {
            "Running":  "card-running",
            "Warning":  "card-warning",
            "Critical": "card-critical",
            "Failed":   "card-failed",
        }.get(row_m["status"], "card-running")
        val_cls = col_cls.replace("card-", "") + "-color"
        icon = {"Running": "✅", "Warning": "⚠️", "Critical": "🔴", "Failed": "💀"}.get(row_m["status"], "❓")
        with cols[i % 5]:
            st.markdown(f"""<div class="card {col_cls}">
                <p class="card-title">{row_m['machine_id']} · {row_m.get('location','')}</p>
                <p class="card-value {val_cls}" style="font-size:1rem">{icon} {row_m['status']}</p>
                <p class="card-sub">{row_m['machine_name']}</p>
                <p class="card-sub">🌡 {row_m['temperature']}°C &nbsp;|&nbsp;
                                    〰 {row_m['vibration']} mm/s &nbsp;|&nbsp;
                                    ⚡ {row_m['rpm']} RPM</p>
            </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — RUNNING vs FAILED PIE
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-header">📈 Fleet Health Overview</p>', unsafe_allow_html=True)

    pie_col, bar_col = st.columns([1, 2])

    with pie_col:
        labels = list(status_counts.keys())
        values = list(status_counts.values())
        colors = [STATUS_COLOR.get(l, "#888") for l in labels]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker_colors=colors,
            hole=0.5,
            textinfo="label+percent",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            title="Machine Status Distribution",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#cdd6f4", margin=dict(t=40, b=0, l=0, r=0),
            showlegend=False, height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with bar_col:
        if not df.empty:
            hourly = (
                df.assign(hour=df["timestamp"].dt.floor("h"))
                  .groupby(["hour", "status"])
                  .size()
                  .reset_index(name="count")
            )
            fig_bar = px.bar(
                hourly, x="hour", y="count", color="status",
                color_discrete_map=STATUS_COLOR,
                title="Status Over Time",
                labels={"hour": "Hour", "count": "Readings", "status": "Status"},
            )
            fig_bar.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                font_color="#cdd6f4", margin=dict(t=40, b=0),
                legend_title_text="", height=300,
            )
            st.plotly_chart(fig_bar, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — ACTIVE ALERTS TABLE
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-header">🚨 Active Alerts</p>', unsafe_allow_html=True)

    if df_alerts.empty:
        st.success("✅ No active alerts for the selected filters.")
    else:
        # Display last 30
        show_alerts = df_alerts.head(30).copy()
        show_alerts["severity_badge"] = show_alerts["severity"].apply(severity_badge)

        # Use st.dataframe with colour coding
        st.dataframe(
            show_alerts[["timestamp", "machine_id", "machine_name", "alert_type", "severity", "message"]]
            .rename(columns={
                "timestamp": "Time", "machine_id": "ID",
                "machine_name": "Machine", "alert_type": "Type",
                "severity": "Severity", "message": "Message",
            }),
            use_container_width=True,
            height=min(35 * len(show_alerts) + 38, 400),
            column_config={
                "Severity": st.column_config.TextColumn("Severity"),
            },
            hide_index=True,
        )


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — TEMPERATURE TREND CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-header">🌡️ Temperature Trends</p>', unsafe_allow_html=True)

    if not df.empty:
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            # Line chart for all machines (downsampled)
            df_temp = df.groupby(["timestamp", "machine_id"])["temperature"].mean().reset_index()
            fig_temp = px.line(
                df_temp, x="timestamp", y="temperature", color="machine_id",
                title="Temperature Over Time (all machines)",
                labels={"temperature": "Temp (°C)", "timestamp": "Time", "machine_id": "Machine"},
            )
            fig_temp.add_hline(y=config.THRESHOLDS["temperature_high"],
                               line_dash="dash", line_color="#ffd60a",
                               annotation_text="Warning", annotation_font_color="#ffd60a")
            fig_temp.add_hline(y=config.THRESHOLDS["temperature_critical"],
                               line_dash="dash", line_color="#ff4444",
                               annotation_text="Critical", annotation_font_color="#ff4444")
            fig_temp.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                   font_color="#cdd6f4", margin=dict(t=40, b=0), height=350)
            st.plotly_chart(fig_temp, use_container_width=True)

        with t_col2:
            # Latest temperature bar
            fig_tb = px.bar(
                df_latest.sort_values("temperature", ascending=True),
                x="temperature", y="machine_id", orientation="h",
                color="temperature",
                color_continuous_scale=["#00b4d8", "#ffd60a", "#ff4444"],
                title="Current Temperature by Machine",
                labels={"temperature": "Temp (°C)", "machine_id": "Machine"},
            )
            fig_tb.add_vline(x=config.THRESHOLDS["temperature_high"],
                             line_dash="dash", line_color="#ffd60a")
            fig_tb.add_vline(x=config.THRESHOLDS["temperature_critical"],
                             line_dash="dash", line_color="#ff4444")
            fig_tb.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                 font_color="#cdd6f4", margin=dict(t=40, b=0),
                                 coloraxis_showscale=False, height=350)
            st.plotly_chart(fig_tb, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — VIBRATION TREND CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-header">〰️ Vibration Trends</p>', unsafe_allow_html=True)

    if not df.empty:
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            df_vib = df.groupby(["timestamp", "machine_id"])["vibration"].mean().reset_index()
            fig_vib = px.line(
                df_vib, x="timestamp", y="vibration", color="machine_id",
                title="Vibration Over Time (all machines)",
                labels={"vibration": "Vibration (mm/s)", "timestamp": "Time", "machine_id": "Machine"},
            )
            fig_vib.add_hline(y=config.THRESHOLDS["vibration_high"],
                              line_dash="dash", line_color="#ffd60a",
                              annotation_text="Warning", annotation_font_color="#ffd60a")
            fig_vib.add_hline(y=config.THRESHOLDS["vibration_critical"],
                              line_dash="dash", line_color="#ff4444",
                              annotation_text="Critical", annotation_font_color="#ff4444")
            fig_vib.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#cdd6f4", margin=dict(t=40, b=0), height=350)
            st.plotly_chart(fig_vib, use_container_width=True)

        with v_col2:
            fig_vb = px.bar(
                df_latest.sort_values("vibration", ascending=True),
                x="vibration", y="machine_id", orientation="h",
                color="vibration",
                color_continuous_scale=["#00b4d8", "#ffd60a", "#ff4444"],
                title="Current Vibration by Machine",
                labels={"vibration": "Vibration (mm/s)", "machine_id": "Machine"},
            )
            fig_vb.add_vline(x=config.THRESHOLDS["vibration_high"],
                             line_dash="dash", line_color="#ffd60a")
            fig_vb.add_vline(x=config.THRESHOLDS["vibration_critical"],
                             line_dash="dash", line_color="#ff4444")
            fig_vb.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                 font_color="#cdd6f4", margin=dict(t=40, b=0),
                                 coloraxis_showscale=False, height=350)
            st.plotly_chart(fig_vb, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — FAILURE STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-header">💀 Failure Statistics</p>', unsafe_allow_html=True)

    if not df.empty:
        fs_col1, fs_col2, fs_col3 = st.columns(3)

        # Failure count per machine
        fail_df = df[df["status"] == "Failed"].groupby("machine_id").size().reset_index(name="failures")
        with fs_col1:
            if fail_df.empty:
                st.info("No failures in selected window.")
            else:
                fig_fail = px.bar(
                    fail_df.sort_values("failures", ascending=False),
                    x="machine_id", y="failures",
                    color="failures", color_continuous_scale=["#ffd60a", "#ff4444"],
                    title="Failure Count by Machine",
                )
                fig_fail.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                       font_color="#cdd6f4", margin=dict(t=40, b=0),
                                       coloraxis_showscale=False, height=300)
                st.plotly_chart(fig_fail, use_container_width=True)

        # Alert type distribution
        with fs_col2:
            if not df_alerts_all.empty:
                atype_df = df_alerts_all["alert_type"].value_counts().reset_index()
                atype_df.columns = ["alert_type", "count"]
                fig_at = px.pie(atype_df, names="alert_type", values="count",
                                title="Alert Types", hole=0.4,
                                color_discrete_sequence=px.colors.sequential.Plasma_r)
                fig_at.update_layout(paper_bgcolor="#0e1117", font_color="#cdd6f4",
                                     margin=dict(t=40, b=0), height=300, showlegend=True)
                st.plotly_chart(fig_at, use_container_width=True)

        # Worst offenders by alert count
        with fs_col3:
            if not df_alerts_all.empty:
                worst = (
                    df_alerts_all.groupby(["machine_id", "machine_name"])
                    .size().reset_index(name="alert_count")
                    .sort_values("alert_count", ascending=False).head(10)
                )
                fig_worst = px.bar(
                    worst, x="alert_count", y="machine_id", orientation="h",
                    color="alert_count", color_continuous_scale=["#ffd60a", "#ff4444"],
                    title="Most Alerts (all time)",
                    labels={"alert_count": "Alerts", "machine_id": "Machine"},
                )
                fig_worst.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                        font_color="#cdd6f4", margin=dict(t=40, b=0),
                                        coloraxis_showscale=False, height=300)
                st.plotly_chart(fig_worst, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — ANOMALY HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-header">🤖 AI Anomaly Detection History</p>', unsafe_allow_html=True)

    if not df.empty:
        a_col1, a_col2 = st.columns(2)

        with a_col1:
            # Anomaly scatter over time
            df_anom = df.copy()
            df_anom["Anomaly"] = df_anom["anomaly"].map({1: "Anomaly", 0: "Normal"})
            fig_sc = px.scatter(
                df_anom.sample(min(2000, len(df_anom))),
                x="timestamp", y="temperature",
                color="Anomaly",
                color_discrete_map={"Anomaly": "#ff4444", "Normal": "#00b4d8"},
                opacity=0.6,
                title="Temperature with Anomaly Overlay",
                labels={"temperature": "Temp (°C)", "timestamp": "Time"},
            )
            fig_sc.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                 font_color="#cdd6f4", margin=dict(t=40, b=0), height=320)
            st.plotly_chart(fig_sc, use_container_width=True)

        with a_col2:
            # Anomaly rate by machine
            anom_rate = (
                df.groupby("machine_id")["anomaly"]
                .agg(["sum", "count"])
                .reset_index()
            )
            anom_rate["rate"] = (anom_rate["sum"] / anom_rate["count"] * 100).round(2)
            fig_rate = px.bar(
                anom_rate.sort_values("rate", ascending=False),
                x="machine_id", y="rate",
                color="rate",
                color_continuous_scale=["#00b4d8", "#ffd60a", "#ff4444"],
                title="Anomaly Rate by Machine (%)",
                labels={"rate": "Anomaly %", "machine_id": "Machine"},
            )
            fig_rate.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                   font_color="#cdd6f4", margin=dict(t=40, b=0),
                                   coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_rate, use_container_width=True)

        # Recent anomalies table
        recent_anoms = (
            df[df["anomaly"] == 1]
            [["timestamp", "machine_id", "machine_name", "temperature",
              "vibration", "pressure", "status"]]
            .sort_values("timestamp", ascending=False)
            .head(20)
        )
        if not recent_anoms.empty:
            st.markdown("**Recent Anomalous Readings**")
            st.dataframe(recent_anoms, use_container_width=True, hide_index=True,
                         height=min(38 * len(recent_anoms) + 38, 350))

    st.divider()
    st.caption(
        f"🏭 AI Industrial Maintenance System · "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"Data window: last {time_window}h"
    )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
