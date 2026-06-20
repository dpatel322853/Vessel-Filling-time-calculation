import math
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

st.set_page_config(page_title="Ethylene Vessel Filling Through RO", layout="wide")

st.title("Ethylene Vessel Filling Through Restriction Orifice")
st.caption("Dynamic engineering calculation tool for vessel pressure, RO flowrate, filling time, choked/non-choked transition, 3D plot, and PDF export.")


def simulate_filling(
    V_m3=90.0,
    T_C=35.0,
    Z=1.0,
    MW_kg_per_kmol=28.054,
    P_up_g_kgcm2=37.5,
    P_initial_g_kgcm2=0.0,
    P_target_g_kgcm2=37.0,
    DP_choke_kgcm2=17.0,
    Qn_choked_Nm3_h=6348.0,
    dt_s=1.0,
    Pn_Pa=101325.0,
    Tn_K=273.15,
    max_time_s=200000.0,
):
    """Simulate vessel filling using calibrated choked flow and sqrt(DP) non-choked scaling."""
    R = 8.314462618
    kgcm2_to_Pa = 98066.5
    P_atm_Pa = 101325.0
    P_atm_kgcm2 = P_atm_Pa / kgcm2_to_Pa
    T_K = T_C + 273.15
    MW = MW_kg_per_kmol / 1000.0  # kg/mol

    P_up_abs_kgcm2 = P_up_g_kgcm2 + P_atm_kgcm2
    P_initial_abs_kgcm2 = P_initial_g_kgcm2 + P_atm_kgcm2
    P_initial_abs_Pa = P_initial_abs_kgcm2 * kgcm2_to_Pa

    rho_n = Pn_Pa * MW / (R * Tn_K)  # kg/Nm3
    mdot_choked = Qn_choked_Nm3_h * rho_n / 3600.0

    mass_kg = P_initial_abs_Pa * V_m3 * MW / (Z * R * T_K)

    results = []
    transition = None
    previous_regime = None
    t = 0.0

    while t <= max_time_s:
        P_v_abs_Pa = mass_kg * Z * R * T_K / (V_m3 * MW)
        P_v_abs_kgcm2 = P_v_abs_Pa / kgcm2_to_Pa
        P_v_g_kgcm2 = P_v_abs_kgcm2 - P_atm_kgcm2
        DP_kgcm2 = max(P_up_abs_kgcm2 - P_v_abs_kgcm2, 0.0)

        if DP_kgcm2 >= DP_choke_kgcm2:
            regime = "Choked"
            Qn = Qn_choked_Nm3_h
            mdot = mdot_choked
        else:
            regime = "Non-choked"
            Qn = Qn_choked_Nm3_h * math.sqrt(max(DP_kgcm2, 0.0) / DP_choke_kgcm2) if DP_choke_kgcm2 > 0 else 0.0
            mdot = Qn * rho_n / 3600.0

        if previous_regime == "Choked" and regime == "Non-choked" and transition is None:
            transition = {
                "time_s": t,
                "time_min": t / 60.0,
                "pressure_g_kgcm2": P_v_g_kgcm2,
                "DP_kgcm2": DP_kgcm2,
                "flow_Nm3_h": Qn,
            }
        previous_regime = regime

        Q_actual_m3_h = mdot * Z * R * T_K / (max(P_v_abs_Pa, 1.0) * MW) * 3600.0

        results.append({
            "Time_s": t,
            "Time_min": t / 60.0,
            "Vessel_Pressure_kgcm2g": P_v_g_kgcm2,
            "Vessel_Pressure_kgcm2a": P_v_abs_kgcm2,
            "Differential_Pressure_kgcm2": DP_kgcm2,
            "Regime": regime,
            "Flow_Nm3_h": Qn,
            "Mass_Flow_kg_s": mdot,
            "Actual_Flow_m3_h": Q_actual_m3_h,
            "Vessel_Mass_kg": mass_kg,
        })

        if P_v_g_kgcm2 >= P_target_g_kgcm2:
            break
        if DP_kgcm2 <= 1e-9 and mdot <= 1e-9:
            break

        mass_kg += mdot * dt_s
        t += dt_s

    df = pd.DataFrame(results)
    summary = {
        "rho_n_kg_Nm3": rho_n,
        "mdot_choked_kg_s": mdot_choked,
        "total_time_s": df["Time_s"].iloc[-1],
        "total_time_min": df["Time_min"].iloc[-1],
        "final_pressure_g_kgcm2": df["Vessel_Pressure_kgcm2g"].iloc[-1],
        "final_flow_Nm3_h": df["Flow_Nm3_h"].iloc[-1],
        "transition": transition,
        "P_atm_kgcm2": P_atm_kgcm2,
    }
    return df, summary


def fig_to_buffer(fig, dpi=160):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf


def create_pdf_report(df, summary, inputs):
    """Create a PDF report containing inputs, KPIs, plots, and sample result table."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Ethylene Vessel Filling Through Restriction Orifice", styles["Title"]))
    story.append(Paragraph("Dynamic calculation report: pressure profile, RO flowrate, choked/non-choked transition, and selected plots.", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    transition_text = "Not detected"
    if summary["transition"]:
        tr = summary["transition"]
        transition_text = f"{tr['time_min']:.2f} min at {tr['pressure_g_kgcm2']:.2f} kg/cm2(g), DP {tr['DP_kgcm2']:.2f} kg/cm2"

    kpi_data = [
        ["Parameter", "Value", "Unit"],
        ["Total filling time", f"{summary['total_time_min']:.2f}", "min"],
        ["Final pressure", f"{summary['final_pressure_g_kgcm2']:.2f}", "kg/cm2(g)"],
        ["Choked mass flowrate", f"{summary['mdot_choked_kg_s']:.3f}", "kg/s"],
        ["Final flowrate", f"{summary['final_flow_Nm3_h']:.0f}", "Nm3/h"],
        ["Normal density", f"{summary['rho_n_kg_Nm3']:.3f}", "kg/Nm3"],
        ["Choked to non-choked transition", transition_text, "-"],
    ]
    kpi_table = Table(kpi_data, colWidths=[2.6 * inch, 3.4 * inch, 1.3 * inch])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.2 * inch))

    input_data = [["Input", "Value", "Unit"]]
    for name, value, unit in inputs:
        input_data.append([name, str(value), unit])
    input_table = Table(input_data, colWidths=[3.0 * inch, 2.0 * inch, 1.6 * inch])
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(Paragraph("Input Data", styles["Heading2"]))
    story.append(input_table)
    story.append(PageBreak())

    # Pressure and flow plots
    fig1, ax1 = plt.subplots(figsize=(7.5, 3.7))
    ax1.plot(df["Time_min"], df["Vessel_Pressure_kgcm2g"])
    ax1.set_xlabel("Time, min")
    ax1.set_ylabel("Vessel pressure, kg/cm2(g)")
    ax1.set_title("Vessel Pressure vs Time")
    ax1.grid(True)
    if summary["transition"]:
        ax1.axvline(summary["transition"]["time_min"], linestyle="--")
    img1 = Image(fig_to_buffer(fig1), width=5.0 * inch, height=2.45 * inch)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7.5, 3.7))
    ax2.plot(df["Time_min"], df["Flow_Nm3_h"])
    ax2.set_xlabel("Time, min")
    ax2.set_ylabel("RO flowrate, Nm3/h")
    ax2.set_title("Restriction Orifice Flowrate vs Time")
    ax2.grid(True)
    if summary["transition"]:
        ax2.axvline(summary["transition"]["time_min"], linestyle="--")
    img2 = Image(fig_to_buffer(fig2), width=5.0 * inch, height=2.45 * inch)
    plt.close(fig2)

    story.append(Paragraph("Trend Plots", styles["Heading2"]))
    story.append(Table([[img1, img2]], colWidths=[5.15 * inch, 5.15 * inch]))
    story.append(Spacer(1, 0.15 * inch))

    # 3D plot for PDF
    fig3 = plt.figure(figsize=(7.5, 4.5))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot(df["Time_min"], df["Vessel_Pressure_kgcm2g"], df["Flow_Nm3_h"])
    ax3.set_xlabel("Time, min")
    ax3.set_ylabel("Pressure, kg/cm2(g)")
    ax3.set_zlabel("Flowrate, Nm3/h")
    ax3.set_title("3D Trend: Pressure vs Time with Flowrate")
    img3 = Image(fig_to_buffer(fig3), width=5.6 * inch, height=3.3 * inch)
    plt.close(fig3)
    story.append(Paragraph("3D Plot", styles["Heading2"]))
    story.append(img3)
    story.append(PageBreak())

    # Result sample table: first 10 + last 10 rows
    cols = ["Time_min", "Vessel_Pressure_kgcm2g", "Differential_Pressure_kgcm2", "Regime", "Flow_Nm3_h", "Mass_Flow_kg_s"]
    if len(df) > 22:
        sample_df = pd.concat([df.head(10), df.tail(10)])
    else:
        sample_df = df.copy()
    sample_df = sample_df[cols].copy()
    for c in ["Time_min", "Vessel_Pressure_kgcm2g", "Differential_Pressure_kgcm2", "Flow_Nm3_h", "Mass_Flow_kg_s"]:
        sample_df[c] = sample_df[c].map(lambda x: f"{x:.3f}")
    table_data = [cols] + sample_df.values.tolist()
    result_table = Table(table_data, repeatRows=1)
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ]))
    story.append(Paragraph("Selected Simulation Results", styles["Heading2"]))
    story.append(Paragraph("Table includes first 10 and last 10 rows. Download CSV from app for full dataset.", styles["BodyText"]))
    story.append(result_table)

    doc.build(story)
    buffer.seek(0)
    return buffer


# Sidebar inputs
st.sidebar.header("Input Data")

with st.sidebar.expander("Fluid / property basis", expanded=True):
    MW = st.number_input("Molecular weight, kg/kmol", min_value=1.0, value=28.054, step=0.001, format="%.3f")
    Z = st.number_input("Compressibility factor, Z", min_value=0.1, value=1.0, step=0.01, format="%.3f")
    T_C = st.number_input("Vessel/upstream gas temperature, °C", value=35.0, step=1.0)
    Pn = st.number_input("Normal pressure, Pa", value=101325.0, step=100.0)
    Tn = st.number_input("Normal temperature, K", value=273.15, step=0.1)

with st.sidebar.expander("Upstream / vessel conditions", expanded=True):
    P_up_g = st.number_input("Upstream pressure, kg/cm²(g)", value=37.5, step=0.1)
    P_initial_g = st.number_input("Initial vessel pressure, kg/cm²(g)", value=0.0, step=0.1)
    P_target_g = st.number_input("Target vessel pressure, kg/cm²(g)", value=37.0, step=0.1)
    V = st.number_input("Vessel volume, m³", min_value=0.1, value=90.0, step=1.0)

with st.sidebar.expander("Restriction orifice", expanded=True):
    d_orifice_mm = st.number_input("Orifice bore diameter, mm", value=19.0, step=0.1)
    thickness_mm = st.number_input("Orifice thickness, mm", value=20.0, step=0.1)
    DP_choke = st.number_input("Choked differential pressure, kg/cm²", min_value=0.01, value=17.0, step=0.1)
    Qn_choked = st.number_input("Choked flowrate, Nm³/h", min_value=0.0, value=6348.0, step=10.0)

with st.sidebar.expander("Numerical settings", expanded=True):
    dt = st.number_input("Time step, s", min_value=0.01, value=1.0, step=0.1)
    max_time = st.number_input("Maximum simulation time, s", min_value=10.0, value=200000.0, step=1000.0)

# Run model
df, summary = simulate_filling(
    V_m3=V,
    T_C=T_C,
    Z=Z,
    MW_kg_per_kmol=MW,
    P_up_g_kgcm2=P_up_g,
    P_initial_g_kgcm2=P_initial_g,
    P_target_g_kgcm2=P_target_g,
    DP_choke_kgcm2=DP_choke,
    Qn_choked_Nm3_h=Qn_choked,
    dt_s=dt,
    Pn_Pa=Pn,
    Tn_K=Tn,
    max_time_s=max_time,
)

# KPI cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total filling time", f"{summary['total_time_min']:.2f} min")
col2.metric("Final pressure", f"{summary['final_pressure_g_kgcm2']:.2f} kg/cm²(g)")
col3.metric("Choked mass flow", f"{summary['mdot_choked_kg_s']:.3f} kg/s")
col4.metric("Final flow", f"{summary['final_flow_Nm3_h']:.0f} Nm³/h")

if summary["transition"]:
    tr = summary["transition"]
    st.success(
        f"Choked to non-choked transition at approximately {tr['time_min']:.2f} min, "
        f"vessel pressure {tr['pressure_g_kgcm2']:.2f} kg/cm²(g), DP {tr['DP_kgcm2']:.2f} kg/cm²."
    )
else:
    st.warning("No choked-to-non-choked transition was detected within the simulation range.")

# 2D plots
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(df["Time_min"], df["Vessel_Pressure_kgcm2g"])
    ax1.set_xlabel("Time, min")
    ax1.set_ylabel("Vessel pressure, kg/cm²(g)")
    ax1.set_title("Vessel Pressure vs Time")
    ax1.grid(True)
    if summary["transition"]:
        ax1.axvline(summary["transition"]["time_min"], linestyle="--")
    st.pyplot(fig1)

with plot_col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(df["Time_min"], df["Flow_Nm3_h"])
    ax2.set_xlabel("Time, min")
    ax2.set_ylabel("RO flowrate, Nm³/h")
    ax2.set_title("Restriction Orifice Flowrate vs Time")
    ax2.grid(True)
    if summary["transition"]:
        ax2.axvline(summary["transition"]["time_min"], linestyle="--")
    st.pyplot(fig2)

# 3D plot
st.subheader("3D Trend Plot")
st.caption("3D plot uses time on X-axis, vessel pressure on Y-axis, and RO flowrate on Z-axis. Colour indicates differential pressure.")
fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(
    x=df["Time_min"],
    y=df["Vessel_Pressure_kgcm2g"],
    z=df["Flow_Nm3_h"],
    mode="lines+markers",
    marker=dict(size=3, color=df["Differential_Pressure_kgcm2"], colorscale="Viridis", colorbar=dict(title="DP, kg/cm²")),
    line=dict(width=4),
    text=df["Regime"],
    hovertemplate="Time: %{x:.2f} min<br>Pressure: %{y:.2f} kg/cm²(g)<br>Flow: %{z:.0f} Nm³/h<br>Regime: %{text}<extra></extra>",
))
fig3d.update_layout(
    scene=dict(
        xaxis_title="Time, min",
        yaxis_title="Vessel pressure, kg/cm²(g)",
        zaxis_title="RO flowrate, Nm³/h",
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    height=650,
)
st.plotly_chart(fig3d, use_container_width=True)

# Downloads
st.subheader("Export Results")
input_list = [
    ("Vessel volume", V, "m³"),
    ("Temperature", T_C, "°C"),
    ("Compressibility factor Z", Z, "-"),
    ("Molecular weight", MW, "kg/kmol"),
    ("Upstream pressure", P_up_g, "kg/cm²(g)"),
    ("Initial vessel pressure", P_initial_g, "kg/cm²(g)"),
    ("Target vessel pressure", P_target_g, "kg/cm²(g)"),
    ("Choked differential pressure", DP_choke, "kg/cm²"),
    ("Choked flowrate", Qn_choked, "Nm³/h"),
    ("Time step", dt, "s"),
    ("Normal pressure", Pn, "Pa"),
    ("Normal temperature", Tn, "K"),
]

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download full results as CSV", data=csv, file_name="ethylene_vessel_filling_results.csv", mime="text/csv")

pdf_buffer = create_pdf_report(df, summary, input_list)
st.download_button(
    "Download PDF report",
    data=pdf_buffer,
    file_name="ethylene_vessel_filling_report.pdf",
    mime="application/pdf",
)

st.subheader("Simulation Table")
st.dataframe(df, use_container_width=True)

st.subheader("Engineering Notes")
st.markdown(
    """
- Base model uses calibrated choked flowrate and square-root pressure-drop scaling after de-choking.
- Pressure calculations use the real gas equation with user-specified constant Z.
- The 3D plot is intended for visual diagnostics; X = time, Y = vessel pressure, Z = RO flowrate, colour = differential pressure.
- PDF export includes inputs, KPIs, 2D plots, 3D plot, and selected simulation rows. CSV export should be used for the full detailed table.
- For higher accuracy, replace constant Z with Peng-Robinson/SRK or validated property package values.
"""
)
