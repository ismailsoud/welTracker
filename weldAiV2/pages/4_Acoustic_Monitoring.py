"""
Acoustic Integrity Monitoring Panel (INNOVATION).
Real-time acoustic emission analysis for early defect detection.
Correlates acoustic signatures with welding parameters.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.acoustic_analyzer import AcousticIntegrityAnalyzer
from iot.simulator import WeldingSensorSimulator
from core.traceability import TraceabilityManager


def main():
    st.set_page_config(page_title="Acoustic Monitoring", page_icon="🔊", layout="wide")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #00B894, #00CEC9);
         padding: 0.8rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Acoustic Integrity Monitoring</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">
            INNOVATION - Early defect detection through acoustic emission analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if "acoustic_analyzer" not in st.session_state:
        st.session_state.acoustic_analyzer = AcousticIntegrityAnalyzer()
        if not st.session_state.acoustic_analyzer.is_trained:
            st.session_state.acoustic_analyzer.train_classifier()
    if "traceability" not in st.session_state:
        from app import init_session_state
        init_session_state()

    analyzer = st.session_state.acoustic_analyzer
    tm = st.session_state.traceability

    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Analysis", "Weld Acoustic Profile",
        "Defect Classification", "Parameter Correlation"
    ])

    with tab1:
        _render_live_analysis(analyzer)

    with tab2:
        _render_weld_profile(analyzer, tm)

    with tab3:
        _render_classification(analyzer)

    with tab4:
        _render_correlation(analyzer, tm)


def _render_live_analysis(analyzer):
    st.subheader("Live Acoustic Emission Simulation")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        n_signals = st.slider("Number of signals", 10, 100, 30)
    with col2:
        defect_ratio = st.slider("Defect injection ratio", 0.0, 0.5, 0.1, 0.05)
    with col3:
        st.write("")
        st.write("")
        generate = st.button("Generate Acoustic Data", type="primary")

    if generate:
        sim = WeldingSensorSimulator()
        sim.start_welding()
        readings = []
        classifications = []

        for i in range(n_signals):
            has_defect = np.random.random() < defect_ratio
            reading = sim.generate_acoustic_reading(has_defect=has_defect)
            readings.append(reading)

            cls_result = analyzer.classify(reading)
            classifications.append(cls_result)

        quality = analyzer.compute_quality_index(classifications)
        signal_df = pd.DataFrame({
            "signal_idx": list(range(1, len(readings) + 1)),
            "amplitude_db": [r["amplitude_db"] for r in readings],
            "energy": [r["energy"] for r in readings],
            "frequency_khz": [r.get("peak_frequency_hz", 0) / 1000 for r in readings],
            "classification": [c["classification"] for c in classifications],
            "confidence": [c["confidence"] for c in classifications],
        })
        signal_df["is_defect"] = signal_df["classification"] != "normal"
        signal_df["rolling_amp"] = signal_df["amplitude_db"].rolling(5, min_periods=1).mean()
        signal_df["rolling_energy"] = signal_df["energy"].rolling(5, min_periods=1).mean()

        # Quality Index Display
        qi = quality["quality_index"]
        qi_color = "#00D26A" if qi > 0.85 else "#F7931E" if qi > 0.7 else "#F23645"
        status_label = quality["status"].replace("_", " ").title()

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Quality Index", f"{qi:.3f}", delta=status_label)
        with col_b:
            st.metric("Normal Ratio", f"{quality['normal_ratio']:.1%}")
        with col_c:
            st.metric("Total Signals", quality["total_readings"])
        with col_d:
            st.metric("Avg Defect Conf.",
                       f"{quality['avg_defect_confidence']:.2%}")
        st.progress(min(max(qi, 0.0), 1.0), text=f"Acoustic quality score: {qi:.1%} ({status_label})")
        st.caption("Interpretation: values near 1.0 indicate stable arc acoustics, while lower values indicate probable defect signatures.")

        st.divider()

        # Waveform Visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Acoustic Waveform (Latest)")
            last_reading = readings[-1]
            if "waveform" in last_reading and last_reading["waveform"]:
                waveform = np.array(last_reading["waveform"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(waveform))),
                    y=waveform, mode="lines",
                    line=dict(color="#00CEC9", width=1.8),
                    name="Waveform",
                    hovertemplate="Sample %{x}<br>Amplitude %{y:.4f}<extra></extra>",
                ))
                fig.add_hline(y=0, line_color="rgba(203,213,225,0.45)", line_dash="dot")
                fig.update_layout(
                    height=250, margin=dict(t=20, b=20),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    xaxis_title="Sample", yaxis_title="Amplitude",
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

        with col2:
            st.subheader("Frequency Spectrum (Latest)")
            if "fft_magnitude" in last_reading and "fft_frequencies" in last_reading:
                fft_mag = last_reading["fft_magnitude"]
                fft_freq = last_reading["fft_frequencies"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[f / 1000 for f in fft_freq[:32]],
                    y=fft_mag[:32],
                    mode="lines+markers",
                    fill="tozeroy",
                    marker=dict(size=5, color="#FF6B35"),
                    line=dict(width=2, color="#FF6B35"),
                    name="FFT Magnitude",
                    hovertemplate="Freq %{x:.1f} kHz<br>Magnitude %{y:.4f}<extra></extra>",
                ))
                fig.update_layout(
                    height=250, margin=dict(t=20, b=20),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    xaxis_title="Frequency (kHz)", yaxis_title="Magnitude",
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

        # Combined trend view: amplitude + energy + defect markers
        color_map = {"normal": "#00D26A", "porosity": "#F7931E", "crack": "#F23645", "lack_of_fusion": "#E74C3C", "slag_inclusion": "#9B59B6"}
        marker_colors = [color_map.get(c, "#94A3B8") for c in signal_df["classification"]]
        trend_fig = make_subplots(specs=[[{"secondary_y": True}]])
        trend_fig.add_trace(
            go.Scatter(
                x=signal_df["signal_idx"],
                y=signal_df["amplitude_db"],
                mode="markers",
                marker=dict(color=marker_colors, size=8),
                text=[f"{c} ({conf:.1%})" for c, conf in zip(signal_df["classification"], signal_df["confidence"])],
                name="Amplitude",
                hovertemplate="Signal %{x}<br>Amplitude %{y:.2f} dB<br>%{text}<extra></extra>",
            ),
            secondary_y=False,
        )
        trend_fig.add_trace(
            go.Scatter(
                x=signal_df["signal_idx"],
                y=signal_df["rolling_amp"],
                mode="lines",
                line=dict(color="#22D3EE", width=2.2),
                name="Amplitude trend (5-pt MA)",
                hovertemplate="Signal %{x}<br>Avg amplitude %{y:.2f} dB<extra></extra>",
            ),
            secondary_y=False,
        )
        trend_fig.add_trace(
            go.Scatter(
                x=signal_df["signal_idx"],
                y=signal_df["rolling_energy"],
                mode="lines",
                line=dict(color="#A78BFA", width=2, dash="dash"),
                name="Energy trend (5-pt MA)",
                hovertemplate="Signal %{x}<br>Avg energy %{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )
        trend_fig.add_hline(y=60, line_dash="dash", line_color="#F59E0B", opacity=0.8, annotation_text="Amplitude warning (60 dB)")
        trend_fig.update_layout(
            title="Acoustic Signal Trends (Amplitude + Energy)",
            template="plotly_dark",
            height=340,
            margin=dict(t=45, b=20, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        trend_fig.update_xaxes(title_text="Signal #", gridcolor="rgba(255,255,255,0.12)")
        trend_fig.update_yaxes(title_text="Amplitude (dB)", secondary_y=False, gridcolor="rgba(255,255,255,0.12)")
        trend_fig.update_yaxes(title_text="Energy", secondary_y=True, gridcolor="rgba(255,255,255,0.10)")
        st.plotly_chart(trend_fig, use_container_width=True, theme=None)

        # Defect Distribution
        st.subheader("Classification Distribution")
        dist = quality.get("defect_distribution", {})
        if dist:
            dist_df = pd.DataFrame([
                {"Class": k, "Count": v} for k, v in dist.items()
            ])
            dist_df["Class"] = dist_df["Class"].str.replace("_", " ").str.title()
            dist_df = dist_df.sort_values("Count", ascending=False)
            dist_df["Percentage"] = (dist_df["Count"] / dist_df["Count"].sum() * 100).round(1)
            fig = px.bar(
                dist_df,
                x="Class",
                y="Count",
                color="Class",
                text=dist_df["Percentage"].astype(str) + "%",
                color_discrete_map={k.replace("_", " ").title(): v for k, v in color_map.items()},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=320,
                margin=dict(t=20, b=20),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis_title="Signals",
                xaxis_title="Detected class",
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
            st.dataframe(
                dist_df[["Class", "Count", "Percentage"]],
                use_container_width=True,
                hide_index=True,
            )


def _render_weld_profile(analyzer, tm):
    st.subheader("Weld Acoustic Profile")

    weld_ids = [wid for wid in tm.acoustic_data if tm.acoustic_data[wid]]
    if not weld_ids:
        st.info("No acoustic data available. Generate data on the Live Analysis tab.")
        return

    selected = st.selectbox("Select Weld", weld_ids)
    acoustics = tm.acoustic_data[selected]

    if not acoustics:
        st.warning("No acoustic data for this weld.")
        return

    # Summary metrics
    amplitudes = [a.amplitude_db for a in acoustics]
    energies = [a.energy for a in acoustics]
    defect_count = sum(1 for a in acoustics if a.classification != "normal")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Readings", len(acoustics))
    with col2:
        st.metric("Avg Amplitude", f"{np.mean(amplitudes):.1f} dB")
    with col3:
        st.metric("Max Energy", f"{max(energies):.0f}")
    with col4:
        st.metric("Defect Signals", defect_count,
                   delta=f"{defect_count/len(acoustics)*100:.1f}%")

    # Amplitude Timeline
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Amplitude (dB)", "Energy"),
                        vertical_spacing=0.15)

    colors = ["#F23645" if a.classification != "normal" else "#00CEC9"
              for a in acoustics]

    fig.add_trace(go.Scatter(
        x=list(range(len(acoustics))),
        y=amplitudes,
        mode="markers",
        marker=dict(color=colors, size=4),
        name="Amplitude",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(range(len(acoustics))),
        y=energies,
        mode="markers",
        marker=dict(color=colors, size=4),
        name="Energy",
    ), row=2, col=1)

    fig.update_layout(
        height=500, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    fig.update_xaxes(title_text="Reading #", gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    st.plotly_chart(fig, use_container_width=True)

    # Classification breakdown
    class_counts = {}
    for a in acoustics:
        class_counts[a.classification] = class_counts.get(a.classification, 0) + 1

    st.subheader("Defect Classification Breakdown")
    cls_df = pd.DataFrame([
        {"Type": k, "Count": v, "Percentage": f"{v/len(acoustics)*100:.1f}%"}
        for k, v in class_counts.items()
    ])
    st.dataframe(cls_df, use_container_width=True, hide_index=True)


def _render_classification(analyzer):
    st.subheader("Acoustic Defect Classifier")
    st.markdown("""
    The acoustic classifier identifies defect types from emission signals:
    - **Normal**: Regular welding acoustic signature
    - **Porosity**: Gas pockets trapped in weld metal
    - **Crack**: Fracture initiation in weld or HAZ
    - **Lack of Fusion**: Incomplete fusion between passes or base metal
    - **Slag Inclusion**: Non-metallic material trapped in weld
    """)

    if hasattr(analyzer, 'training_report') and analyzer.training_report:
        report = analyzer.training_report
        st.subheader("Classifier Performance")

        full_metrics = report.get("_full", {})
        best_algo = full_metrics.get("best_algorithm", "")
        cv_f1 = full_metrics.get("cv_macro_f1", 0)
        if best_algo:
            st.info(f"Best algorithm: **{best_algo}** | CV macro-F1: **{cv_f1:.4f}** (5-fold stratified)")

        overall_acc = report.get("accuracy", 0)
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0)
        weighted_f1 = report.get("weighted avg", {}).get("f1-score", 0)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Overall Accuracy", f"{overall_acc:.2%}")
        with m2:
            st.metric("Macro F1", f"{macro_f1:.2%}")
        with m3:
            st.metric("Weighted F1", f"{weighted_f1:.2%}")

        classes = analyzer.DEFECT_CLASSES
        metrics_data = []
        for cls in classes:
            if cls in report:
                metrics_data.append({
                    "Class": cls.replace("_", " ").title(),
                    "Precision": f"{report[cls]['precision']:.2%}",
                    "Recall": f"{report[cls]['recall']:.2%}",
                    "F1-Score": f"{report[cls]['f1-score']:.2%}",
                    "Support": report[cls]["support"],
                })

        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True,
                         hide_index=True)

        cv_comp = full_metrics.get("cv_comparison", {})
        if cv_comp:
            with st.expander("Algorithm Comparison (5-Fold CV)"):
                comp_df = pd.DataFrame([
                    {"Algorithm": k, "Mean macro-F1": v.get("mean_f1", 0),
                     "Std": v.get("std_f1", 0)}
                    for k, v in cv_comp.items()
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        best_params = full_metrics.get("best_params", {})
        if best_params:
            with st.expander("Tuned Hyperparameters"):
                st.json(best_params)

    st.divider()
    st.subheader("Defect Acoustic Signatures")
    sig_data = []
    for defect, sig in analyzer.defect_signatures.items():
        sig_data.append({
            "Defect Type": defect.replace("_", " ").title(),
            "Freq Range (Hz)": f"{sig['freq_range'][0]:,} - {sig['freq_range'][1]:,}",
            "Amplitude Range (dB)": f"{sig['amplitude_range'][0]} - {sig['amplitude_range'][1]}",
            "Energy Range": f"{sig['energy_range'][0]} - {sig['energy_range'][1]}",
            "Burst Pattern": "Yes" if sig["burst_pattern"] else "No",
        })
    st.dataframe(pd.DataFrame(sig_data), use_container_width=True, hide_index=True)


def _render_correlation(analyzer, tm):
    st.subheader("Acoustic-Parameter Correlation Analysis")
    st.markdown("""
    This analysis reveals how welding parameters influence acoustic emissions,
    enabling early identification of parameter combinations that lead to defects.
    """)

    weld_ids = [wid for wid in tm.acoustic_data
                if tm.acoustic_data[wid] and tm.weld_passes.get(wid)]
    if not weld_ids:
        st.info("Need both acoustic and welding pass data for correlation analysis.")
        return

    selected = st.selectbox("Select Weld for Correlation", weld_ids)

    acoustics = tm.acoustic_data[selected]
    passes = tm.weld_passes[selected]

    acoustic_readings = [
        {"amplitude_db": a.amplitude_db, "energy": a.energy,
         "frequency_hz": a.frequency_hz}
        for a in acoustics[:len(passes) * 10]
    ]

    welding_params_expanded = []
    for p in passes:
        for _ in range(min(10, len(acoustic_readings) // max(len(passes), 1))):
            welding_params_expanded.append({
                "current_a": p.current_a + np.random.normal(0, 2),
                "voltage_v": p.voltage_v + np.random.normal(0, 0.5),
                "travel_speed_mm_min": p.travel_speed_mm_min + np.random.normal(0, 5),
            })

    min_len = min(len(acoustic_readings), len(welding_params_expanded))
    if min_len < 3:
        st.warning("Insufficient data for correlation analysis.")
        return

    correlation_result = analyzer.correlate_with_params(
        acoustic_readings[:min_len],
        welding_params_expanded[:min_len],
    )

    # Display correlations
    correlations = correlation_result.get("correlations", {})
    if correlations:
        corr_data = []
        for pair, data in correlations.items():
            corr_data.append({
                "Parameter Pair": pair.replace("_vs_", " vs ").replace("_", " ").title(),
                "Correlation (r)": f"{data['correlation']:.4f}",
                "P-Value": f"{data['p_value']:.6f}",
                "Significant": "Yes" if data["significant"] else "No",
            })
        st.dataframe(pd.DataFrame(corr_data), use_container_width=True,
                     hide_index=True)

    insights = correlation_result.get("insights", [])
    if insights:
        st.subheader("Key Insights")
        for insight in insights:
            st.info(insight)

    # Scatter plots
    if min_len > 3:
        a_amps = [r["amplitude_db"] for r in acoustic_readings[:min_len]]
        w_currents = [r["current_a"] for r in welding_params_expanded[:min_len]]
        w_voltages = [r["voltage_v"] for r in welding_params_expanded[:min_len]]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(x=w_currents, y=a_amps,
                             labels={"x": "Current (A)", "y": "Amplitude (dB)"},
                             title="Acoustic Amplitude vs Current",
                             trendline="ols",
                             color_discrete_sequence=["#FF6B35"])
            fig.update_layout(
                height=300, margin=dict(t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(x=w_voltages, y=a_amps,
                             labels={"x": "Voltage (V)", "y": "Amplitude (dB)"},
                             title="Acoustic Amplitude vs Voltage",
                             trendline="ols",
                             color_discrete_sequence=["#4ECDC4"])
            fig.update_layout(
                height=300, margin=dict(t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
