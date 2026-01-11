"""
Balatro Simulator Web App
Simple Streamlit interface for running simulations.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from balatro_sim.simulator import Simulator, RunSummary, BatchResult
from balatro_sim.presets import PRESETS

# Page config
st.set_page_config(
    page_title="Balatro Simulator",
    page_icon="🃏",
    layout="centered"
)

st.title("🃏 Balatro Simulator")
st.markdown("*Monte Carlo simulation of Balatro runs*")

# Initialize simulator (cached)
@st.cache_resource
def get_simulator():
    return Simulator()

sim = get_simulator()

# Sidebar for settings
st.sidebar.header("Settings")

# Preset selection
preset_options = list(PRESETS.keys())
preset_display = {k: f"{PRESETS[k].name}" for k in preset_options}

selected_preset = st.sidebar.selectbox(
    "Preset",
    options=preset_options,
    format_func=lambda x: preset_display[x]
)

# Show preset details
preset = PRESETS[selected_preset]
st.sidebar.markdown(f"*{preset.description}*")
st.sidebar.markdown(f"**Strategy:** {preset.strategy.value}")
if preset.starting_jokers:
    st.sidebar.markdown(f"**Starting Jokers:** {', '.join(preset.starting_jokers)}")

# Run mode
run_mode = st.sidebar.radio("Mode", ["Single Run", "Batch Runs"])

if run_mode == "Batch Runs":
    num_runs = st.sidebar.slider("Number of Runs", min_value=10, max_value=500, value=100, step=10)
else:
    verbose = st.sidebar.checkbox("Verbose Output", value=True)

st.divider()

# Run button
if st.button("🎲 Run Simulation", type="primary", use_container_width=True):

    if run_mode == "Single Run":
        with st.spinner("Running simulation..."):
            result = sim.run(selected_preset, verbose=False)

        # Display result
        if result.victory:
            st.success("🏆 VICTORY!")
        else:
            st.error("💀 DEFEAT")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ante Reached", result.ante_reached)
        with col2:
            st.metric("Blinds Beaten", f"{result.blinds_beaten}/24")
        with col3:
            st.metric("Final Money", f"${result.final_money}")

        # Details
        st.subheader("Run Details")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Jokers Collected:**")
            if result.jokers_collected:
                for joker in result.jokers_collected:
                    st.markdown(f"- {joker}")
            else:
                st.markdown("*None*")

        with col2:
            st.markdown("**Hand Levels:**")
            leveled = {k: v for k, v in result.hand_levels.items() if v > 1}
            if leveled:
                for hand, level in sorted(leveled.items(), key=lambda x: -x[1])[:5]:
                    st.markdown(f"- {hand}: Lv.{level}")
            else:
                st.markdown("*No upgrades*")

        st.markdown(f"**Planets Used:** {result.planets_used}")

    else:  # Batch mode
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run batch with progress updates
        wins = 0
        total_blinds = 0
        total_ante = 0
        max_ante = 0
        total_money = 0
        total_jokers = 0
        total_planets = 0
        ante_distribution = {}

        for i in range(num_runs):
            summary = sim.run(selected_preset, verbose=False)

            if summary.victory:
                wins += 1
            total_blinds += summary.blinds_beaten
            total_ante += summary.ante_reached
            max_ante = max(max_ante, summary.ante_reached)
            total_money += summary.final_money
            total_jokers += len(summary.jokers_collected)
            total_planets += summary.planets_used
            ante_distribution[summary.ante_reached] = ante_distribution.get(summary.ante_reached, 0) + 1

            # Update progress
            progress_bar.progress((i + 1) / num_runs)
            status_text.text(f"Run {i + 1}/{num_runs}... ({wins} wins so far)")

        progress_bar.empty()
        status_text.empty()

        # Build result
        result = BatchResult(
            runs=num_runs,
            wins=wins,
            win_rate=wins / num_runs * 100,
            avg_blinds=total_blinds / num_runs,
            avg_ante=total_ante / num_runs,
            max_ante=max_ante,
            avg_money=total_money / num_runs,
            avg_jokers=total_jokers / num_runs,
            avg_planets=total_planets / num_runs,
            ante_distribution=ante_distribution,
            preset_used=selected_preset,
        )

        # Display results
        st.subheader(f"Results ({num_runs} runs)")

        # Win rate highlight
        if result.win_rate > 50:
            st.success(f"🏆 Win Rate: {result.wins}/{result.runs} ({result.win_rate:.1f}%)")
        elif result.win_rate > 0:
            st.warning(f"Win Rate: {result.wins}/{result.runs} ({result.win_rate:.1f}%)")
        else:
            st.error(f"Win Rate: {result.wins}/{result.runs} ({result.win_rate:.1f}%)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Blinds", f"{result.avg_blinds:.1f}")
        with col2:
            st.metric("Avg Ante", f"{result.avg_ante:.1f}")
        with col3:
            st.metric("Max Ante", result.max_ante)
        with col4:
            st.metric("Avg Money", f"${result.avg_money:.0f}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Jokers", f"{result.avg_jokers:.1f}")
        with col2:
            st.metric("Avg Planets", f"{result.avg_planets:.1f}")

        # Ante distribution chart
        st.subheader("Ante Distribution")

        # Prepare data for chart
        import pandas as pd
        chart_data = pd.DataFrame({
            'Ante': list(ante_distribution.keys()),
            'Runs': list(ante_distribution.values())
        }).sort_values('Ante')

        st.bar_chart(chart_data.set_index('Ante'))

# Footer
st.divider()
st.markdown("*Built with Balatro Sim engine*")
