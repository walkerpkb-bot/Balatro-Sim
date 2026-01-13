"""
Balatro Simulator Web App
Detailed Streamlit interface for running simulations.
"""

import json
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from balatro_sim.simulator import Simulator, RunSummary, BatchResult, __version__ as sim_version
from balatro_sim.presets import PRESETS, StrategyType

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Balatro Simulator",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .stMetric label {
        color: #a0a0a0 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #e94560 !important;
        font-weight: bold;
    }

    /* Expander styling */
    .stExpander {
        background: #0a0a0a;
        border: 1px solid #1a1a2e;
        border-radius: 8px;
    }

    /* Card-like containers */
    .card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2a2a4e;
        margin: 5px 0;
    }

    /* Joker card style */
    .joker-card {
        background: linear-gradient(135deg, #2d1b4e 0%, #1a1a2e 100%);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #4a3a6e;
        margin: 5px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Victory/Defeat banners */
    .victory-banner {
        background: linear-gradient(135deg, #1b4d1b 0%, #0a2a0a 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #2d8a2d;
    }
    .defeat-banner {
        background: linear-gradient(135deg, #4d1b1b 0%, #2a0a0a 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #8a2d2d;
    }

    /* Section headers */
    .section-header {
        color: #e94560;
        border-bottom: 2px solid #e94560;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }

    /* Shop caption */
    .shop-caption {
        background: #1a1a1a;
        padding: 8px 12px;
        border-radius: 5px;
        border-left: 3px solid #f0c040;
        margin: 5px 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load image assets
@st.cache_data
def load_image_assets():
    assets_path = Path(__file__).parent / "balatro_sim" / "data" / "image_assets.json"
    if assets_path.exists():
        with open(assets_path) as f:
            return json.load(f)
    return {"jokers": {}, "packs": {}, "decks": {}}

IMAGE_ASSETS = load_image_assets()


def get_joker_image(joker_name: str) -> str:
    return IMAGE_ASSETS.get("jokers", {}).get(joker_name, "")


def get_pack_image(pack_type: str, is_mega: bool = False) -> str:
    pack_data = IMAGE_ASSETS.get("packs", {}).get(pack_type, {})
    return pack_data.get("mega" if is_mega else "normal", "")


def get_deck_image(deck_name: str) -> str:
    return IMAGE_ASSETS.get("decks", {}).get(deck_name, "")


# Initialize simulator (cached)
@st.cache_resource
def get_simulator():
    return Simulator()

sim = get_simulator()

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("## 🎰 Settings")
    st.caption(f"Simulator v{sim_version}")
    st.divider()

    # Preset selection
    preset_options = list(PRESETS.keys())
    preset_display = {k: f"{PRESETS[k].name}" for k in preset_options}

    selected_preset = st.selectbox(
        "Deck Preset",
        options=preset_options,
        format_func=lambda x: preset_display[x]
    )

    preset = PRESETS[selected_preset]

    # Show deck image centered
    deck_name = preset.deck_type.value if hasattr(preset.deck_type, 'value') else str(preset.deck_type)
    deck_img = get_deck_image(deck_name)
    if deck_img:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(deck_img, width=100)

    st.caption(f"*{preset.description}*")

    st.divider()

    # Strategy selection
    strategy_options = {
        StrategyType.BASIC: "Basic - Simple highest score",
        StrategyType.SMART: "Smart - Joker synergies + hand levels",
        StrategyType.OPTIMIZED: "Optimized - Chases flushes/straights",
        StrategyType.AGGRESSIVE: "Aggressive - High risk, high reward"
    }
    selected_strategy = st.selectbox(
        "AI Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=1  # Default to Smart
    )

    st.divider()

    # Run mode
    run_mode = st.radio("Simulation Mode", ["Single Run", "Batch Analysis"], horizontal=True)

    num_runs = 100  # Default
    if run_mode == "Batch Analysis":
        num_runs = st.slider("Number of Runs", min_value=10, max_value=500, value=100, step=10)

    st.divider()

    # Strategy info
    with st.expander("Strategy Details"):
        st.markdown(f"**Type:** {preset.strategy.value}")
        if preset.starting_jokers:
            st.markdown(f"**Starting Jokers:** {', '.join(preset.starting_jokers)}")

# ============== MAIN AREA ==============
st.markdown("# 🃏 Balatro Simulator")
st.caption("Monte Carlo simulation of Balatro runs • Analyze strategies and deck performance")

st.divider()

# Run button
run_clicked = st.button("🎲 Run Simulation", type="primary", use_container_width=True)

if run_clicked:
    if run_mode == "Single Run":
        with st.spinner("🎴 Shuffling deck and running simulation..."):
            result = sim.run(selected_preset, verbose=False, strategy_override=selected_strategy)

        # Victory/Defeat banner
        if result.victory:
            st.markdown("""
                <div class="victory-banner">
                    <h1>🏆 VICTORY!</h1>
                    <p>Defeated all 8 antes</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="defeat-banner">
                    <h1>💀 DEFEAT</h1>
                    <p>Fell at Ante {result.ante_reached}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("")  # Spacing

        # Top-level metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Ante Reached", result.ante_reached, delta=None)
        with m2:
            st.metric("Blinds Beaten", f"{result.blinds_beaten}/24")
        with m3:
            st.metric("Final Money", f"${result.final_money}")
        with m4:
            bosses_beat = sum(1 for b in (result.blind_history or []) if b.boss_name and b.success)
            total_bosses = len(result.bosses_encountered or [])
            st.metric("Bosses Defeated", f"{bosses_beat}/{total_bosses}")

        st.divider()

        # Tabbed interface for details
        tab_timeline, tab_collection, tab_stats = st.tabs(["📜 Timeline", "🎴 Collection", "📊 Stats"])

        with tab_timeline:
            st.markdown("### Run Timeline")

            if result.blind_history:
                for i, blind in enumerate(result.blind_history):
                    icon = "✅" if blind.success else "❌"

                    if blind.boss_name:
                        blind_label = f"👹 BOSS: {blind.boss_name}"
                    else:
                        blind_label = f"{blind.blind_type} Blind"

                    margin_str = f"+{blind.margin_pct:.0f}%" if blind.margin_pct > 0 else f"{blind.margin_pct:.0f}%"
                    margin_color = "🟢" if blind.margin_pct > 20 else ("🟡" if blind.margin_pct > 0 else "🔴")

                    with st.expander(f"{icon} **Ante {blind.ante}** - {blind_label} • {blind.score:,} / {blind.required:,} ({margin_color} {margin_str})"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Score", f"{blind.score:,}")
                            st.metric("Required", f"{blind.required:,}")
                        with col2:
                            st.metric("Hands Used", blind.hands_used)
                            st.metric("Discards Used", blind.discards_used)
                        with col3:
                            st.metric("Margin", margin_str)
                            if blind.boss_name:
                                st.info(f"👹 {blind.boss_name}")

                        if blind.hands_played:
                            st.markdown("**Hands Played:**")
                            for hand_type, score in blind.hands_played:
                                bar_len = min(30, max(1, score // 300))
                                bar = "█" * bar_len
                                st.code(f"{hand_type:20} {score:>8,}  {bar}")

                    # Shop visit
                    if result.shop_history and i < len(result.shop_history):
                        shop = result.shop_history[i]
                        packs = getattr(shop, 'packs_opened', None) or []
                        has_purchases = shop.jokers_bought or shop.vouchers_bought or shop.planets_used or packs

                        if has_purchases:
                            shop_parts = []

                            if packs:
                                pack_str = ", ".join([f"📦 {p.get('type', '?')}" for p in packs])
                                shop_parts.append(pack_str)
                            if shop.jokers_bought:
                                shop_parts.append(f"🃏 {', '.join(shop.jokers_bought)}")
                            if shop.vouchers_bought:
                                shop_parts.append(f"🎫 {', '.join(shop.vouchers_bought)}")
                            if shop.planets_used:
                                shop_parts.append(f"🪐 {', '.join(shop.planets_used)}")

                            st.markdown(f"""
                                <div class="shop-caption">
                                    🛒 <strong>Shop:</strong> {' • '.join(shop_parts)} <em>(-${shop.money_spent})</em>
                                </div>
                            """, unsafe_allow_html=True)

        with tab_collection:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🃏 Jokers Collected")
                if result.jokers_collected:
                    for joker in result.jokers_collected:
                        joker_img = get_joker_image(joker)
                        with st.container():
                            c1, c2 = st.columns([1, 4])
                            with c1:
                                if joker_img:
                                    st.image(joker_img, width=60)
                                else:
                                    st.markdown("🃏")
                            with c2:
                                st.markdown(f"**{joker}**")
                else:
                    st.info("No jokers collected")

                st.markdown("### 🎫 Vouchers")
                if result.vouchers_acquired:
                    for voucher in result.vouchers_acquired:
                        st.markdown(f"• **{voucher}**")
                else:
                    st.caption("None acquired")

            with col2:
                st.markdown("### 📈 Hand Levels")
                leveled = {k: v for k, v in result.hand_levels.items() if v > 1}
                if leveled:
                    for hand, level in sorted(leveled.items(), key=lambda x: -x[1])[:10]:
                        progress = min(1.0, (level - 1) / 10)
                        st.markdown(f"**{hand}**")
                        st.progress(progress, text=f"Level {level}")
                else:
                    st.info("No hand upgrades")

                st.markdown("### 👹 Bosses")
                if result.bosses_encountered:
                    for boss in result.bosses_encountered:
                        boss_blinds = [b for b in result.blind_history if b.boss_name == boss]
                        if boss_blinds:
                            b = boss_blinds[0]
                            if b.success:
                                st.success(f"✅ {boss} (Ante {b.ante})")
                            else:
                                st.error(f"❌ {boss} (Ante {b.ante})")
                else:
                    st.caption("No bosses encountered")

        with tab_stats:
            st.markdown("### Performance Breakdown")

            col1, col2 = st.columns(2)

            with col1:
                # Hand type usage
                st.markdown("**Hands Played by Type:**")
                hand_counts = {}
                for blind in (result.blind_history or []):
                    for hand_type, score in (blind.hands_played or []):
                        hand_counts[hand_type] = hand_counts.get(hand_type, 0) + 1

                if hand_counts:
                    df = pd.DataFrame({
                        'Hand Type': list(hand_counts.keys()),
                        'Count': list(hand_counts.values())
                    }).sort_values('Count', ascending=False)
                    st.bar_chart(df.set_index('Hand Type'))

            with col2:
                # Score progression
                st.markdown("**Score Progression:**")
                if result.blind_history:
                    scores = [b.score for b in result.blind_history]
                    required = [b.required for b in result.blind_history]
                    df = pd.DataFrame({
                        'Score': scores,
                        'Required': required
                    })
                    st.line_chart(df)

    else:  # Batch Analysis
        progress_bar = st.progress(0)
        status_text = st.empty()

        wins = 0
        total_blinds = 0
        total_ante = 0
        max_ante = 0
        total_money = 0
        total_jokers = 0
        total_planets = 0
        total_vouchers = 0
        total_bosses_beat = 0
        ante_distribution = {}

        for i in range(num_runs):
            summary = sim.run(selected_preset, verbose=False, strategy_override=selected_strategy)

            if summary.victory:
                wins += 1
            total_blinds += summary.blinds_beaten
            total_ante += summary.ante_reached
            max_ante = max(max_ante, summary.ante_reached)
            total_money += summary.final_money
            total_jokers += len(summary.jokers_collected)
            total_planets += summary.planets_used
            total_vouchers += len(summary.vouchers_acquired or [])
            total_bosses_beat += sum(1 for b in (summary.blind_history or []) if b.boss_name and b.success)
            ante_distribution[summary.ante_reached] = ante_distribution.get(summary.ante_reached, 0) + 1

            progress_bar.progress((i + 1) / num_runs)
            status_text.text(f"Running simulation {i + 1}/{num_runs}... ({wins} wins)")

        progress_bar.empty()
        status_text.empty()

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

        # Results header
        st.markdown(f"## 📊 Batch Results ({num_runs} runs)")

        # Win rate highlight
        if result.win_rate > 50:
            st.success(f"🏆 **Win Rate: {result.wins}/{result.runs} ({result.win_rate:.1f}%)**")
        elif result.win_rate > 0:
            st.warning(f"⚠️ **Win Rate: {result.wins}/{result.runs} ({result.win_rate:.1f}%)**")
        else:
            st.error(f"💀 **Win Rate: {result.wins}/{result.runs} ({result.win_rate:.1f}%)**")

        st.divider()

        # Metrics grid
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Blinds", f"{result.avg_blinds:.1f}")
        with col2:
            st.metric("Avg Ante", f"{result.avg_ante:.1f}")
        with col3:
            st.metric("Max Ante", result.max_ante)
        with col4:
            st.metric("Avg Money", f"${result.avg_money:.0f}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Jokers", f"{result.avg_jokers:.1f}")
        with col2:
            st.metric("Avg Planets", f"{result.avg_planets:.1f}")
        with col3:
            st.metric("Avg Vouchers", f"{total_vouchers / num_runs:.1f}")
        with col4:
            st.metric("Avg Bosses Beat", f"{total_bosses_beat / num_runs:.1f}")

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Ante Distribution")
            chart_data = pd.DataFrame({
                'Ante': list(ante_distribution.keys()),
                'Runs': list(ante_distribution.values())
            }).sort_values('Ante')
            st.bar_chart(chart_data.set_index('Ante'))

        with col2:
            st.markdown("### Win/Loss Breakdown")
            pie_data = pd.DataFrame({
                'Result': ['Wins', 'Losses'],
                'Count': [wins, num_runs - wins]
            })
            st.bar_chart(pie_data.set_index('Result'))

# Footer
st.divider()
st.caption("Built with Balatro Sim engine • [GitHub](https://github.com/walkerpkb-bot/Balatro-Sim)")
