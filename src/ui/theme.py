"""
Gestion de temas y estilos de la interfaz.

Este modulo maneja la aplicacion de temas claro/oscuro a la interfaz de Streamlit.
"""

import streamlit as st
from config.settings import THEMES


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convierte #RRGGBB en tupla RGB segura."""
    cleaned = (hex_color or "").strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(ch * 2 for ch in cleaned)
    if len(cleaned) != 6:
        return (37, 99, 235)
    try:
        return tuple(int(cleaned[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (37, 99, 235)


def _rgba(hex_color: str, alpha: float) -> str:
    """Devuelve color rgba con alpha clamp."""
    r, g, b = _hex_to_rgb(hex_color)
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r}, {g}, {b}, {a})"


def apply_theme(theme_name: str) -> None:
    """
    Aplica un tema a la interfaz de Streamlit.

    Args:
        theme_name: Nombre del tema ('claro' u 'oscuro')
    """
    theme = THEMES.get(theme_name, THEMES["oscuro"])
    is_dark = theme_name == "oscuro"

    accent_soft = _rgba(theme["accent"], 0.14 if is_dark else 0.10)
    accent_line = _rgba(theme["accent"], 0.35 if is_dark else 0.22)
    border = _rgba(theme["text"], 0.16 if is_dark else 0.14)
    muted_text = _rgba(theme["text"], 0.76 if is_dark else 0.70)
    card_bg = _rgba(theme["surface"], 0.82 if is_dark else 0.92)
    card_bg_alt = _rgba(theme["surface"], 0.65 if is_dark else 0.80)
    soft_bg = _rgba(theme["text"], 0.05 if is_dark else 0.035)
    ring = _rgba(theme["accent"], 0.35 if is_dark else 0.25)

    st.markdown(
        f"""
        <style>
        :root {{
            --ltm-bg: {theme['background']};
            --ltm-surface: {theme['surface']};
            --ltm-text: {theme['text']};
            --ltm-muted: {muted_text};
            --ltm-accent: {theme['accent']};
            --ltm-accent-alt: {theme['accent_alt']};
            --ltm-accent-soft: {accent_soft};
            --ltm-border: {border};
            --ltm-card: {card_bg};
            --ltm-card-alt: {card_bg_alt};
            --ltm-soft-bg: {soft_bg};
            --ltm-shadow: {theme['shadow']};
        }}

        .stApp {{
            background:
                radial-gradient(circle at 2% 2%, {accent_soft}, transparent 45%),
                radial-gradient(circle at 98% 98%, {accent_soft}, transparent 45%),
                var(--ltm-bg);
            color: var(--ltm-text);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}

        /* Dashboard Compact Mode */
        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1600px;
            transition: max-width 0.3s ease;
        }}

        [data-testid="stSidebar"] {{
            border-right: 1px solid var(--ltm-border);
        }}

        div[data-testid="stVerticalBlock"] {{
            gap: 1rem;
        }}

        .stApp h1, .stApp h2, .stApp h3 {{
            color: var(--ltm-text) !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em !important;
        }}

        .stButton > button {{
            border-radius: 10px !important;
            border: none !important;
            background: var(--ltm-accent) !important;
            color: white !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px {ring}, 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            background: var(--ltm-accent-alt) !important;
        }}

        .stExpander {{
            border: 1px solid var(--ltm-border) !important;
            border-radius: 12px !important;
            background: var(--ltm-surface) !important;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important;
            margin-bottom: 0.5rem !important;
        }}

        .stExpander:hover {{
            border-color: {accent_line} !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        }}

        .stExpander details > summary {{
            padding-top: 0.1rem !important;
            padding-bottom: 0.1rem !important;
        }}

        .ltm-hero {{
            border: 1px solid var(--ltm-border);
            background:
                linear-gradient(140deg, var(--ltm-card), var(--ltm-card-alt));
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
        }}

        .ltm-hero-title {{
            font-size: 1.28rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            line-height: 1.25;
        }}

        .ltm-hero-subtitle {{
            color: var(--ltm-muted);
            font-size: 0.93rem;
            margin-bottom: 0.55rem;
        }}

        .ltm-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin-top: 0.2rem;
        }}

        .ltm-chip {{
            display: inline-flex;
            align-items: center;
            padding: 0.23rem 0.58rem;
            border-radius: 999px;
            border: 1px solid var(--ltm-border);
            background: var(--ltm-soft-bg);
            color: var(--ltm-muted);
            font-size: 0.78rem;
            font-weight: 600;
        }}

        .ltm-chip-active {{
            color: var(--ltm-text);
            background: var(--ltm-accent-soft);
            border-color: {accent_line};
        }}

        .ltm-chip-done {{
            color: var(--ltm-text);
            background: {accent_soft};
            border-color: {accent_line};
        }}

        .ltm-card {{
            border: 1px solid var(--ltm-border);
            border-radius: 12px;
            padding: 0.52rem 0.68rem;
            background: var(--ltm-card);
            margin-bottom: 0.35rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }}

        .ltm-card-title {{
            font-size: 0.83rem;
            font-weight: 700;
            color: var(--ltm-text);
            margin-bottom: 0.2rem;
        }}

        .ltm-card-text {{
            font-size: 0.8rem;
            color: var(--ltm-muted);
            line-height: 1.35;
        }}

        .ltm-preview-column-marker,
        .ltm-controls-scroll-marker {{
            height: 0;
            width: 0;
            overflow: hidden;
            pointer-events: none;
        }}

        @media (min-width: 1100px) {{
            div[data-testid="column"]:has(.ltm-controls-scroll-marker) {{
                max-height: calc(100vh - 0.6rem);
                overflow-y: auto;
                overflow-x: hidden;
                padding-right: 0.32rem;
                scrollbar-gutter: stable;
            }}
            div[data-testid="column"]:has(.ltm-controls-scroll-marker) > div[data-testid="stVerticalBlock"] {{
                padding-right: 0.05rem;
            }}
            div[data-testid="column"]:has(.ltm-preview-column-marker) .stRadio > label {{
                margin-bottom: 0.08rem !important;
                font-size: 0.76rem !important;
            }}
            div[data-testid="column"]:has(.ltm-preview-column-marker) .stRadio [role="radiogroup"] {{
                gap: 0.24rem !important;
            }}
            div[data-testid="column"]:has(.ltm-preview-column-marker) .stButton > button {{
                padding: 0.28rem 0.45rem !important;
                min-height: 2rem !important;
                border-radius: 9px !important;
            }}
            div[data-testid="column"]:has(.ltm-preview-column-marker) [data-testid="stCaptionContainer"] p {{
                font-size: 0.74rem !important;
                margin-top: 0.05rem !important;
                margin-bottom: 0.06rem !important;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
        }}

        .ltm-step-intro {{
            margin: 0.35rem 0 0.7rem;
        }}

        .ltm-step-eyebrow {{
            display: inline-block;
            margin-bottom: 0.18rem;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--ltm-accent-alt);
        }}

        .ltm-step-title {{
            font-size: 2.05rem;
            line-height: 1.08;
            font-weight: 800;
            color: var(--ltm-text);
            margin-bottom: 0.24rem;
        }}

        .ltm-step-copy {{
            max-width: 64rem;
            font-size: 0.94rem;
            line-height: 1.5;
            color: var(--ltm-muted);
        }}

        .ltm-section-label {{
            margin: 0.15rem 0 0.4rem;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--ltm-muted);
        }}

        .ltm-card-title-strong {{
            font-size: 0.98rem;
            font-weight: 800;
            color: var(--ltm-text);
            margin-bottom: 0.28rem;
            line-height: 1.2;
        }}

        .ltm-card-copy {{
            font-size: 0.84rem;
            color: var(--ltm-muted);
            line-height: 1.48;
        }}

        .ltm-card-list {{
            margin: 0.12rem 0 0;
            padding-left: 1rem;
            color: var(--ltm-muted);
            font-size: 0.84rem;
            line-height: 1.55;
        }}

        .ltm-card-list li {{
            margin-bottom: 0.12rem;
        }}

        .ltm-status-card {{
            border: 1px solid var(--ltm-border);
            border-radius: 12px;
            background: linear-gradient(180deg, var(--ltm-card), var(--ltm-card-alt));
            padding: 0.72rem 0.8rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }}

        .ltm-status-kicker {{
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--ltm-muted);
            margin-bottom: 0.18rem;
        }}

        .ltm-status-title {{
            font-size: 1rem;
            font-weight: 800;
            color: var(--ltm-text);
            margin-bottom: 0.22rem;
            line-height: 1.2;
        }}

        .ltm-status-copy {{
            font-size: 0.84rem;
            color: var(--ltm-muted);
            line-height: 1.45;
        }}

        .ltm-status-copy strong {{
            color: var(--ltm-text);
            font-weight: 700;
        }}

        div[data-testid="stMetric"] {{
            border: 1px solid var(--ltm-border);
            border-radius: 12px;
            padding: 0.5rem 0.6rem;
            background: var(--ltm-card);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }}

        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input {{
            background: var(--ltm-card-alt) !important;
            border: 1px solid var(--ltm-border) !important;
            border-radius: 10px !important;
        }}

        .stSlider [data-baseweb="slider"] {{
            padding-top: 0.15rem;
            padding-bottom: 0.1rem;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 12px !important;
            border: 1px solid var(--ltm-border) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_selector() -> str:
    """
    Renderiza el selector de tema en la barra lateral.

    Returns:
        Nombre del tema seleccionado ('claro' u 'oscuro')
    """
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "oscuro"

    st.markdown("### Apariencia")
    theme_choice = st.radio(
        "Modo de color",
        options=("Claro", "Oscuro"),
        index=0 if st.session_state.ui_theme == "claro" else 1,
        key="theme_selector",
    )

    selected_key = "claro" if theme_choice == "Claro" else "oscuro"
    if st.session_state.ui_theme != selected_key:
        st.session_state.ui_theme = selected_key
        st.rerun()

    return selected_key


def render_stepper(current_step: int) -> None:
    """
    Renderiza una barra de navegacion por pasos (stepper) elegante.
    """
    steps = [
        ("1", "Cargar", "📥"),
        ("2", "Clasificar", "🔍"),
        ("3", "Procesar", "🏗️"),
    ]
    
    cols = st.columns(len(steps))
    for i, (num, label, icon) in enumerate(steps):
        step_idx = i + 1
        is_active = step_idx == current_step
        is_done = step_idx < current_step
        
        status_class = "ltm-stepper-active" if is_active else ("ltm-stepper-done" if is_done else "ltm-stepper-pending")
        
        cols[i].markdown(
            f"""
            <div class="ltm-stepper-item {status_class}">
                <div class="ltm-stepper-icon">{icon if not is_done else "✅"}</div>
                <div class="ltm-stepper-label">
                    <span class="ltm-stepper-num">Paso {num}</span><br>
                    <span class="ltm-stepper-text">{label}</span>
                </div>
            </div>
            <style>
            .ltm-stepper-item {{
                display: flex;
                align-items: center;
                gap: 0.8rem;
                padding: 0.7rem 1rem;
                border-radius: 12px;
                border: 1px solid var(--ltm-border);
                background: var(--ltm-card);
                transition: all 0.2s ease;
                opacity: 0.6;
            }}
            .ltm-stepper-active {{
                opacity: 1;
                border-color: var(--ltm-accent);
                background: var(--ltm-accent-soft);
                box-shadow: 0 4px 12px var(--ltm-shadow);
                transform: translateY(-1px);
            }}
            .ltm-stepper-done {{
                opacity: 0.9;
                border-color: #10b981;
                background: rgba(16, 185, 129, 0.05);
            }}
            .ltm-stepper-icon {{
                font-size: 1.2rem;
                width: 32px;
                height: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: var(--ltm-soft-bg);
                border-radius: 8px;
            }}
            .ltm-stepper-active .ltm-stepper-icon {{
                background: var(--ltm-accent);
                color: white;
            }}
            .ltm-stepper-num {{
                font-size: 0.7rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: var(--ltm-muted);
            }}
            .ltm-stepper-active .ltm-stepper-num {{
                color: var(--ltm-accent);
            }}
            .ltm-stepper-text {{
                font-size: 0.9rem;
                font-weight: 700;
                color: var(--ltm-text);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
