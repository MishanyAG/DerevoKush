import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from collections import defaultdict, deque


st.set_page_config(page_title="Динамическое дерево отказов", layout="wide")


# ============================================================
# ДАННЫЕ ДЕРЕВА
# ============================================================

NODE_LABELS: Dict[str, str] = {
    "G1": "Критические последствия пожара на предприятии",
    "G2": "Гибель людей",
    "G3": "Травмирование людей",
    "G4": "Крупный прямой материальный ущерб",
    "G5": "Критическое повреждение производственного оборудования",
    "G6": "Вторичные опасные события при пожаре",
    "G7": "Позднее обнаружение пожара",
    "G8": "Невозможность безопасной эвакуации",
    "G9": "Высокая интенсивность опасных факторов пожара",
    "G10": "Воздействие открытого пламени",
    "G11": "Воздействие дыма и токсичных газов",
    "G12": "Воздействие взрывных и ударных факторов",
    "G13": "Горение основных производственных помещений",
    "G14": "Уничтожение материальных ценностей",
    "G15": "Повреждение инфраструктуры предприятия",
    "G16": "Авария электрооборудования",
    "G17": "Авария газового оборудования",
    "G18": "Повреждение теплогенерирующих установок",
    "G19": "Повреждение печей",
    "G20": "Взрыв",
    "G21": "Нарушения при огневых работах",
    "G22": "Недостаточная эффективность тушения",
    "F1": "Отсутствие своевременного контроля очага",
    "F2": "Неисправность средств обнаружения пожара",
    "F3": "Несвоевременное сообщение о пожаре",
    "F4": "Плохое знание персоналом путей эвакуации",
    "F5": "Загромождение путей эвакуации",
    "F6": "Блокирование выходов продуктами горения",
    "F7": "Высокая температура в зоне пожара",
    "F8": "Токсичность продуктов горения",
    "F9": "Быстрое задымление помещений",
    "F10": "Отсутствие средств индивидуальной защиты",
    "F11": "Близость персонала к очагу пожара",
    "F12": "Высокая концентрация дыма",
    "F13": "Недостаточная вентиляция и дымоудаление",
    "F14": "Взрыв в зоне пожара",
    "F15": "Разрушение конструкций или оборудования",
    "F16": "Неисправность производственного оборудования",
    "F17": "Нарушение технологического процесса",
    "F18": "Нарушение правил эксплуатации электрооборудования",
    "F19": "Самовозгорание веществ и материалов",
    "F20": "Поджог",
    "F21": "Грозовые разряды",
    "F22": "Повреждение электросетей",
    "F23": "Повреждение строительных конструкций",
    "F24": "Высокая стоимость и сложность восстановления оборудования",
    "F25": "Короткое замыкание",
    "F26": "Перегрузка электросети",
    "F27": "Неправильная эксплуатация электрооборудования",
    "F28": "Утечка газа",
    "F29": "Нарушение правил эксплуатации газового оборудования",
    "F30": "Нарушение правил эксплуатации теплогенерирующих установок",
    "F31": "Перегрев оборудования",
    "F32": "Нарушение правил устройства и эксплуатации печей",
    "F33": "Разрушение элементов печи",
    "F34": "Накопление взрывоопасной смеси",
    "F35": "Воспламенение взрывоопасной смеси",
    "F36": "Нарушение правил пожарной безопасности при огневых работах",
    "F37": "Нарушение при проведении электрогазосварочных работ",
    "F38": "Недостаточная численность сил пожарной охраны",
    "F39": "Недостаточное количество инструктажей",
    "F40": "Нехватка средств пожаротушения",
}

NODE_PROBABILITIES: Dict[str, float] = {
    "G1": 0.971,
    "G2": 0.610,
    "G3": 0.388,
    "G4": 0.505,
    "G5": 0.592,
    "G6": 0.398,
    "G7": 0.187,
    "G8": 0.279,
    "G9": 0.334,
    "G10": 0.135,
    "G11": 0.199,
    "G12": 0.117,
    "G13": 0.280,
    "G14": 0.106,
    "G15": 0.230,
    "G16": 0.303,
    "G17": 0.190,
    "G18": 0.163,
    "G19": 0.136,
    "G20": 0.003,
    "G21": 0.199,
    "G22": 0.247,
    "F1": 0.08,
    "F2": 0.05,
    "F3": 0.07,
    "F4": 0.10,
    "F5": 0.09,
    "F6": 0.12,
    "F7": 0.11,
    "F8": 0.13,
    "F9": 0.14,
    "F10": 0.06,
    "F11": 0.08,
    "F12": 0.12,
    "F13": 0.09,
    "F14": 0.05,
    "F15": 0.07,
    "F16": 0.09,
    "F17": 0.08,
    "F18": 0.14,
    "F19": 0.06,
    "F20": 0.03,
    "F21": 0.02,
    "F22": 0.08,
    "F23": 0.07,
    "F24": 0.10,
    "F25": 0.11,
    "F26": 0.10,
    "F27": 0.13,
    "F28": 0.08,
    "F29": 0.12,
    "F30": 0.10,
    "F31": 0.07,
    "F32": 0.09,
    "F33": 0.05,
    "F34": 0.06,
    "F35": 0.05,
    "F36": 0.11,
    "F37": 0.10,
    "F38": 0.08,
    "F39": 0.10,
    "F40": 0.09,
}

EDGES: List[Tuple[str, str]] = [
    ("G1", "G2"), ("G1", "G3"), ("G1", "G4"), ("G1", "G5"), ("G1", "G6"),
    ("G2", "G7"), ("G2", "G8"), ("G2", "G9"),
    ("G3", "G10"), ("G3", "G11"), ("G3", "G12"),
    ("G4", "G13"), ("G4", "G14"), ("G4", "G15"),
    ("G5", "G16"), ("G5", "G17"), ("G5", "G18"), ("G5", "G19"),
    ("G6", "G20"), ("G6", "G21"), ("G6", "G22"),
    ("G7", "F1"), ("G7", "F2"), ("G7", "F3"),
    ("G8", "F4"), ("G8", "F5"), ("G8", "F6"),
    ("G9", "F7"), ("G9", "F8"), ("G9", "F9"),
    ("G10", "F10"), ("G10", "F11"),
    ("G11", "F12"), ("G11", "F13"),
    ("G12", "F14"), ("G12", "F15"),
    ("G13", "F16"), ("G13", "F17"), ("G13", "F18"),
    ("G14", "F19"), ("G14", "F20"), ("G14", "F21"),
    ("G15", "F22"), ("G15", "F23"), ("G15", "F24"),
    ("G16", "F25"), ("G16", "F26"), ("G16", "F27"),
    ("G17", "F28"), ("G17", "F29"),
    ("G18", "F30"), ("G18", "F31"),
    ("G19", "F32"), ("G19", "F33"),
    ("G20", "F34"), ("G20", "F35"),
    ("G21", "F36"), ("G21", "F37"),
    ("G22", "F38"), ("G22", "F39"), ("G22", "F40"),
]

NODE_LEVELS: Dict[str, int] = {
    "G1": 0,
    "G2": 1, "G3": 1, "G4": 1, "G5": 1, "G6": 1,
    "G7": 2, "G8": 2, "G9": 2,
    "G10": 2, "G11": 2, "G12": 2,
    "G13": 2, "G14": 2, "G15": 2,
    "G16": 2, "G17": 2, "G18": 2, "G19": 2,
    "G20": 2, "G21": 2, "G22": 2,
    "F1": 3, "F2": 3, "F3": 3,
    "F4": 3, "F5": 3, "F6": 3,
    "F7": 3, "F8": 3, "F9": 3,
    "F10": 3, "F11": 3,
    "F12": 3, "F13": 3,
    "F14": 3, "F15": 3,
    "F16": 3, "F17": 3, "F18": 3,
    "F19": 3, "F20": 3, "F21": 3,
    "F22": 3, "F23": 3, "F24": 3,
    "F25": 3, "F26": 3, "F27": 3,
    "F28": 3, "F29": 3,
    "F30": 3, "F31": 3,
    "F32": 3, "F33": 3,
    "F34": 3, "F35": 3,
    "F36": 3, "F37": 3,
    "F38": 3, "F39": 3, "F40": 3,
}

LEVEL_NODE_ORDER: Dict[int, List[str]] = {
    0: ["G1"],
    1: ["G2", "G3", "G4", "G5", "G6"],
    2: [
        "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17",
        "G18", "G19", "G20", "G21", "G22",
    ],
    3: [
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13",
        "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24",
        "F25", "F26", "F27", "F28", "F29", "F30", "F31", "F32", "F33", "F34", "F35",
        "F36", "F37", "F38", "F39", "F40",
    ],
}


@dataclass
class EdgeState:
    source: str
    target: str
    edge_probability: float
    target_probability: float
    is_active: bool


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def build_node_positions() -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    level_y = {0: 0.93, 1: 0.72, 2: 0.45, 3: 0.12}

    for level, nodes in LEVEL_NODE_ORDER.items():
        count = len(nodes)
        if count == 1:
            x_values = [0.5]
        else:
            x_values = [(i + 1) / (count + 1) for i in range(count)]
        for node, x in zip(nodes, x_values):
            positions[node] = (x, level_y[level])
    return positions


def make_edge_probabilities(
    rng: random.Random,
    edge_mode: str,
    manual_probability: float,
) -> Dict[Tuple[str, str], float]:
    probs: Dict[Tuple[str, str], float] = {}
    for edge in EDGES:
        if edge_mode == "Случайные":
            probs[edge] = round(rng.uniform(0.0, 1.0), 3)
        else:
            probs[edge] = round(float(manual_probability), 3)
    return probs


def evaluate_edges(edge_probabilities: Dict[Tuple[str, str], float]) -> List[EdgeState]:
    states: List[EdgeState] = []
    for source, target in EDGES:
        p_edge = edge_probabilities[(source, target)]
        p_target = NODE_PROBABILITIES[target]
        is_active = p_edge <= p_target
        states.append(
            EdgeState(
                source=source,
                target=target,
                edge_probability=p_edge,
                target_probability=p_target,
                is_active=is_active,
            )
        )
    return states


def build_slice_states(
    time_points: List[float],
    edge_mode: str,
    manual_probability: float,
    seed: int,
) -> Dict[float, List[EdgeState]]:
    result: Dict[float, List[EdgeState]] = {}
    for idx, t in enumerate(time_points):
        rng = random.Random(seed + idx)
        edge_probs = make_edge_probabilities(rng, edge_mode, manual_probability)
        raw_states = evaluate_edges(edge_probs)
        result[t] = enforce_tree_connectivity(raw_states)
    return result


def edge_states_to_df(states: List[EdgeState]) -> pd.DataFrame:
    rows = []
    for state in states:
        rows.append(
            {
                "Источник": state.source,
                "Цель": state.target,
                "Описание цели": NODE_LABELS[state.target],
                "P_дуги": state.edge_probability,
                "P_узла": state.target_probability,
                "Статус": "Активна" if state.is_active else "Неактивна",
            }
        )
    return pd.DataFrame(rows)


def enforce_tree_connectivity(states: List[EdgeState]) -> List[EdgeState]:
    """Если верхняя дуга неактивна, нижние ветки под ней тоже не строятся."""
    state_map = {(s.source, s.target): s for s in states}
    children_map: Dict[str, List[str]] = defaultdict(list)
    for source, target in EDGES:
        children_map[source].append(target)

    reachable_edges = set()
    visited_nodes = {"G1"}
    queue = deque(["G1"])

    while queue:
        node = queue.popleft()
        for child in children_map.get(node, []):
            st_edge = state_map[(node, child)]
            if st_edge.is_active:
                reachable_edges.add((node, child))
                if child not in visited_nodes:
                    visited_nodes.add(child)
                    queue.append(child)

    filtered_states: List[EdgeState] = []
    for s in states:
        filtered_states.append(
            EdgeState(
                source=s.source,
                target=s.target,
                edge_probability=s.edge_probability,
                target_probability=s.target_probability,
                is_active=(s.source, s.target) in reachable_edges,
            )
        )
    return filtered_states


def slice_summary_df(slice_states: Dict[float, List[EdgeState]]) -> pd.DataFrame:
    rows = []
    for t, states in slice_states.items():
        active_count = sum(1 for s in states if s.is_active)
        inactive_count = len(states) - active_count
        g1_incoming_active = any(s.target == "G1" and s.is_active for s in states)
        rows.append(
            {
                "t": t,
                "Активных дуг": active_count,
                "Неактивных дуг": inactive_count,
                "Есть активный вход в G1": "Да" if g1_incoming_active else "Нет",
            }
        )
    return pd.DataFrame(rows)


def plot_graph(states: List[EdgeState], show_inactive: bool) -> go.Figure:
    positions = build_node_positions()
    fig = go.Figure()

    active_edges = [s for s in states if s.is_active]
    inactive_edges = [s for s in states if not s.is_active]

    if show_inactive and inactive_edges:
        for state in inactive_edges:
            x0, y0 = positions[state.source]
            x1, y1 = positions[state.target]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line={"width": 1, "dash": "dot", "color": "rgba(150,150,150,0.4)"},
                    hoverinfo="text",
                    text=f"{state.source} → {state.target}<br>P_дуги={state.edge_probability}<br>P_узла={state.target_probability}<br>Неактивна",
                    showlegend=False,
                )
            )

    for state in active_edges:
        x0, y0 = positions[state.source]
        x1, y1 = positions[state.target]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": 2},
                hoverinfo="text",
                text=f"{state.source} → {state.target}<br>P_дуги={state.edge_probability}<br>P_узла={state.target_probability}<br>Активна",
                showlegend=False,
            )
        )

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_symbols = []

    for node in NODE_LABELS:
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"{node}<br>{NODE_LABELS[node]}<br>P={NODE_PROBABILITIES[node]}"
        )
        is_f = node.startswith("F")
        node_colors.append("#ffb703" if is_f else "#8ecae6")
        node_symbols.append("circle" if is_f else "square")

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker={"size": 20, "color": node_colors, "symbol": node_symbols, "line": {"width": 1, "color": "#333"}},
            text=list(NODE_LABELS.keys()),
            textposition="middle center",
            hoverinfo="text",
            hovertext=node_text,
            showlegend=False,
        )
    )

    fig.update_layout(
        height=900,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def plot_active_edges_over_time(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary_df["t"],
            y=summary_df["Активных дуг"],
            mode="lines+markers",
            name="Активные дуги",
        )
    )
    fig.update_layout(
        title="Динамика количества активных дуг",
        xaxis_title="t",
        yaxis_title="Количество",
        height=450,
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Параметры моделирования")

seed = st.sidebar.number_input(
    "Начальное значение генератора",
    min_value=0,
    value=42,
    step=1,
)

show_inactive = st.sidebar.checkbox("Показывать неактивные дуги", value=True)

max_t = st.sidebar.select_slider(
    "Последний срез времени",
    options=[0.5, 1.0, 1.5, 2.0],
    value=1.0,
)

step = 0.1
time_points = [round(i * step, 3) for i in range(int(max_t / step) + 1)]
selected_t = st.sidebar.select_slider("Выбранный срез t", options=time_points, value=time_points[0])

slice_states = build_slice_states(time_points, "Случайные", 0.3, int(seed))
current_states = slice_states[selected_t]
summary_df = slice_summary_df(slice_states)
current_df = edge_states_to_df(current_states)


# ============================================================
# UI
# ============================================================

st.title("🔥 Динамическое дерево отказов")
st.markdown(
    "Веб-приложение для моделирования срезов дерева отказов во времени на основе фиксированных вероятностей узлов и случайных вероятностей дуг."
)

st.info(
    "Срез t — это отдельное состояние графа при шаге 0.1. На каждом срезе для дуг генерируются случайные вероятности, а нижние ветки строятся только если до них существует активный путь от G1."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Исходные данные",
    "Срез графа",
    "Таблица дуг",
    "Динамика по времени",
])

with tab1:
    st.subheader("Узлы дерева")
    nodes_df = pd.DataFrame(
        {
            "Узел": list(NODE_LABELS.keys()),
            "Описание": [NODE_LABELS[n] for n in NODE_LABELS],
            "P(узла)": [NODE_PROBABILITIES[n] for n in NODE_LABELS],
        }
    )
    st.dataframe(nodes_df, use_container_width=True)

    st.subheader("Дуги дерева")
    base_edges_df = pd.DataFrame(
        {
            "Источник": [s for s, _ in EDGES],
            "Цель": [t for _, t in EDGES],
            "Описание цели": [NODE_LABELS[t] for _, t in EDGES],
        }
    )
    st.dataframe(base_edges_df, use_container_width=True)

with tab2:
    st.subheader(f"Текущий срез графа для t = {selected_t}")
    active_count = int((current_df["Статус"] == "Активна").sum())
    inactive_count = int((current_df["Статус"] == "Неактивна").sum())

    col1, col2 = st.columns(2)
    col1.metric("Активных дуг", active_count)
    col2.metric("Неактивных дуг", inactive_count)

    fig = plot_graph(current_states, show_inactive)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader(f"Таблица дуг для t = {selected_t}")
    st.dataframe(current_df, use_container_width=True)

    csv_data = current_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Скачать таблицу дуг в CSV",
        data=csv_data,
        file_name=f"edge_slice_t_{selected_t}.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader("Изменение структуры графа по времени")
    st.dataframe(summary_df, use_container_width=True)
    fig_dyn = plot_active_edges_over_time(summary_df)
    st.plotly_chart(fig_dyn, use_container_width=True)
