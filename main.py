import numpy as np
from scipy.integrate import solve_ivp
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

TAU = 2 * np.pi

def grid_field(f, xlim=(-4, 4), ylim=(-4, 4), n=25):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    U, V = f(X, Y)
    s = np.hypot(U, V) + 1e-9
    return X, Y, U/s, V/s, s

def integrate_trajectories(f, seeds, t_span=(0, 40), dt=0.02):
    curves = []
    for x0, y0 in seeds:
        def rhs(t, z):
            u, v = f(z[0], z[1])
            return [u, v]
        try:
            sol = solve_ivp(rhs, t_span, [x0, y0], max_step=dt, rtol=1e-6, atol=1e-8)
            curves.append((sol.y[0], sol.y[1]))
        except Exception:
            continue
    return curves

def linear_system(a, b, c, d):
    def f(X, Y):
        return a * X + b * Y, c * X + d * Y
    return f

def linear_classification(a, b, c, d):
    tr = a + d
    det = a * d - b * c
    disc = tr**2 - 4 * det
    if det < 0:
        k = "Сідло"
    else:
        if disc < 0:
            if abs(tr) < 1e-8:
                k = "Центр"
            elif tr < 0:
                k = "Стійкий фокус"
            else:
                k = "Нестійкий фокус"
        elif disc > 0:
            if tr < 0:
                k = "Стійкий вузол"
            elif tr > 0:
                k = "Нестійкий вузол"
            else:
                k = "Вироджений випадок"
        else:
            k = "Вироджений вузол/центр"
    return k, tr, det, disc

def Z_linear(phi, a, b, c, d):
    s, cph = np.sin(phi), np.cos(phi)
    return d * s**2 + (b + c) * s * cph + a * cph**2

def N_linear(phi, a, b, c, d):
    s, cph = np.sin(phi), np.cos(phi)
    return -b * s**2 + (d - a) * s * cph + c * cph**2

def integral_Z_over_N(a, b, c, d, n=4000):
    ph = np.linspace(0, TAU, n, endpoint=False)
    Z = Z_linear(ph, a, b, c, d)
    N = N_linear(ph, a, b, c, d)
    m = np.abs(N) > 1e-6
    return float(np.trapz(Z[m]/N[m], ph[m]))

def homog_polar(A, B, m=2):
    def f(X, Y):
        r = np.hypot(X, Y) + 1e-9
        phi = np.arctan2(Y, X)
        rdot = (r**m) * A(phi)
        phidot = (r**(m-1)) * B(phi)
        U = rdot * np.cos(phi) - r * np.sin(phi) * phidot
        V = rdot * np.sin(phi) + r * np.cos(phi) * phidot
        return U, V
    return f

def ex_center_linear():
    return linear_system(0.0, -1.0, 1.0, 0.0)

def ex_stable_focus():
    a = 0.6
    return linear_system(-a, -1.0, 1.0, -a)

def ex_unstable_focus():
    a = 0.6
    return linear_system(a, -1.0, 1.0, a)

def ex_quadratic_rays():
    def f(X, Y):
        return X*X - Y*Y, 2*X*Y
    return f

def ex_cubic_three_rays():
    def f(X, Y):
        return X**3 - 3*X*Y**2, 3*X**2*Y - Y**3
    return f

def ex_polar_center_m2():
    eps = 0.25
    return homog_polar(lambda p: eps*np.sin(2*p), lambda p: 1+0*p, m=2)

def ex_polar_stable_focus_m3():
    a = -0.2
    return homog_polar(lambda p: 0*p + a, lambda p: 1+0*p, m=3)

def ex_polar_unstable_focus_m3():
    a = 0.2
    return homog_polar(lambda p: 0*p + a, lambda p: 1+0*p, m=3)

def ex_quasi_saddle_spiral():
    k = 0.15
    def f(X, Y):
        P = X**2 - Y**2 - k*X*Y - 0.1*Y
        Q = 2*X*Y + k*(X**2+Y**2) + 0.1*X
        return P, Q
    return f

def ex_swirl_quartic():
    def f(X, Y):
        r2 = X**2 + Y**2
        return -(X*r2) + Y, -(Y*r2) - X
    return f

PRESETS = {
    "Лінійна: центр": ex_center_linear,
    "Лінійна: стійкий фокус": ex_stable_focus,
    "Лінійна: нестійкий фокус": ex_unstable_focus,
    "Однорідна m=2: промені": ex_quadratic_rays,
    "Однорідна m=3: три промені": ex_cubic_three_rays,
    "Полярна m=2: центр": ex_polar_center_m2,
    "Полярна m=3: стійкий фокус": ex_polar_stable_focus_m3,
    "Полярна m=3: нестійкий фокус": ex_polar_unstable_focus_m3,
    "Квазі-однорідна: сідло-спіраль": ex_quasi_saddle_spiral,
    "Нелінійний вихор m=4": ex_swirl_quartic,
}

def figure_phase_portrait(f, xlim=(-4, 4), ylim=(-4, 4), title="", template="plotly_dark", show_heat=True):
    X, Y, U, V, S = grid_field(f, xlim=xlim, ylim=ylim, n=35)
    fig = go.Figure()
    if show_heat:
        fig.add_trace(go.Heatmap(x=X[0], y=Y[:,0], z=S, colorscale="Turbo", showscale=False, opacity=0.55))
        fig.add_trace(go.Contour(x=X[0], y=Y[:,0], z=S, contours=dict(showlines=True, coloring="lines"), showscale=False, line=dict(width=1)))
    fig.add_trace(go.Cone(x=X.flatten(), y=Y.flatten(), z=0*X.flatten(), u=U.flatten(), v=V.flatten(), w=0*U.flatten(), sizemode="absolute", sizeref=0.55, showscale=False, anchor="tip", name="field", opacity=0.85))
    seeds = []
    for r in np.linspace(0.6, 3.5, 9):
        for ang in np.linspace(0, TAU, 16, endpoint=False):
            seeds.append((r*np.cos(ang), r*np.sin(ang)))
    curves = integrate_trajectories(lambda x, y: f(np.array([x]), np.array([y])), seeds, t_span=(0, 70), dt=0.03)
    for x, y in curves:
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line=dict(width=2), hoverinfo="none", showlegend=False))
    fig.update_layout(title=title, height=820, margin=dict(l=30, r=30, t=60, b=30), xaxis=dict(scaleanchor="y", scaleratio=1, range=[xlim[0], xlim[1]], zeroline=True), yaxis=dict(range=[ylim[0], ylim[1]], zeroline=True), template=template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def eigen_overlay(a, b, c, d, R=3.8):
    M = np.array([[a, b], [c, d]])
    w, V = np.linalg.eig(M)
    lines = []
    for i in range(2):
        v = np.real(V[:, i])
        v = v/np.linalg.norm(v)
        x = np.array([-R, R])*v[0]
        y = np.array([-R, R])*v[1]
        lines.append(go.Scatter(x=x, y=y, mode="lines", line=dict(width=3, dash="dot"), showlegend=False))
    return lines, w

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])
app.title = "Тема 4 — Однорідні та квазіоднорідні системи"

glass = {"backdropFilter":"blur(10px)","WebkitBackdropFilter":"blur(10px)","background":"rgba(255,255,255,0.10)","border":"1px solid rgba(255,255,255,0.25)","boxShadow":"0 10px 30px rgba(0,0,0,.25)","borderRadius":"20px","padding":"18px"}

linear_controls = dbc.Card([
    html.H4("Лінійна система x' = ax + by, y' = cx + dy", className="mb-3"),
    dbc.Row([dbc.Col([html.Small("a"), dcc.Slider(-2, 2, 0.01, value=-0.2, id="a")], md=6), dbc.Col([html.Small("b"), dcc.Slider(-2, 2, 0.01, value=-1.0, id="b")], md=6)]),
    dbc.Row([dbc.Col([html.Small("c"), dcc.Slider(-2, 2, 0.01, value=1.0, id="c")], md=6), dbc.Col([html.Small("d"), dcc.Slider(-2, 2, 0.01, value=-0.2, id="d")], md=6)]),
    html.Div(dbc.Button("Оновити", id="btn-update-lin", color="primary", className="mt-3 w-100")),
    html.Div(id="lin-info", className="mt-3")
], style=glass)

preset_controls = dbc.Card([
    html.H4("Приклади однорідних/квазіоднорідних", className="mb-3"),
    dcc.Dropdown(options=[{"label": k, "value": k} for k in PRESETS.keys()], value="Лінійна: центр", id="preset", clearable=False),
    html.Div(dbc.Button("Побудувати", id="btn-build-preset", color="primary", className="mt-3 w-100")),
    html.Div(dbc.Switch(id="dark", label="Темний режим графіків", value=True), className="mt-3")
], style=glass)

lab_controls = dbc.Card([
    html.H4("Futuristic Lab", className="mb-3"),
    dcc.Dropdown(options=[{"label": k, "value": k} for k in PRESETS.keys()], value="Нелінійний вихор m=4", id="lab-preset", clearable=False),
    dbc.Row([dbc.Col([html.Small("К-сть частинок"), dcc.Slider(50, 1500, 50, value=600, id="npart")], md=8), dbc.Col([html.Small("Δt"), dcc.Slider(0.005, 0.05, 0.005, value=0.02, id="dt")], md=4)]),
    dbc.Row([dbc.Col(dbc.Button("Seed", id="seed", color="secondary", className="w-100"), md=4), dbc.Col(dbc.Button("Start/Stop", id="run", color="success", className="w-100"), md=4), dbc.Col(dbc.Button("Reset", id="reset", color="danger", className="w-100"), md=4)]),
    dcc.Interval(id="tick", interval=40, disabled=True),
    dcc.Store(id="store", data={})
], style=glass)

app.layout = html.Div([
    html.Div(style={"position":"fixed","inset":"0","background":"radial-gradient(1200px 600px at 20% 10%, rgba(14,165,233,.35), transparent 60%), radial-gradient(1000px 800px at 80% 0%, rgba(168,85,247,.25), transparent 60%), linear-gradient(135deg,#0f172a,#1f2937 40%,#0ea5e9 140%)","zIndex":"-1"}),
    dbc.Container([
        html.Div([html.H2("Однорідні та квазіоднорідні рівняння на площині", className="text-white fw-bold"), html.P("Фазові портрети, класифікація, демонстрації та анімації", className="text-white-50")], className="py-3"),
        dcc.Tabs([
            dcc.Tab(label="Лінійна система", children=[dbc.Row([dbc.Col(linear_controls, md=4), dbc.Col(dcc.Loading(dcc.Graph(id="fig-linear"), type="dot"), md=8)], className="gy-4")]),
            dcc.Tab(label="Приклади", children=[dbc.Row([dbc.Col(preset_controls, md=4), dbc.Col(dcc.Loading(dcc.Graph(id="fig-preset"), type="dot"), md=8)], className="gy-4")]),
            dcc.Tab(label="ALL-IN-ONE", children=[dbc.Row([dbc.Col(dbc.Card([html.H4("Вибір прикладу"), dcc.Dropdown(options=[{"label": k, "value": k} for k in PRESETS.keys()], value="Полярна m=2: центр", id="demo-preset", clearable=False), html.Div(dbc.Button("Показати", id="btn-build-demo", className="mt-3 w-100", color="primary"))], style=glass), md=4), dbc.Col(dcc.Loading(dcc.Graph(id="fig-demo"), type="dot"), md=8)], className="gy-4"), dbc.Row([dbc.Col(dcc.Loading(dcc.Graph(id="fig-rt"), type="dot"), md=6), dbc.Col(dcc.Loading(dcc.Graph(id="fig-phit"), type="dot"), md=6)], className="gy-4")]) ,
            dcc.Tab(label="FUTURISTIC LAB", children=[dbc.Row([dbc.Col(lab_controls, md=4), dbc.Col(dcc.Loading(dcc.Graph(id="fig-lab"), type="dot"), md=8)], className="gy-4")])
        ], persistence=True, style=glass)
    ], fluid=True)
])

@app.callback(Output("fig-linear", "figure"), Output("lin-info", "children"), Input("btn-update-lin", "n_clicks"), State("a", "value"), State("b", "value"), State("c", "value"), State("d", "value"), prevent_initial_call=False)
def update_linear(_, a, b, c, d):
    f = linear_system(a, b, c, d)
    fig = figure_phase_portrait(lambda X, Y: f(X, Y), title="Лінійна система", template="plotly_dark", show_heat=True)
    kind, tr, det, disc = linear_classification(a, b, c, d)
    I = integral_Z_over_N(a, b, c, d)
    lines, w = eigen_overlay(a, b, c, d)
    for L in lines:
        fig.add_trace(L)
    pills = dbc.Row([dbc.Col(dbc.Badge(f"{kind}", color="info", pill=True, className="me-1"), md="auto"), dbc.Col(dbc.Badge(f"tr={tr:.3f}", color="secondary", pill=True, className="me-1"), md="auto"), dbc.Col(dbc.Badge(f"det={det:.3f}", color="secondary", pill=True, className="me-1"), md="auto"), dbc.Col(dbc.Badge(f"Δ={disc:.3f}", color="secondary", pill=True, className="me-1"), md="auto"), dbc.Col(dbc.Badge(f"∫Z/N≈{I:.4f}", color="warning", pill=True), md="auto")], className="g-2")
    return fig, pills

@app.callback(Output("fig-preset", "figure"), Input("btn-build-preset", "n_clicks"), State("preset", "value"), State("dark", "value"), prevent_initial_call=False)
def build_preset(_, name, dark):
    f = PRESETS[name]()
    t = "plotly_dark" if dark else "plotly_white"
    return figure_phase_portrait(lambda X, Y: f(X, Y), title=f"Приклад: {name}", template=t, show_heat=True)

@app.callback(Output("fig-demo", "figure"), Output("fig-rt", "figure"), Output("fig-phit", "figure"), Input("btn-build-demo", "n_clicks"), State("demo-preset", "value"), prevent_initial_call=False)
def build_demo(_, name):
    f = PRESETS[name]()
    fig = figure_phase_portrait(lambda X, Y: f(X, Y), title=f"Демонстрація: {name}", template="plotly_dark", show_heat=False)
    def rhs(t, z):
        u, v = f(z[0], z[1])
        return [u, v]
    sol = solve_ivp(rhs, (0, 60), [2.5, 0.8], max_step=0.02)
    x, y, t = sol.y[0], sol.y[1], sol.t
    r = np.hypot(x, y)
    phi = np.unwrap(np.arctan2(y, x))
    fr = go.Figure(go.Scatter(x=t, y=r, mode="lines"))
    fr.update_layout(title="r(t)", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
    fphi = go.Figure(go.Scatter(x=t, y=phi, mode="lines"))
    fphi.update_layout(title="φ(t)", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig, fr, fphi

@app.callback(Output("store", "data"), Output("tick", "disabled"), Input("seed", "n_clicks"), Input("run", "n_clicks"), Input("reset", "n_clicks"), State("store", "data"), State("lab-preset", "value"), State("npart", "value"), prevent_initial_call=False)
def lab_state(seed, run, reset, data, name, n):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, True
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    if trig == "seed" or data == {}:
        rng = np.random.default_rng(0)
        xs = rng.uniform(-3.5, 3.5, int(n)).tolist()
        ys = rng.uniform(-3.5, 3.5, int(n)).tolist()
        return {"x": xs, "y": ys, "run": False, "preset": name}, True
    if trig == "run":
        data["run"] = not data.get("run", False)
        return data, not data["run"]
    if trig == "reset":
        return {}, True
    return data, not data.get("run", False)

@app.callback(Output("fig-lab", "figure"), Input("tick", "n_intervals"), State("store", "data"), State("lab-preset", "value"), State("dt", "value"), prevent_initial_call=False)
def render_lab(_, data, name, dt):
    f = PRESETS[name]()
    fig = figure_phase_portrait(lambda X, Y: f(X, Y), title=f"Futuristic Lab: {name}", template="plotly_dark", show_heat=True)
    if data and "x" in data:
        xs = np.array(data["x"]) ; ys = np.array(data["y"]) ; r = np.hypot(xs, ys)
        for _ in range(2):
            U, V = f(xs, ys)
            xs = xs + dt * U
            ys = ys + dt * V
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="markers", marker=dict(size=4, opacity=0.85), showlegend=False))
        data["x"], data["y"] = xs.tolist(), ys.tolist()
    return fig

if __name__ == "__main__":
    app.run(debug=False)
