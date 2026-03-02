import base64
import io
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from dash import Input, Output, State, dash_table, dcc, html

from refactored_project.src.core.model_factory import DEFAULT_MODEL_NAME, build_model, model_weight_name


def normalize_sequence(seq_raw):
    seq_len = len(seq_raw)
    seq_out = np.zeros((seq_len, 1, 12, 8), dtype=np.float32)
    for i in range(seq_len):
        frame = seq_raw[i]
        d_min, d_max = frame.min(), frame.max()
        if d_max - d_min > 1e-6:
            frame = (frame - d_min) / (d_max - d_min)
        else:
            frame = frame - d_min
        seq_out[i, 0] = frame.reshape(12, 8)
    return seq_out


def stats_sequence(seq_raw):
    seq_len = len(seq_raw)
    stats = np.zeros((seq_len, 3), dtype=np.float32)
    for i in range(seq_len):
        fr = seq_raw[i]
        stats[i] = np.array([np.mean(fr), np.max(fr), np.std(fr)], dtype=np.float32)
    return stats


REFACTORED_ROOT = Path(__file__).resolve().parents[2]
ROOT = REFACTORED_ROOT.parent
LEGACY_ROOT = REFACTORED_ROOT / "legacy_mirror"


def resolve_output_dir() -> Path:
    env_out = os.environ.get("OUTPUT_DIR", "").strip()
    if env_out:
        return Path(env_out)
    return REFACTORED_ROOT / "logs" / "outputs"


def read_csv_from_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="ignore")))
    return df.iloc[:, -96:].values.astype(np.float32)


def load_model():
    output_dir = resolve_output_dir()
    model_name = os.environ.get("INFER_MODEL", DEFAULT_MODEL_NAME)
    model_path = output_dir / model_weight_name(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, seq_len=10).to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()
    return model, device


def collect_archive_rows(page=1, page_size=20):
    rows = []
    if LEGACY_ROOT.exists():
        for p in LEGACY_ROOT.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(LEGACY_ROOT)).replace("\\", "/")
                rows.append(
                    {
                        "path": rel,
                        "size_kb": round(p.stat().st_size / 1024, 2),
                        "suffix": p.suffix.lower(),
                    }
                )
    rows.sort(key=lambda x: x["path"])
    total = len(rows)
    start = max(0, (page - 1) * page_size)
    end = min(total, start + page_size)
    return rows[start:end], total


def collect_metrics():
    out_dir = resolve_output_dir()
    if not out_dir.exists():
        return None
    ds = out_dir / "file3_metrics_refactor_dualstream.json"
    ns = out_dir / "file3_metrics_refactor_nolstm.json"
    if not ds.exists() or not ns.exists():
        return None
    import json

    with open(ds, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(ns, "r", encoding="utf-8") as f:
        m2 = json.load(f)
    return [m1, m2]


def gallery_images(page=1, page_size=8):
    img_ext = {".png", ".jpg", ".jpeg"}
    dirs = [
        LEGACY_ROOT / "GitHub_Docs_Package" / "final-result",
        LEGACY_ROOT / "failed_attempts" / "active_learning_results" / "file3_eval",
        REFACTORED_ROOT / "logs" / "outputs",
    ]
    imgs = []
    for d in dirs:
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in img_ext:
                    imgs.append(p)
    imgs = sorted(imgs, key=lambda x: str(x))
    total = len(imgs)
    start = max(0, (page - 1) * page_size)
    end = min(total, start + page_size)
    return imgs[start:end], total


def image_to_data_uri(path: Path):
    try:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


MODEL, DEVICE = load_model()


def help_panel(title, bullets):
    return html.Div(
        [
            html.B(title),
            html.Ul([html.Li(item) for item in bullets], style={"margin": "6px 0 0 18px"}),
        ],
        style={"backgroundColor": "#f8f9fa", "border": "1px solid #e5e7eb", "padding": "10px", "borderRadius": "8px", "marginBottom": "12px"},
    )


app = dash.Dash(__name__)
app.title = "Tactile Nodule Plotly GUI"

app.layout = html.Div(
    [
        html.H3("触觉结节检测研究工作台（Plotly分页版）"),
        dcc.Tabs(
            id="main-tabs",
            value="tab-live",
            children=[
                dcc.Tab(
                    label="实时推理",
                    value="tab-live",
                    children=[
                        html.Br(),
                        help_panel(
                            "本页参数说明",
                            [
                                "上传CSV：导入单个触觉序列文件，系统会自动读取最后96列为12×8热图数据。",
                                "frame-slider（滑条数值）：表示当前查看的是第几个时间帧（从0开始）。",
                                "只有当滑条>=9时才会执行推理，因为模型需要连续10帧窗口。",
                                "结节概率：分类头输出经sigmoid后的概率，越接近1代表越可能存在结节。",
                                "预测大小/预测深度：回归头输出的连续值，单位为cm。",
                            ],
                        ),
                        dcc.Upload(id="upload-csv", children=html.Button("上传CSV"), multiple=False),
                        html.Br(),
                        dcc.Slider(id="frame-slider", min=0, max=9, value=9, step=1),
                        html.Br(),
                        html.Div(id="pred-text", style={"fontSize": "16px", "fontWeight": "bold"}),
                        dcc.Graph(id="heatmap"),
                        dcc.Store(id="data-store"),
                    ],
                ),
                dcc.Tab(
                    label="模型指标对比",
                    value="tab-metrics",
                    children=[
                        html.Br(),
                        help_panel(
                            "字段含义说明",
                            [
                                "accuracy：总体分类正确率（TP+TN占比）。",
                                "precision：预测为正样本中真正例的比例（查准率）。",
                                "recall：真实正样本被检出的比例（查全率）。",
                                "f1：precision与recall的调和平均，更平衡地衡量分类性能。",
                                "mae_size / mae_depth：仅在命中正样本时统计的大小/深度平均绝对误差（越小越好）。",
                                "表格与柱状图均来自统一评估脚本输出，便于主线与消融模型横向对比。",
                            ],
                        ),
                        dcc.Graph(id="metrics-bar"),
                        dash_table.DataTable(id="metrics-table", page_size=10),
                    ],
                ),
                dcc.Tab(
                    label="成果文件分页浏览",
                    value="tab-archive",
                    children=[
                        html.Br(),
                        help_panel(
                            "本页字段说明",
                            [
                                "archive-page：要查看的页码（每页20条）。",
                                "path：文件相对路径，用于定位具体产物或脚本。",
                                "size_kb：文件大小（KB），可用于快速判断是否为空文件。",
                                "suffix：文件后缀（如 .json/.png/.pth）。",
                                "用途：快速确认重构交付物是否齐全、是否落盘成功。",
                            ],
                        ),
                        dcc.Input(id="archive-page", type="number", value=1, min=1, step=1),
                        html.Button("刷新", id="archive-refresh", n_clicks=0),
                        html.Div(id="archive-total"),
                        dash_table.DataTable(id="archive-table", page_size=20, style_cell={"textAlign": "left", "fontSize": 12}),
                    ],
                ),
                dcc.Tab(
                    label="图像画廊分页",
                    value="tab-gallery",
                    children=[
                        html.Br(),
                        help_panel(
                            "本页使用说明",
                            [
                                "gallery-page：图片页码（每页8张）。",
                                "展示来源：历史结果目录 + 当前重构输出目录。",
                                "用途：快速核查对比图、流程图、评估图是否已生成且内容正常。",
                            ],
                        ),
                        dcc.Input(id="gallery-page", type="number", value=1, min=1, step=1),
                        html.Button("刷新", id="gallery-refresh", n_clicks=0),
                        html.Div(id="gallery-total"),
                        html.Div(id="gallery-grid"),
                    ],
                ),
            ],
        ),
    ],
    style={"maxWidth": "960px", "margin": "auto"},
)


@app.callback(Output("data-store", "data"), Output("frame-slider", "max"), Input("upload-csv", "contents"))
def on_upload(contents):
    if not contents:
        return None, 9
    data = read_csv_from_contents(contents)
    max_idx = max(9, len(data) - 1)
    return data.tolist(), max_idx


@app.callback(Output("heatmap", "figure"), Output("pred-text", "children"), Input("frame-slider", "value"), State("data-store", "data"))
def update_view(frame_idx, data_store):
    if not data_store:
        fig = px.imshow(np.zeros((12, 8)), color_continuous_scale="Turbo", origin="lower", aspect="auto")
        return fig, "请先上传CSV。"

    data = np.array(data_store, dtype=np.float32)
    frame_idx = int(frame_idx)
    frame = data[frame_idx].reshape(12, 8)
    fig = px.imshow(frame, color_continuous_scale="Turbo", origin="lower", aspect="auto", title=f"Frame {frame_idx}")

    pred_text = "当前帧不足10帧窗口，暂无法推理。"
    if frame_idx >= 9:
        seq_raw = data[frame_idx - 9 : frame_idx + 1]
        seq_norm = normalize_sequence(seq_raw)
        stats = stats_sequence(seq_raw)
        x = torch.tensor(seq_norm).unsqueeze(0).to(DEVICE)
        s = torch.tensor(stats).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob, size, depth = MODEL(x, s)
        p = float(torch.sigmoid(prob).item())
        pred_text = f"结节概率: {p:.4f} | 预测大小: {float(size.item()):.3f} cm | 预测深度: {float(depth.item()):.3f} cm"

    return fig, pred_text


@app.callback(Output("metrics-bar", "figure"), Output("metrics-table", "data"), Input("main-tabs", "value"))
def update_metrics(_):
    metrics = collect_metrics()
    if not metrics:
        fig = go.Figure()
        fig.update_layout(title="尚未找到评估结果，请先运行 evaluate_file3.py")
        return fig, []

    df = pd.DataFrame(metrics)
    comp = pd.DataFrame(
        {
            "model": df["model"],
            "accuracy": df["accuracy"],
            "f1": df["f1"],
            "mae_size": df["mae_size"],
            "mae_depth": df["mae_depth"],
        }
    )
    bar_df = comp.melt(id_vars=["model"], value_vars=["accuracy", "f1", "mae_size", "mae_depth"], var_name="metric", value_name="value")
    fig = px.bar(bar_df, x="metric", y="value", color="model", barmode="group", title="DualStream vs NoLSTM 统一口径对比")
    return fig, comp.to_dict("records")


@app.callback(
    Output("archive-table", "data"),
    Output("archive-total", "children"),
    Input("archive-refresh", "n_clicks"),
    State("archive-page", "value"),
)
def update_archive(_, page):
    page = int(page or 1)
    rows, total = collect_archive_rows(page=page, page_size=20)
    return rows, f"总文件数: {total} | 当前页: {page}"


@app.callback(
    Output("gallery-grid", "children"),
    Output("gallery-total", "children"),
    Input("gallery-refresh", "n_clicks"),
    State("gallery-page", "value"),
)
def update_gallery(_, page):
    page = int(page or 1)
    imgs, total = gallery_images(page=page, page_size=8)
    cards = []
    for p in imgs:
        rel = str(p).replace("\\", "/")
        src = image_to_data_uri(p)
        cards.append(
            html.Div(
                [
                    html.Div(Path(rel).name, style={"fontSize": 12, "marginBottom": 6}),
                    html.Img(src=src, style={"width": "100%", "maxHeight": "220px", "objectFit": "contain", "border": "1px solid #ddd"}),
                ],
                style={"width": "48%", "display": "inline-block", "margin": "1%"},
            )
        )
    return cards, f"总图片数: {total} | 当前页: {page}"


if __name__ == "__main__":
    print("Plotly GUI running at http://127.0.0.1:8052/ (Ctrl+C to stop)")
    app.run(debug=False, port=8052)
