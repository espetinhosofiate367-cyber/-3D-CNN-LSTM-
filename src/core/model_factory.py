from refactored_project.src.core.models import DualStream3DCNNLSTM, DualStream3DCNNNoLSTM


DEFAULT_MODEL_NAME = "dualstream"


def build_model(model_name: str, seq_len: int = 10):
    if model_name == "dualstream":
        return DualStream3DCNNLSTM(seq_len=seq_len)
    if model_name == "nolstm":
        return DualStream3DCNNNoLSTM(seq_len=seq_len)
    raise ValueError(f"Unsupported model_name: {model_name}")


def model_weight_name(model_name: str):
    if model_name == "dualstream":
        return "dualstream_3dcnn_lstm_active_refactor.pth"
    if model_name == "nolstm":
        return "dualstream_3dcnn_nolstm_active_refactor.pth"
    raise ValueError(f"Unsupported model_name: {model_name}")
