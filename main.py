# main.py
# Orchestrator end-to-end: generate -> preprocess -> train -> evaluate
# + optional UI
#
# Rulare:
#   python main.py --step all
#   python main.py --step generate
#   python main.py --step preprocess
#   python main.py --step train
#   python main.py --step evaluate
#   python main.py --step ui

import argparse
import sys
from pathlib import Path

# --- Config paths (există în proiectul tău) ---
from config.config import (
    RAW_CSV,
    DATA_TRAIN_DIR,
    DATA_VAL_DIR,
    DATA_TEST_DIR,
    SCALER_PATH,
    MLP_MODEL_PATH,
)

# --- Importuri din modulele tale (conform structurii pe care ai descris-o) ---
# IMPORTANT:
# Dacă fișierele tale au alte nume / locații, schimbă doar aceste importuri.
from src.data_acquisition.generate_dataset import main as generate_main
from src.preprocessing.split_and_scale import main as preprocess_main
from src.neural_network.train_mlp import main as train_main
from src.neural_network.evaluate import main as evaluate_main


def _assert_exists(path: Path, what: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} nu a fost găsit:\n  -> {path}")


def step_generate():
    print("\n[STEP] GENERATE DATASET")
    generate_main()
    _assert_exists(RAW_CSV, "Fisierul dataset CSV")
    print(f"✓ OK: {RAW_CSV}")


def step_preprocess():
    print("\n[STEP] PREPROCESS + SPLIT + SCALE")
    _assert_exists(RAW_CSV, "Fisierul dataset CSV (rulează mai întâi generate)")
    preprocess_main()

    _assert_exists(SCALER_PATH, "Scaler (models/scaler.joblib)")
    _assert_exists(DATA_TRAIN_DIR / "X.npy", "Train X.npy")
    _assert_exists(DATA_TRAIN_DIR / "y.npy", "Train y.npy")
    _assert_exists(DATA_VAL_DIR / "X.npy", "Validation X.npy")
    _assert_exists(DATA_VAL_DIR / "y.npy", "Validation y.npy")
    _assert_exists(DATA_TEST_DIR / "X.npy", "Test X.npy")
    _assert_exists(DATA_TEST_DIR / "y.npy", "Test y.npy")

    print("✓ OK: splits + scaler salvate")


def step_train():
    print("\n[STEP] TRAIN MODEL (MLP)")
    _assert_exists(DATA_TRAIN_DIR / "X.npy", "Train data (rulează mai întâi preprocess)")
    _assert_exists(DATA_VAL_DIR / "X.npy", "Validation data (rulează mai întâi preprocess)")
    train_main()
    _assert_exists(MLP_MODEL_PATH, "Modelul antrenat (models/trained_model.h5)")
    print(f"✓ OK: model salvat -> {MLP_MODEL_PATH}")


def step_evaluate():
    print("\n[STEP] EVALUATE MODEL")
    _assert_exists(DATA_TEST_DIR / "X.npy", "Test data (rulează mai întâi preprocess)")
    _assert_exists(MLP_MODEL_PATH, "Modelul antrenat (rulează mai întâi train)")
    evaluate_main()
    print("✓ OK: evaluare completă (verifică results/ pentru metrici & figuri)")


def step_ui():
    """
    UI-ul la tine e în Streamlit (din ce ai descris).
    Îl pornesc prin subprocess ca să nu-ți blochez importurile.
    Ajustează path-ul dacă UI-ul tău nu e exact aici.
    """
    print("\n[STEP] UI (Streamlit)")
    ui_path = Path("src/app/main.py")
    _assert_exists(ui_path, "Fisier UI (Streamlit) src/app/main.py")

    # Pornire streamlit: echivalent cu `streamlit run src/app/main.py`
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_path)]
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Runner pipeline proiect RN")
    parser.add_argument(
        "--step",
        choices=["generate", "preprocess", "train", "evaluate", "ui", "all"],
        default="all",
        help="Ce rulezi acum"
    )
    args = parser.parse_args()

    if args.step == "generate":
        step_generate()
    elif args.step == "preprocess":
        step_preprocess()
    elif args.step == "train":
        step_train()
    elif args.step == "evaluate":
        step_evaluate()
    elif args.step == "ui":
        step_ui()
    else:  # all
        step_generate()
        step_preprocess()
        step_train()
        step_evaluate()

    print("\n✅ DONE")


if __name__ == "__main__":
    main()
