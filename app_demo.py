"""Paprasta Streamlit demonstracinė sąsaja.

Įkeliamas (arba sugeneruojamas) naktinio kelio kadras -> parodomas sprendimas, tikimybė ir LED indikacija.
Sistema veikia tik PATARIAMUOJU režimu: realių automobilio šviesų ji nejungia, vairuotojas visada gali perimti valdymą.

Paleidimas:  streamlit run app_demo.py   (prieš tai: python code/train_and_eval.py)
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
import generate_dataset as gd  # noqa: E402
import train_and_eval as te  # noqa: E402

FREE_FRAMES_TO_RETURN = 15  # patikrinimo filtras: tiek kadrų iš eilės kelias turi būti laisvas

st.set_page_config(page_title="Automatinis šviesų perjungimas", page_icon="🚗", layout="centered")
st.title("Automatinis šviesų perjungimas")
st.caption("Patariamasis režimas: sistema tik rodo sprendimą, realių šviesų nejungia. Vairuotojas visada gali perimti valdymą.")


@st.cache_resource
def get_model():
    if te.torch is None:
        return None, "PyTorch neįdiegtas (pip install torch)."
    path = ROOT / "models" / "cnn.pt"
    if not path.exists():
        return None, "Nerastas models/cnn.pt. Pirmiausia paleiskite: python code/train_and_eval.py"
    return te.load_cnn(path), None


model, err = get_model()
if err:
    st.error(err)
    st.stop()

if "rng" not in st.session_state:
    st.session_state.rng = np.random.default_rng()
    st.session_state.free_streak = 0
    st.session_state.high_beam = True
    st.session_state.frame = None

st.sidebar.header("Nustatymai")
source = st.sidebar.radio("Kadro šaltinis", ["Sugeneruoti sintetinį kadrą", "Įkelti savo kadrą"])
threshold = st.sidebar.slider("Slenkstis (tikimybė, nuo kurios laikoma, kad priekyje automobilis)", 0.1, 0.9, 0.5, 0.05)
use_filter = st.sidebar.checkbox(f"Patikrinimo filtras (grįžti į tolimąsias po {FREE_FRAMES_TO_RETURN} laisvų kadrų)", value=True)

frame = None
if source == "Sugeneruoti sintetinį kadrą":
    c1, c2, c3 = st.columns(3)
    label = None
    if c1.button("Atsitiktinis kadras"):
        label = int(st.session_state.rng.integers(0, 2))
    if c2.button("Kelias laisvas (0 klasė)"):
        label = 0
    if c3.button("Priekyje automobilis (1 klasė)"):
        label = 1
    if label is not None:
        st.session_state.frame = gd.make_frame(label, st.session_state.rng)
        st.session_state.true_label = label
        new_frame = True
    else:
        new_frame = False
    frame = st.session_state.frame
else:
    up = st.file_uploader("Naktinio kelio kadras (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if up is not None:
        frame = np.array(Image.open(up).convert("RGB").resize((gd.IMG, gd.IMG)))
        uid = getattr(up, "file_id", up.name)
        new_frame = st.session_state.get("last_upload") != uid
        st.session_state.last_upload = uid
        st.session_state.true_label = None

if frame is None:
    st.info("Pasirinkite arba įkelkite kadrą.")
    st.stop()

prob = float(te.cnn_proba(model, frame[None])[0])
car_ahead = prob >= threshold

# Patikrinimo filtras: į artimąsias - iš karto, į tolimąsias - tik po keliolikos laisvų kadrų
if new_frame:
    if car_ahead:
        st.session_state.high_beam, st.session_state.free_streak = False, 0
    else:
        st.session_state.free_streak += 1
        if not use_filter or st.session_state.free_streak >= FREE_FRAMES_TO_RETURN:
            st.session_state.high_beam = True
decision_high = st.session_state.high_beam if use_filter else (not car_ahead)

left, right = st.columns([1, 1])
left.image(Image.fromarray(frame).resize((256, 256), Image.NEAREST), caption="Kameros kadras (64×64)")
with right:
    st.metric("Tikimybė, kad priekyje automobilis", f"{prob * 100:.1f} %")
    st.progress(min(max(prob, 0.0), 1.0))
    color, text = ("#1f77ff", "TOLIMOSIOS šviesos (mėlynas LED)") if decision_high else ("#2ca02c", "ARTIMOSIOS šviesos (žalias LED)")
    st.markdown(f"<div style='display:flex;align-items:center;gap:12px'><div style='width:34px;height:34px;border-radius:50%;"
                f"background:{color};box-shadow:0 0 14px {color}'></div><b>{text}</b></div>", unsafe_allow_html=True)
    if use_filter:
        st.caption(f"Laisvų kadrų iš eilės: {st.session_state.free_streak} / {FREE_FRAMES_TO_RETURN}")
    if st.session_state.get("true_label") is not None:
        st.caption("Tikroji klasė: " + gd.CLASS_NAMES[st.session_state.true_label])
st.warning("Tai mokomasis prototipas, apmokytas su sintetiniais duomenimis. Vairuotojas atsakingas už realų šviesų perjungimą.")
