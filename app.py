import os
import json
import joblib
import difflib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any
from sklearn.linear_model import LinearRegression

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="🚘 Prediksi Harga Mobil + Chat Gemini",
    layout="wide",
    page_icon="🚗",
)

# =========================
# CSS KUSTOM (Tema PINK + Chat Bar Putih, Teks Hitam)
# =========================
st.markdown("""
<style>
:root {
    --primary-color: #E91E63;    /* Pink utama (magenta) */
    --secondary-color: #FF4081;  /* Hot pink */
    --accent-color: #FFC1E3;     /* Pink pastel */
    --bg1: #1a0b14;              /* Latar gradasi gelap 1 */
    --bg2: #3a0f27;              /* Latar gradasi gelap 2 */
    --bg3: #6a1b3f;              /* Latar gradasi gelap 3 */
    --text-light: #FFEFF6;       /* Teks terang nuansa pink */
    --shadow: 0 0 35px rgba(233, 30, 99, 0.40); /* Bayangan pink */
}

/* Latar belakang app */
html, body, .stApp {
    background: linear-gradient(145deg, var(--bg1), var(--bg2), var(--bg3));
    background-size: 300% 300%;
    animation: bgflow 12s ease infinite;
    color: var(--text-light);
    font-family: "Poppins", sans-serif;
}
@keyframes bgflow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Header utama */
.main-header {
    font-size: 2.4em;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(270deg, var(--accent-color), var(--secondary-color), var(--primary-color));
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 8s ease infinite;
    margin-bottom: 25px;
}
@keyframes gradientFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border-right: 1px solid rgba(255,255,255,0.14);
}
section[data-testid="stSidebar"] * { color: var(--text-light) !important; }

/* Kontainer chat (panel riwayat) */
.chat-container {
    background: rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 15px;
    box-shadow: var(--shadow);
}

/* Bubble chat */
.chat-bubble-user {
    background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 10px 15px;
    margin: 10px 0;
    max-width: 85%;
    word-wrap: break-word;
    box-shadow: 0 0 12px rgba(233,30,99,0.35);
}
.chat-bubble-ai {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 18px 18px 18px 0;
    padding: 10px 15px;
    margin: 10px 0;
    max-width: 85%;
    word-wrap: break-word;
}

/* Tombol umum */
.stButton>button {
    background: linear-gradient(135deg, var(--accent-color), var(--primary-color));
    border: none !important;
    color: #2b0e1c !important; /* teks kontras di atas accent */
    border-radius: 10px;
    padding: 8px 14px;
    font-weight: 700;
    box-shadow: 0 0 12px rgba(233,30,99,0.45);
}
.stButton>button:hover {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    transform: scale(1.03);
    transition: 0.15s ease;
    color: #fff !important;
}

/* Input umum */
.stSelectbox, .stNumberInput, .stTextInput, .stSlider { color: var(--text-light) !important; }
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.22);
}
input, textarea {
    background: rgba(255,255,255,0.07) !important;
    color: var(--text-light) !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
}

/* Dataframe card */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: var(--shadow);
    padding: 6px;
}

/* Divider & caption */
hr, .stDivider { border-color: rgba(255,255,255,0.18) !important; }
.stCaption { color: var(--text-light) !important; }

/* ====== CHAT INPUT BAR PUTIH, TEKS HITAM ====== */
div[data-testid="stChatInput"] {
    background: #ffffff !important;
    border-top: 1px solid rgba(0,0,0,0.12) !important;
}
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] [contenteditable="true"] {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid rgba(0,0,0,0.18) !important;
}
div[data-testid="stChatInput"] textarea::placeholder { color: #444 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🚘 Prediksi Harga Mobil + Chat Gemini AI</div>", unsafe_allow_html=True)

# =========================
# UTIL
# =========================
def rupiah(x: float) -> str:
    try:
        return f"Rp {x:,.0f}".replace(",", ".")
    except Exception:
        return "Rp -"

@st.cache_resource
def load_model_if_exists(path: str):
    """Memuat model jika tersedia, atau buat dummy model jika gagal."""
    if os.path.exists(path):
        try:
            st.sidebar.info("✅ Model ditemukan dan dimuat dari disk.")
            return joblib.load(path)
        except Exception as e:
            st.sidebar.error(f"⚠ Gagal memuat model.pkl: {e}")
            st.sidebar.warning("🔄 Menggunakan model dummy sementara.")
            return LinearRegression().fit(np.random.rand(10, 3), np.random.rand(10))
    else:
        st.sidebar.warning("⚠ model.pkl tidak ditemukan. Menggunakan model dummy.")
        return LinearRegression().fit(np.random.rand(10, 3), np.random.rand(10))

def prepare_input(form_data: Dict[str, Any], example_schema: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{k: form_data.get(k, v) for k, v in example_schema.items()}])

def save_to_dataset(input_df: pd.DataFrame, prediction: float, path: str):
    df_to_save = input_df.copy()
    df_to_save["prediction"] = prediction
    if os.path.exists(path):
        try:
            dataset = pd.read_csv(path)
        except Exception:
            dataset = pd.DataFrame()
        dataset = pd.concat([dataset, df_to_save], ignore_index=True)
    else:
        dataset = df_to_save
    dataset.to_csv(path, index=False)
    return dataset

# =========================
# KONFIG GEMINI (ENV FIRST)
# =========================
# (Keamanan) ambil dari environment/secret; jangan hardcode API key
gemini_api_key_env = os.environ.get("GEMINI_API_KEY", "AIzaSyBpcTaHImTZlM0k8-hRDufX4l-NCfuLT_Y")
try:
    import google.generativeai as genai
    HAVE_GENAI = True
except Exception:
    HAVE_GENAI = False

with st.sidebar:
    st.subheader("🤖 Gemini")
    if not HAVE_GENAI:
        st.error("Paket google-generativeai belum terpasang.\nJalankan:\n`python -m pip install google-generativeai`")
    key_status = "TERDETEKSI" if gemini_api_key_env else "TIDAK TERDETEKSI"
    st.write(f"GEMINI_API_KEY: *{key_status}*")
    manual_key = "" if gemini_api_key_env else st.text_input("API Key (opsional, tidak disimpan)", type="password")
    if manual_key:
        st.session_state["manual_key"] = manual_key
    model_name = st.selectbox("Model LLM", ["gemini-2.5-flash","gemini-flash-latest","gemini-2.5-pro"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.6, 0.1)
    max_output_tokens = st.slider("Max Output Tokens", 256, 4096, 1024, 64)
    answer_all = st.toggle("🟦 Gemini jawab SEMUA pertanyaan (disarankan)", value=True)
    list_models = st.toggle("Tampilkan daftar model yang mendukung generateContent", value=False)

def _get_gemini_client(api_key: str):
    if not HAVE_GENAI:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception:
        return None

def _active_api_key() -> str:
    return gemini_api_key_env or st.session_state.get("manual_key", "") or (manual_key if 'manual_key' in locals() else "")

def _dataset_summary(df: pd.DataFrame, max_models: int = 20) -> str:
    if df is None or df.empty:
        return "Dataset kosong/tidak dimuat."
    cols = ", ".join(df.columns.tolist()[:20])
    summary = f"Kolom: {cols}."
    if "model" in df.columns:
        uniq = sorted(df['model'].dropna().astype(str).unique().tolist())
        if uniq:
            summary += f" Contoh model: {', '.join(uniq[:max_models])}."
    if "price" in df.columns:
        try:
            mean_price = float(df['price'].mean())
            summary += f" Rata-rata price (unit asli dataset): {mean_price:,.2f}."
        except Exception:
            pass
    return summary

def gemini_chat(
    user_message: str,
    history: list[tuple[str, str]] | None = None,
    df: pd.DataFrame | None = None,
    last_prediction: float | None = None,
    last_input: dict | None = None,
) -> str:
    """
    Chat generik: arahkan SEMUA pertanyaan ke Gemini.
    Sertakan konteks ringan: ringkasan dataset, prediksi terakhir (jika ada), dan riwayat chat.
    """
    api_key = _active_api_key()
    if not api_key:
        return "⚠ Gemini belum aktif karena API Key tidak terpasang. Set GEMINI_API_KEY di environment atau isi di sidebar."

    client = _get_gemini_client(api_key)
    if client is None:
        return "⚠ Gagal menginisialisasi klien Gemini. Periksa API key atau koneksi."

    ds_ctx = _dataset_summary(df)
    last_ctx = ""
    if last_prediction is not None and last_input:
        try:
            last_ctx = f"Prediksi terakhir: {json.dumps(last_input)} → £{last_prediction:,.2f} (≈ {rupiah(last_prediction*20000)})."
        except Exception:
            pass

    chat_history_text = ""
    if history:
        tail = history[-6:]
        lines = []
        for role, msg in tail:
            r = "User" if role == "user" else "Assistant"
            lines.append(f"{r}: {msg}")
        chat_history_text = "\n".join(lines)

    system_prompt = f"""
Kamu adalah asisten AI serbaguna untuk pengguna Indonesia.
Tugasmu: jawab pertanyaan apapun secara jelas, ringkas, dan akurat.
Jika pertanyaan tentang data mobil, kamu boleh memanfaatkan ringkasan dataset di bawah.
Jika bukan tentang mobil, jawab seperti asisten umum (tanpa bergantung dataset).
Gunakan bahasa Indonesia yang natural. Sertakan langkah atau contoh jika relevan.
Hindari spekulasi berlebihan — jika ragu, jelaskan asumsi atau minta klarifikasi singkat.

KONTEKS DATASET (opsional):
{ds_ctx}

KONTEKS PREDIKSI TERAKHIR (opsional):
{last_ctx}

RIWAYAT CHAT (ringkas):
{chat_history_text}
""".strip()

    try:
        prompt = f"{system_prompt}\n\nPERTANYAAN PENGGUNA:\n{user_message}\n"
        resp = client.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
        )
        text = (getattr(resp, "text", None) or "").strip()
        return text or "Maaf, aku tidak menerima teks balasan dari Gemini."
    except Exception as e:
        return f"⚠ Gagal memanggil Gemini API: {e}"

def local_chat_response(user_message: str, last_prediction=None, last_input=None, df=None):
    """
    Fallback lokal jika Gemini mati.
    """
    msg = (user_message or "").lower().strip()

    if msg in ["model", "daftar model", "model mobil"]:
        if df is not None and not df.empty and "model" in df.columns:
            all_models = sorted(df["model"].dropna().astype(str).unique().tolist())
            return "🚗 Jenis mobil yang tersedia:\n" + ", ".join(all_models[:50])
        else:
            return "Dataset belum dimuat atau kolom 'model' tidak ditemukan."

    if "harga" in msg:
        if df is not None and not df.empty and "model" in df.columns and "price" in df.columns:
            models = df["model"].dropna().astype(str).unique().tolist()
            best_match = difflib.get_close_matches(msg, [m.lower() for m in models], n=1, cutoff=0.6)
            if best_match:
                found = best_match[0]
                matched_model = next((m for m in models if m.lower() == found), found)
                avg_price = df[df["model"].astype(str).str.lower() == matched_model.lower()]["price"].mean()
                if pd.notna(avg_price):
                    harga_rupiah = avg_price * 20000
                    return f"💰 Harga rata-rata {matched_model} sekitar {rupiah(harga_rupiah)} (≈ £{avg_price:,.2f})."
                else:
                    return f"Harga {matched_model} belum tersedia di dataset."
            else:
                return "Untuk harga model itu belum ada di dataset. Coba sebutkan detail spesifik (tahun, transmisi, kilometer)."
        else:
            return "Dataset tidak tersedia. Kamu bisa tetap bertanya hal lain kok 🙂"

    if any(x in msg for x in ["halo", "hai", "hi"]):
        return "Halo! 👋 Aku siap bantu. Tanyakan apa saja."

    return "Aku siap jawab pertanyaanmu. Jika tentang mobil, sebutkan model/tahun untuk estimasi."

# =========================
# PATH & DATA
# =========================
BASE_DIR = os.getcwd()
MODEL_PATH = "model.pkl"
DATASET_PATH = "toyota.csv"
EXAMPLE_PATH = "example_schema.json"

# ===== Sidebar: status & uploader =====
with st.sidebar:
    st.subheader("📁 Berkas")
    st.write(f"Folder kerja: {BASE_DIR}")
    st.write(f"model.pkl: {'✅' if os.path.exists(MODEL_PATH) else '❌'}")
    st.write(f"toyota.csv: {'✅' if os.path.exists(DATASET_PATH) else '❌'}")
    st.write("---")
    up_model = st.file_uploader("Upload model.pkl (opsional)", type=["pkl"])
    if up_model is not None:
        try:
            with open(MODEL_PATH, "wb") as f:
                f.write(up_model.read())
            st.success("model.pkl tersimpan.")
        except Exception as e:
            st.error(f"Gagal menyimpan model: {e}")

    up_csv = st.file_uploader("Upload toyota.csv (opsional)", type=["csv"])
    if up_csv is not None:
        try:
            df_tmp = pd.read_csv(up_csv)
            df_tmp.to_csv(DATASET_PATH, index=False)
            st.success("toyota.csv tersimpan.")
        except Exception as e:
            st.error(f"Gagal menyimpan dataset: {e}")

    st.divider()
    if list_models and HAVE_GENAI:
        try:
            genai.configure(api_key=_active_api_key())
            names = []
            for m in genai.list_models():
                if "generateContent" in getattr(m, "supported_generation_methods", []):
                    names.append(m.name)
            if names:
                st.caption("Model yang mendukung generateContent:")
                st.code("\n".join(names), language="text")
        except Exception as e:
            st.warning(f"List model gagal: {e}")

# Load model
model = load_model_if_exists(MODEL_PATH)

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
    st.sidebar.success(f"📊 Dataset dimuat ({len(df)} baris)")
except Exception as e:
    st.sidebar.warning(f"⚠ Gagal memuat dataset: {e}")
    df = pd.DataFrame()

# Load example schema (atau fallback)
try:
    with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
        example_schema = json.load(f)
except Exception:
    example_schema = {
        "model": "Avanza", "year": 2020, "transmission": "Manual",
        "mileage": 15000, "fuelType": "Bensin", "tax": 1500000,
        "mpg": 14.5, "engineSize": 1.3
    }

# =========================
# ANTARMUKA
# =========================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("🚘 Form Prediksi Harga Mobil")
    with st.form("form_prediksi"):
        inputs = {}
        model_list = df["model"].dropna().astype(str).unique().tolist() if not df.empty and "model" in df.columns else ["Avanza", "Yaris", "Rush"]
        inputs["model"] = st.selectbox("Model Mobil", model_list)
        inputs["year"] = st.number_input("Tahun Produksi", 1990, 2025, int(example_schema.get("year", 2020)))
        inputs["transmission"] = st.selectbox("Transmisi", ["Manual", "Automatic"])
        inputs["mileage"] = st.number_input("Jarak Tempuh (km)", 0, 500000, int(example_schema.get("mileage", 15000)))
        inputs["fuelType"] = st.selectbox("Jenis Bahan Bakar", ["Bensin", "Diesel", "Hybrid"])
        inputs["tax"] = st.number_input("Pajak Tahunan (Rp)", 0, 10000000, int(example_schema.get("tax", 1500000)))
        mpg_value = float(example_schema.get("mpg", 14.5))
        inputs["mpg"] = st.number_input("Efisiensi BBM (km/l)", 0.0, max(100.0, mpg_value), mpg_value)
        inputs["engineSize"] = st.number_input("Kapasitas Mesin (L)", 0.0, 5.0, float(example_schema.get("engineSize", 1.3)))
        submitted = st.form_submit_button("🚀 Prediksi Harga")

    if submitted and model is not None:
        X = prepare_input(inputs, example_schema)
        try:
            pred = model.predict(X)
            price = float(pred[0]) if isinstance(pred, (list, np.ndarray, pd.Series)) else float(pred)
            harga_rupiah = price * 20000
            save_to_dataset(X, price, DATASET_PATH)
            st.success(f"💰 Estimasi harga {inputs['model']} sekitar {rupiah(harga_rupiah)} (≈ £{price:,.2f})")
            st.session_state["last_prediction"] = price
            st.session_state["last_input"] = inputs

            # Simpan history ke session
            hist_row = {**inputs, "pred_gbp": price, "pred_idr": harga_rupiah}
            if "pred_history" not in st.session_state:
                st.session_state["pred_history"] = []
            st.session_state["pred_history"].append(hist_row)
        except Exception as e:
            st.error(f"Gagal melakukan prediksi. Pastikan pipeline & fitur cocok. Detail: {e}")

    # Tabel riwayat & unduh
    if "pred_history" in st.session_state and st.session_state["pred_history"]:
        st.markdown("### 📜 Riwayat Prediksi")
        hist_df = pd.DataFrame(st.session_state["pred_history"])
        st.dataframe(hist_df, use_container_width=True)
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Unduh Riwayat (CSV)", data=csv_bytes, file_name="riwayat_prediksi.csv", mime="text/csv")

with right:
    st.subheader("💬 Chat Asisten (Gemini-first)")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Reset Chat"):
            st.session_state["chat_history"] = []
            st.toast("Chat direset.")
    with c2:
        st.caption("Tips: tanya apa saja. Tekan Enter untuk mengirim.")
    with c3:
        mode_label = "Gemini-first ✅" if answer_all else "Fallback Lokal"
        st.caption(mode_label)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Riwayat chat (bubble)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for role, msg in st.session_state.chat_history:
        bubble = "chat-bubble-user" if role == "user" else "chat-bubble-ai"
        st.markdown(f"<div class='{bubble}'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ===== Kirim cukup tekan Enter (st.chat_input) =====
    user_msg = st.chat_input("Ketik pertanyaan di sini…")
    if user_msg:
        st.session_state.chat_history.append(("user", user_msg))
        if answer_all:
            response = gemini_chat(
                user_msg,
                history=st.session_state.get("chat_history"),
                df=df,
                last_prediction=st.session_state.get("last_prediction"),
                last_input=st.session_state.get("last_input"),
            )
        else:
            response = local_chat_response(
                user_msg,
                st.session_state.get("last_prediction"),
                st.session_state.get("last_input"),
                df
            )
        st.session_state.chat_history.append(("assistant", response))
        st.rerun()

st.markdown("---")
st.caption("✨ Aplikasi prediksi harga mobil + chat Gemini AI — tema PINK, chat bar putih, Enter untuk kirim ✨")
