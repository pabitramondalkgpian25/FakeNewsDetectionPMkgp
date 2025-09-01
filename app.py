import streamlit as st
import pickle
import numpy as np

# ------------------- Page setup -------------------
st.set_page_config(page_title="Fake News Authenticity", page_icon="📰", layout="wide")

# Minimal CSS for newspaper vibe + better button
st.markdown("""
<style>
/* page background */
.main { background: #f6f6f3; }

/* center title */
h1.title {
  text-align:center; letter-spacing:1px; margin: 0.2rem 0 0.4rem 0;
  font-family: "Times New Roman", Georgia, serif;
}

/* tagline under title */
p.tagline {
  text-align:center; margin-top:0; color:#333; font-size: 16px;
}

/* logos row alignment */
.header-row { margin-bottom: 0.4rem; }

/* make text_area look like a news column */
section.newsbox .stTextArea textarea {
  font-family: "Georgia", "Times New Roman", serif !important;
  font-size: 18px; line-height: 1.6;
  background: #fffdf7; border: 1px solid #e2dccc; border-left: 6px solid #c9b27c;
}

/* nicer primary button */
div.stButton > button {
  width: 100%;
  padding: 0.8rem 1rem; font-size: 18px; border-radius: 10px;
  border: 0; background: #1f6feb; color: white;
}
div.stButton > button:hover { background: #195ecb; }

/* result card */
.result {
  background: #ffffff; border: 1px solid #e1e1e1; border-radius: 12px;
  padding: 18px; font-size: 18px;
}
.confidence { font-size: 14px; color: #666; }
</style>
""", unsafe_allow_html=True)

# ------------------- Load model + vectorizer (cached) -------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"❗ Could not load model/vectorizer: {e}")
        st.stop()

model, vectorizer = load_artifacts()

# figure out index of class "1" (we assume 1 = REAL, 0 = FAKE as in your notebook)
def prob_real(y_proba, classes_):
    # y_proba is shape (1, n_classes)
    classes = list(classes_)
    if 1 in classes:
        idx = classes.index(1)
    else:
        # fallback: assume positive class is the last column
        idx = -1
    return float(y_proba[0][idx])

# ------------------- Header with logos -------------------
c1, c2, c3 = st.columns([1, 2, 1], gap="large")

with c1:
    st.image("iitkgplogo1.png", caption=None, width=250)

with c2:
    st.markdown("<h1 class='title'>Fake   News    Authenticity</h1>", unsafe_allow_html=True)
    st.markdown(
    "<p style='text-align:center; font-size:20px; margin:0;'>Presented by</p>"
    "<h2 style='text-align:center; font-size:28px; margin:0;'><b>Pabitra Mondal</b></h2>"
    "<p style='text-align:center; font-size:18px; margin:0;'>Computer Science & Data Processing</p>"
    "<p style='text-align:center; font-size:18px; margin:0;'>Dept. of Maths. | IIT Kharagpur</p>",
    unsafe_allow_html=True
)


with c3:
    st.image("newspaperlogo.png", caption=None, width=250)

st.markdown("---")

# ------------------- Input area -------------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Write a news article")
    st.markdown("<section class='newsbox'>", unsafe_allow_html=True)
    news_text = st.text_area(
        label="News text",
        placeholder="Paste the news text here (body text or paragraph)...",
        height=160,
        label_visibility="collapsed"
    )
    st.markdown("</section>", unsafe_allow_html=True)

    detect = st.button("🔍 Detect Authenticity")

    if detect:
        if not news_text or not news_text.strip():
            st.warning("Please paste some news text first.")
        else:
            try:
                # IMPORTANT: pass raw text to the same vectorizer you trained
                X = vectorizer.transform([news_text.strip()])
                y_pred = model.predict(X)[0]
                if hasattr(model, "predict_proba"):
                    p_real = prob_real(model.predict_proba(X), model.classes_)
                else:
                    # logistic regression normally has predict_proba; fallback
                    p_real = 0.5

                # Simple decision without threshold
                is_real = (y_pred == 0)   # 0 → REAL, 1 → FAKE

                st.markdown("<div class='result'>", unsafe_allow_html=True)
                if is_real:
                   st.success("✅ REAL news detected")
                   st.markdown(f"<div class='confidence'>Logistic Regression Probablity : {p_real:.3f}</div>", unsafe_allow_html=True)
                else:
                   st.error("❌ FAKE news detected")
                   st.markdown(f"<div class='confidence'>Logistic Regression Probablity : {p_real:.3f}</div>", unsafe_allow_html=True)
                   st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                   st.error(f"Prediction failed: {e}")

with right:
                  st.subheader("Guidelines :")
                  st.write(
                              "- Write a short article of new or unseen news text.\n"
                              "- Result shows ✅ for REAL and ❌ for FAKE.\n"
                              "- Model trained on USA-based news.\n"
                          )















