"""
app.py  —  Spam Mail Detector · Streamlit Web App
Run with:  streamlit run app.py
"""

import os, sys, pickle, re, string, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud

sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import clean_text, engineer_features, FEATURE_COLS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #FAFAF8; }
  .stApp { font-family: 'Segoe UI', sans-serif; }
  .big-metric { font-size: 2.4rem; font-weight: 700; }
  .spam-tag  { background:#FCEBEB; color:#A32D2D; padding:4px 14px;
               border-radius:20px; font-weight:600; font-size:1rem; }
  .ham-tag   { background:#E1F5EE; color:#0F6E56; padding:4px 14px;
               border-radius:20px; font-weight:600; font-size:1rem; }
  .feature-pill { background:#F1EFE8; padding:3px 10px;
                  border-radius:12px; font-size:0.82rem; margin:2px; }
  div[data-testid="stMetricValue"] { font-size:1.8rem !important; }
  .section-header { font-size:1.05rem; font-weight:600; color:#2C2C2A;
                    border-left:3px solid #E24B4A; padding-left:10px;
                    margin:18px 0 10px; }
</style>
""", unsafe_allow_html=True)

C_SPAM = "#E24B4A"
C_HAM  = "#1D9E75"

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "models/best_model.pkl"
    if not os.path.exists(path):
        st.error("Model not found. Please run:  python train.py")
        st.stop()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["name"]

@st.cache_data
def load_data():
    for p, sep in [("data/sms.tsv","\t"), ("data/sms.csv",",")]:
        if os.path.exists(p):
            df = pd.read_csv(p, sep=sep)
            if df.shape[1] == 2:
                df.columns = ["label","message"]
            df = engineer_features(df)
            df["clean"] = df["message"].apply(clean_text)
            return df
    return None

model, model_name = load_model()
df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Spam Detector")
    st.markdown(f"**Model:** `{model_name}`")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Live Classifier", "📊 Analytics Dashboard",
                                  "🧪 Batch Test", "ℹ️ About"])
    st.markdown("---")
    st.markdown("**Dataset**")
    if df is not None:
        st.metric("Total messages", f"{len(df):,}")
        st.metric("Spam", f"{(df['label']=='spam').sum():,}")
        st.metric("Ham",  f"{(df['label']=='ham').sum():,}")
    st.markdown("---")
    st.caption("Internship ML Project · NLP Pipeline")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: LIVE CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔍 Live Classifier":
    st.title("🛡️ Spam Mail Detector")
    st.markdown("Type or paste any message below to instantly classify it as **spam** or **ham**.")

    col_input, col_result = st.columns([1.4, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-header">Enter Message</div>', unsafe_allow_html=True)
        user_msg = st.text_area("", height=160, placeholder="e.g. WINNER!! You've won £500 cash. Call now to claim!", label_visibility="collapsed")

        ex_col1, ex_col2 = st.columns(2)
        if ex_col1.button("📩 Load spam example"):
            user_msg = "WINNER!! Congratulations! You've been selected to win a £500 cash prize. Call 09061743386 NOW to claim! Send STOP to opt out."
            st.session_state["loaded_msg"] = user_msg
        if ex_col2.button("✉️ Load ham example"):
            user_msg = "Hey! Are you free for lunch tomorrow? The usual place at 12:30 works for me."
            st.session_state["loaded_msg"] = user_msg

        if "loaded_msg" in st.session_state and not user_msg:
            user_msg = st.session_state["loaded_msg"]

        threshold = st.slider("Classification threshold (spam probability)", 0.1, 0.9, 0.5, 0.05,
                               help="Lower = more aggressive spam filtering")

    with col_result:
        if user_msg.strip():
            clean = clean_text(user_msg)
            try:
                proba = model.predict_proba([clean])[0]
                spam_prob = proba[1]
            except Exception:
                pred  = model.predict([clean])[0]
                spam_prob = 1.0 if pred == 1 else 0.0

            is_spam = spam_prob >= threshold
            label   = "SPAM" if is_spam else "HAM"
            conf    = spam_prob if is_spam else (1 - spam_prob)

            st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)

            tag_html = f'<span class="{"spam-tag" if is_spam else "ham-tag"}">{label}</span>'
            st.markdown(f"### {tag_html}", unsafe_allow_html=True)

            st.metric("Spam probability", f"{spam_prob*100:.1f}%")
            st.metric("Confidence", f"{conf*100:.1f}%")

            # Visual gauge
            fig_g, ax_g = plt.subplots(figsize=(5, 0.5))
            ax_g.barh(0, 1, color="#F1EFE8", height=0.6)
            ax_g.barh(0, spam_prob, color=C_SPAM if is_spam else C_HAM, height=0.6)
            ax_g.axvline(threshold, color="#888", lw=1.5, linestyle="--")
            ax_g.set_xlim(0,1); ax_g.axis("off")
            fig_g.patch.set_alpha(0)
            st.pyplot(fig_g, use_container_width=True)
            plt.close()

            # Feature breakdown
            st.markdown('<div class="section-header">Detected signals</div>', unsafe_allow_html=True)
            signals = []
            if re.search(r"http|www|bit\.ly", user_msg, re.I): signals.append("🔗 URL present")
            if re.search(r"\b\d{10,11}\b", user_msg):           signals.append("📞 Phone number")
            if re.search(r"[£$€]|\bpound", user_msg, re.I):     signals.append("💰 Currency mention")
            if re.search(r"\bfree\b", user_msg, re.I):           signals.append("🎁 'FREE' keyword")
            if re.search(r"\bwin\b|\bwinner\b|\bwon\b", user_msg, re.I): signals.append("🏆 Win/Winner")
            if re.search(r"\burgent\b|\bimportant\b", user_msg, re.I):   signals.append("🚨 Urgency language")
            if re.search(r"\bcall\b|\btxt\b|\btext\b", user_msg, re.I):  signals.append("📲 Call-to-action")
            caps = sum(c.isupper() for c in user_msg)/max(len(user_msg),1)
            if caps > 0.2: signals.append(f"🔠 High CAPS ({caps*100:.0f}%)")
            if user_msg.count("!") >= 2: signals.append(f"❗ {user_msg.count('!')} exclamation marks")

            if signals:
                pills = "".join(f'<span class="feature-pill">{s}</span> ' for s in signals)
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown("*No strong spam signals detected.*")
        else:
            st.info("Enter a message on the left to get a prediction.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: ANALYTICS DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    if df is None:
        st.error("Dataset not found."); st.stop()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total messages",   f"{len(df):,}")
    c2.metric("Spam",             f"{(df['label']=='spam').sum():,}")
    c3.metric("Ham",              f"{(df['label']=='ham').sum():,}")
    c4.metric("Spam rate",        f"{(df['label']=='spam').mean()*100:.1f}%")
    c5.metric("Avg spam length",  f"{df[df['label']=='spam']['feat_length'].mean():.0f} ch")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "☁️ Word Clouds", "🔬 Feature Analysis"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = plt.subplots(figsize=(5,3.5))
            for lbl, c, nm in [("ham",C_HAM,"Ham"),("spam",C_SPAM,"Spam")]:
                vals = df[df["label"]==lbl]["feat_length"]
                ax.hist(vals, bins=40, alpha=0.65, color=c, label=f"{nm} (μ={vals.mean():.0f})", density=True)
            ax.set_title("Message Length Distribution", fontweight="bold")
            ax.set_xlabel("Characters"); ax.set_ylabel("Density")
            ax.legend(); ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        with col_b:
            fig, ax = plt.subplots(figsize=(5,3.5))
            data = [df[df["label"]=="ham"]["feat_caps_ratio"].values,
                    df[df["label"]=="spam"]["feat_caps_ratio"].values]
            bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                            medianprops=dict(color="white",lw=2))
            for patch, c in zip(bp["boxes"], [C_HAM, C_SPAM]):
                patch.set_facecolor(c); patch.set_alpha(0.8)
            ax.set_xticklabels(["Ham","Spam"])
            ax.set_title("CAPS Ratio by Class", fontweight="bold")
            ax.set_ylabel("Fraction uppercase")
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        col_c, col_d = st.columns(2)
        with col_c:
            fig, ax = plt.subplots(figsize=(5,3.5))
            for lbl, c, nm in [("ham",C_HAM,"Ham"),("spam",C_SPAM,"Spam")]:
                vals = df[df["label"]==lbl]["feat_word_count"]
                ax.hist(vals, bins=30, alpha=0.65, color=c, label=f"{nm} (μ={vals.mean():.1f})", density=True)
            ax.set_title("Word Count Distribution", fontweight="bold")
            ax.set_xlabel("Words"); ax.legend()
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        with col_d:
            fig, ax = plt.subplots(figsize=(5,3.5))
            features = ["feat_has_url","feat_has_phone","feat_has_currency",
                        "feat_has_free","feat_has_win","feat_has_urgent","feat_has_call"]
            labels_short = ["URL","Phone","Currency","FREE kw","Win kw","Urgent","Call-to-action"]
            spam_rates = [df[df["label"]=="spam"][f].mean()*100 for f in features]
            ham_rates  = [df[df["label"]=="ham"][f].mean()*100  for f in features]
            x = range(len(features))
            w = 0.38
            ax.bar([i-w/2 for i in x], ham_rates,  width=w, color=C_HAM,  label="Ham",  alpha=0.85)
            ax.bar([i+w/2 for i in x], spam_rates, width=w, color=C_SPAM, label="Spam", alpha=0.85)
            ax.set_xticks(list(x)); ax.set_xticklabels(labels_short, rotation=30, ha="right", fontsize=9)
            ax.set_title("Spam Signal Presence (%)", fontweight="bold")
            ax.set_ylabel("%"); ax.legend()
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            st.markdown("**Spam messages — frequent words**")
            spam_text = " ".join(df[df["label"]=="spam"]["clean"].tolist())
            wc = WordCloud(width=600, height=320, background_color="white",
                           colormap="Reds", max_words=80, collocations=False).generate(spam_text)
            fig, ax = plt.subplots(figsize=(6,3.2))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig, use_container_width=True); plt.close()

        with col_wc2:
            st.markdown("**Ham messages — frequent words**")
            ham_text = " ".join(df[df["label"]=="ham"]["clean"].tolist())
            wc = WordCloud(width=600, height=320, background_color="white",
                           colormap="Greens", max_words=80, collocations=False).generate(ham_text)
            fig, ax = plt.subplots(figsize=(6,3.2))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        st.markdown("**Correlation heatmap — engineered features**")
        feat_df = df[FEATURE_COLS + ["label_num"]].copy()
        feat_df.columns = [c.replace("feat_","") for c in feat_df.columns[:-1]] + ["label"]
        corr = feat_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax,
                    cmap="RdYlGn", center=0, linewidths=0.5,
                    annot_kws={"size": 8}, vmin=-1, vmax=1)
        ax.set_title("Feature Correlation Matrix", fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: BATCH TEST
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Batch Test":
    st.title("🧪 Batch Message Classifier")
    st.markdown("Paste multiple messages (one per line) and classify them all at once.")

    batch_input = st.text_area("Messages (one per line)", height=200,
        value="\n".join([
            "WINNER!! You've won £500 cash. Call 09061743386 NOW!",
            "Hey, are you free for lunch tomorrow at 12:30?",
            "URGENT: Your account will be suspended. Verify at http://bit.ly/abc",
            "Can you pick up some milk on your way home?",
            "FREE entry to win a trip to Bahamas. Text WIN to 87575",
            "Happy birthday! Hope you have a wonderful day!",
            "HOT singles in your area! Text MEET to 8837. Cost £1.50/msg",
            "Running 10 mins late, sorry! Almost there.",
        ]))

    if st.button("🔍 Classify all messages"):
        messages = [m.strip() for m in batch_input.strip().split("\n") if m.strip()]
        if not messages:
            st.warning("Please enter at least one message.")
        else:
            with st.spinner("Classifying…"):
                cleans = [clean_text(m) for m in messages]
                try:
                    probas = model.predict_proba(cleans)[:,1]
                except Exception:
                    preds  = model.predict(cleans)
                    probas = np.array([1.0 if p==1 else 0.0 for p in preds])

                labels  = ["SPAM" if p >= 0.5 else "HAM" for p in probas]
                results = pd.DataFrame({
                    "Message":      messages,
                    "Label":        labels,
                    "Spam prob (%)": (probas * 100).round(1),
                })

            # Summary
            n_spam = sum(1 for l in labels if l=="SPAM")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total classified", len(messages))
            c2.metric("Spam detected",  n_spam)
            c3.metric("Ham (safe)",     len(messages)-n_spam)

            st.markdown("---")
            for _, row in results.iterrows():
                col_l, col_m, col_p = st.columns([0.12, 0.72, 0.16])
                tag = f'<span class="{"spam-tag" if row["Label"]=="SPAM" else "ham-tag"}">{row["Label"]}</span>'
                col_l.markdown(tag, unsafe_allow_html=True)
                col_m.markdown(f"<small>{row['Message'][:120]}</small>", unsafe_allow_html=True)
                color = C_SPAM if row["Label"]=="SPAM" else C_HAM
                col_p.markdown(f'<span style="color:{color};font-weight:600;">{row["Spam prob (%)"]:.1f}%</span>',
                               unsafe_allow_html=True)

            st.markdown("---")
            csv = results.to_csv(index=False)
            st.download_button("⬇️ Download results CSV", csv,
                               file_name="spam_predictions.csv", mime="text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
## Spam Mail Detector
An end-to-end NLP machine learning pipeline that classifies SMS messages as **spam** or **ham**.

---
### 🔧 Pipeline Overview

| Stage | Details |
|-------|---------|
| **Dataset** | SMS Spam Collection (UCI) — 5,474 messages |
| **Preprocessing** | Lowercase · URL/phone/currency tokenisation · stopword removal |
| **Features** | TF-IDF bigrams (8k features) + 14 hand-crafted signals |
| **Models** | Naive Bayes · Complement NB · Logistic Regression · Linear SVM · Ensemble |
| **Tuning** | GridSearchCV (3-fold, F1 optimised) |
| **Evaluation** | Accuracy · Precision · Recall · F1 · ROC-AUC · 5-fold CV |

---
### 📁 Project Structure
```
spam_detector/
├── data/
│   ├── sms.csv                  # Dataset
│   └── generate_dataset.py      # Dataset generator
├── models/
│   └── best_model.pkl           # Saved best model
├── outputs/
│   └── results_dashboard.png    # Training visualisations
├── preprocessor.py              # Text cleaning & feature engineering
├── train.py                     # Full training pipeline
├── app.py                       # This Streamlit app
├── requirements.txt
└── README.md
```

---
### 🛠️ Skills Demonstrated
- Text preprocessing (regex, stopwords, tokenisation)
- Feature extraction (Bag of Words, TF-IDF with bigrams)
- Hand-crafted NLP features (caps ratio, URL detection, etc.)
- Multiple classifier training and comparison
- Hyperparameter tuning with GridSearchCV
- Ensemble methods (Soft Voting Classifier)
- Cross-validation (5-fold stratified)
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Model persistence (pickle)
- Interactive deployment with Streamlit
""")
