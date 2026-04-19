"""
train.py
--------
Full ML training pipeline for Spam Mail Detector.

Stages:
  1.  Load & EDA
  2.  Text preprocessing + feature engineering
  3.  Train/test split (stratified)
  4.  Baseline models  (BoW NB, TF-IDF NB)
  5.  Advanced models  (Logistic Regression, LinearSVC, Random Forest)
  6.  Hyperparameter tuning (GridSearchCV on best model)
  7.  Ensemble  (Voting Classifier)
  8.  Evaluation – Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
  9.  Cross-validation (5-fold)
  10. Feature importance analysis
  11. Rich visualisation dashboard (8 plots)
  12. Save best model to models/
"""

import os, sys, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes      import MultinomialNB, ComplementNB
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import LinearSVC
from sklearn.ensemble         import RandomForestClassifier, VotingClassifier
from sklearn.pipeline         import Pipeline, FeatureUnion
from sklearn.preprocessing    import StandardScaler
from sklearn.base             import BaseEstimator, TransformerMixin
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score, roc_auc_score,
                                      confusion_matrix, roc_curve, auc,
                                      classification_report)
from scipy.sparse             import hstack, issparse

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import clean_text, engineer_features, FEATURE_COLS

# ── colours ─────────────────────────────────────────────────────────────────
C_SPAM  = "#E24B4A"
C_HAM   = "#1D9E75"
C_DARK  = "#2C2C2A"
C_LIGHT = "#F1EFE8"
PALETTE = [C_HAM, C_SPAM]

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
BANNER = "=" * 65

def load_data():
    for path, sep in [("data/sms.tsv","\t"), ("data/sms.csv",",")]:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] == 2:
                df.columns = ["label","message"]
            print(f"  Loaded {path}  →  {len(df):,} rows")
            return df
    print("  No dataset file found. Run:  python data/generate_dataset.py")
    sys.exit(1)

print(BANNER)
print("  SPAM MAIL DETECTOR  ·  Full ML Training Pipeline")
print(BANNER)

df = load_data()
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EDA] Dataset Overview")
print(f"  Total     : {len(df):,}")
print(f"  Spam      : {df['label_num'].sum():,}  ({df['label_num'].mean()*100:.1f}%)")
print(f"  Ham       : {(df['label_num']==0).sum():,}  ({(df['label_num']==0).mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESS + FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[PREP] Cleaning text & engineering features …")
df["clean"] = df["message"].apply(clean_text)
df = engineer_features(df)

spam_df = df[df["label"]=="spam"]
ham_df  = df[df["label"]=="ham"]

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_text  = df["clean"]
X_feats = df[FEATURE_COLS].values.astype(float)
y       = df["label_num"]

(X_text_train, X_text_test,
 X_feat_train, X_feat_test,
 y_train, y_test) = train_test_split(
    X_text, X_feats, y,
    test_size=0.2, random_state=42, stratify=y
)
print(f"\n[SPLIT] Train={len(X_text_train):,}  Test={len(X_text_test):,}  (80/20 stratified)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODELS
# ─────────────────────────────────────────────────────────────────────────────
# Shared TF-IDF for pipelines
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=8000,
                        sublinear_tf=True, min_df=2)

models: dict = {
    "Naive Bayes (BoW)": Pipeline([
        ("vec", CountVectorizer(ngram_range=(1,2), max_features=5000)),
        ("clf", MultinomialNB(alpha=0.1)),
    ]),
    "Complement NB (TF-IDF)": Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)),
        ("clf", ComplementNB(alpha=0.1)),
    ]),
    "Logistic Regression": Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=5, max_iter=1000, random_state=42)),
    ]),
    "Linear SVM": Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, random_state=42, max_iter=2000)),
    ]),
}

print("\n[TRAIN] Baseline & Advanced Models")
print("-" * 65)
print(f"  {'Model':<28} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6} {'AUC':>7}")
print("-" * 65)

results: dict = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in models.items():
    pipe.fit(X_text_train, y_train)
    y_pred = pipe.predict(X_text_test)

    try:
        proba = pipe.predict_proba(X_text_test)[:,1]
        roc   = roc_auc_score(y_test, proba)
    except Exception:
        proba = None
        roc   = float("nan")

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    cv_f1 = cross_val_score(pipe, X_text, y, cv=cv, scoring="f1").mean()

    results[name] = dict(
        acc=acc, prec=prec, rec=rec, f1=f1, roc=roc,
        cm=cm, proba=proba, y_pred=y_pred, cv_f1=cv_f1, pipe=pipe
    )
    print(f"  {name:<28} {acc*100:>5.1f}% {prec*100:>5.1f}% {rec*100:>6.1f}% {f1*100:>5.1f}% {roc if not np.isnan(roc) else 'N/A':>7}")

print("-" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING  (Logistic Regression)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TUNE] GridSearchCV on Logistic Regression …")
param_grid = {
    "vec__max_features": [5000, 8000],
    "vec__ngram_range":  [(1,1),(1,2)],
    "clf__C":            [1, 5, 10],
}
grid = GridSearchCV(
    Pipeline([
        ("vec", TfidfVectorizer(sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=0
)
grid.fit(X_text_train, y_train)
best_params = grid.best_params_
best_pipe   = grid.best_estimator_
print(f"  Best params : {best_params}")
print(f"  CV F1       : {grid.best_score_*100:.2f}%")

y_pred_tuned = best_pipe.predict(X_text_test)
proba_tuned  = best_pipe.predict_proba(X_text_test)[:,1]
results["LR (Tuned)"] = dict(
    acc   = accuracy_score(y_test, y_pred_tuned),
    prec  = precision_score(y_test, y_pred_tuned, zero_division=0),
    rec   = recall_score(y_test, y_pred_tuned, zero_division=0),
    f1    = f1_score(y_test, y_pred_tuned, zero_division=0),
    roc   = roc_auc_score(y_test, proba_tuned),
    cm    = confusion_matrix(y_test, y_pred_tuned),
    proba = proba_tuned, y_pred=y_pred_tuned,
    cv_f1 = grid.best_score_, pipe=best_pipe
)
r = results["LR (Tuned)"]
print(f"  Test  F1    : {r['f1']*100:.2f}%   AUC: {r['roc']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ENSEMBLE (Voting Classifier)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ENSEMBLE] Building Soft Voting Classifier …")

# Need probability-capable models only
lr_pipe  = Pipeline([
    ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)),
    ("clf", LogisticRegression(C=5, max_iter=1000, random_state=42)),
])
cnb_pipe = Pipeline([
    ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)),
    ("clf", ComplementNB(alpha=0.1)),
])
nb_pipe  = Pipeline([
    ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)),
    ("clf", MultinomialNB(alpha=0.1)),
])

# VotingClassifier expects a flat estimator; wrap pipelines
from sklearn.pipeline import make_pipeline

class TextSelector(BaseEstimator, TransformerMixin):
    """Returns the pandas Series / array unchanged (passthrough for text)."""
    def fit(self, X, y=None): return self
    def transform(self, X): return X

ensemble = VotingClassifier(
    estimators=[("lr", lr_pipe), ("cnb", cnb_pipe), ("nb", nb_pipe)],
    voting="soft"
)
ensemble.fit(X_text_train, y_train)
y_pred_ens = ensemble.predict(X_text_test)
proba_ens  = ensemble.predict_proba(X_text_test)[:,1]
results["Ensemble (Voting)"] = dict(
    acc   = accuracy_score(y_test, y_pred_ens),
    prec  = precision_score(y_test, y_pred_ens, zero_division=0),
    rec   = recall_score(y_test, y_pred_ens, zero_division=0),
    f1    = f1_score(y_test, y_pred_ens, zero_division=0),
    roc   = roc_auc_score(y_test, proba_ens),
    cm    = confusion_matrix(y_test, y_pred_ens),
    proba = proba_ens, y_pred=y_pred_ens,
    cv_f1 = cross_val_score(ensemble, X_text, y, cv=3, scoring="f1").mean(),
    pipe  = ensemble
)
r = results["Ensemble (Voting)"]
print(f"  Ensemble F1 : {r['f1']*100:.2f}%   AUC: {r['roc']:.4f}")

# pick best overall model
best_name = max(results, key=lambda k: results[k]["f1"])
print(f"\n  ★  Best model: {best_name}  (F1={results[best_name]['f1']*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
save_path = "models/best_model.pkl"
with open(save_path, "wb") as f:
    pickle.dump({"model": results[best_name]["pipe"], "name": best_name}, f)
print(f"  Model saved → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. CROSS-VALIDATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[CV] 5-Fold Cross-Validation F1 Scores")
for name, r in results.items():
    print(f"  {name:<28} CV-F1 = {r['cv_f1']*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE IMPORTANCE (from LR)
# ─────────────────────────────────────────────────────────────────────────────
lr_result = results["Logistic Regression"]
lr_vect   = lr_result["pipe"].named_steps["vec"]
lr_clf    = lr_result["pipe"].named_steps["clf"]
feat_names = lr_vect.get_feature_names_out()
coefs      = lr_clf.coef_[0]
top_spam_idx = coefs.argsort()[-20:][::-1]
top_ham_idx  = coefs.argsort()[:20]
top_spam_words  = [(feat_names[i], coefs[i]) for i in top_spam_idx]
top_ham_words   = [(feat_names[i], -coefs[i]) for i in top_ham_idx]

# ─────────────────────────────────────────────────────────────────────────────
# 11. VISUALISATION DASHBOARD (8 plots)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[VIZ] Building visualisation dashboard …")

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Class distribution ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts = df["label"].value_counts()[["ham","spam"]]
bars = ax1.bar(["Ham","Spam"], counts.values, color=PALETTE, width=0.5, edgecolor="white", linewidth=2)
ax1.set_title("Class Distribution", fontweight="bold", fontsize=13)
ax1.set_ylabel("Message count")
for b, v in zip(bars, counts.values):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+30,
             f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)
ax1.set_ylim(0, max(counts.values)*1.25)
ax1.spines[["top","right"]].set_visible(False)

# ── Plot 2: Message length distributions ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for label, color, name in [("ham",C_HAM,"Ham"),("spam",C_SPAM,"Spam")]:
    vals = df[df["label"]==label]["feat_length"]
    ax2.hist(vals, bins=40, alpha=0.65, color=color, label=f"{name} (μ={vals.mean():.0f})", density=True)
ax2.set_title("Message Length Distribution", fontweight="bold", fontsize=13)
ax2.set_xlabel("Characters")
ax2.set_ylabel("Density")
ax2.legend(fontsize=9)
ax2.spines[["top","right"]].set_visible(False)

# ── Plot 3: Caps ratio boxplot ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
data_caps = [df[df["label"]=="ham"]["feat_caps_ratio"].values,
             df[df["label"]=="spam"]["feat_caps_ratio"].values]
bp = ax3.boxplot(data_caps, patch_artist=True, widths=0.4,
                 medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax3.set_xticklabels(["Ham","Spam"])
ax3.set_title("CAPS Ratio by Class", fontweight="bold", fontsize=13)
ax3.set_ylabel("Fraction of uppercase letters")
ax3.spines[["top","right"]].set_visible(False)

# ── Plot 4: Model F1 comparison ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
model_names = list(results.keys())
f1_vals     = [results[m]["f1"]*100 for m in model_names]
short_names = ["NB (BoW)","CNB (TF-IDF)","Log.Reg.","Lin.SVM","LR (Tuned)","Ensemble"]
bar_colors  = [C_HAM if v < max(f1_vals) else C_SPAM for v in f1_vals]
hbars = ax4.barh(short_names, f1_vals, color=bar_colors, edgecolor="white", linewidth=1.5)
ax4.set_title("F1 Score — Model Comparison", fontweight="bold", fontsize=13)
ax4.set_xlabel("F1 Score (%)")
ax4.set_xlim(min(f1_vals)-2, 101)
for bar, val in zip(hbars, f1_vals):
    ax4.text(val+0.1, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=10)
ax4.axvline(x=95, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax4.spines[["top","right"]].set_visible(False)

# ── Plot 5: 5-Fold CV scores ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
cv_vals = [results[m]["cv_f1"]*100 for m in model_names]
ax5.barh(short_names, cv_vals, color=[C_HAM]*len(cv_vals), alpha=0.75, edgecolor="white")
ax5.set_title("5-Fold CV F1 Score", fontweight="bold", fontsize=13)
ax5.set_xlabel("Mean F1 (%)")
ax5.set_xlim(min(cv_vals)-2, 101)
for i,(bar,val) in enumerate(zip(ax5.patches, cv_vals)):
    ax5.text(val+0.1, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=9)
ax5.spines[["top","right"]].set_visible(False)

# ── Plot 6: Confusion matrix (best model) ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
cm_best = results[best_name]["cm"]
sns.heatmap(cm_best, annot=True, fmt="d", ax=ax6,
            cmap=sns.color_palette(["#E1F5EE","#1D9E75"], as_cmap=True),
            xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"],
            linewidths=2, linecolor="white", cbar=False, annot_kws={"size":13})
ax6.set_title(f"Confusion Matrix\n({best_name})", fontweight="bold", fontsize=13)
ax6.set_ylabel("True Label"); ax6.set_xlabel("Predicted Label")

# ── Plot 7: ROC Curves (all models with proba) ───────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
roc_colors = ["#378ADD","#1D9E75","#E24B4A","#BA7517","#533AB7"]
ci = 0
for name, r in results.items():
    if r["proba"] is not None and not np.isnan(r["roc"]):
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax7.plot(fpr, tpr, lw=1.8, color=roc_colors[ci % len(roc_colors)],
                 label=f"{name.split()[0]} (AUC={r['roc']:.3f})")
        ci += 1
ax7.plot([0,1],[0,1],"--", color="gray", lw=1, alpha=0.5)
ax7.fill_between([0,1],[0,1], alpha=0.05, color="gray")
ax7.set_title("ROC Curves — All Models", fontweight="bold", fontsize=13)
ax7.set_xlabel("False Positive Rate"); ax7.set_ylabel("True Positive Rate")
ax7.legend(fontsize=8, loc="lower right")
ax7.spines[["top","right"]].set_visible(False)

# ── Plot 8: Top spam-indicator features (LR coefficients) ───────────────────
ax8 = fig.add_subplot(gs[2, 2])
words8  = [w for w,_ in top_spam_words[:12]]
scores8 = [s for _,s in top_spam_words[:12]]
ypos    = range(len(words8))
ax8.barh(ypos, scores8[::-1], color=C_SPAM, alpha=0.85, edgecolor="white")
ax8.set_yticks(ypos); ax8.set_yticklabels(words8[::-1], fontsize=9)
ax8.set_title("Top Spam Signals\n(LR coefficients)", fontweight="bold", fontsize=13)
ax8.set_xlabel("Coefficient value")
ax8.spines[["top","right"]].set_visible(False)

# ── Plot 9: Word clouds (full bottom row) ────────────────────────────────────
spam_text = " ".join(df[df["label"]=="spam"]["clean"].tolist())
ham_text  = " ".join(df[df["label"]=="ham"]["clean"].tolist())

ax9  = fig.add_subplot(gs[3, :2])
wc_spam = WordCloud(width=900, height=320, background_color="white",
                    colormap="Reds", max_words=100,
                    collocations=False).generate(spam_text)
ax9.imshow(wc_spam, interpolation="bilinear")
ax9.axis("off"); ax9.set_title("Spam Word Cloud", fontweight="bold", fontsize=13)

ax10 = fig.add_subplot(gs[3, 2])
wc_ham = WordCloud(width=420, height=320, background_color="white",
                   colormap="Greens", max_words=80,
                   collocations=False).generate(ham_text)
ax10.imshow(wc_ham, interpolation="bilinear")
ax10.axis("off"); ax10.set_title("Ham Word Cloud", fontweight="bold", fontsize=13)

fig.suptitle("Spam Mail Detector — Complete ML Results Dashboard",
             fontsize=17, fontweight="bold", y=1.005)

os.makedirs("outputs", exist_ok=True)
out_path = "outputs/results_dashboard.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  Dashboard saved → {out_path}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 12. FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + BANNER)
print("  FINAL RESULTS SUMMARY")
print(BANNER)
print(f"  {'Model':<28} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6} {'AUC':>7} {'CV-F1':>7}")
print("-" * 65)
for name, r in results.items():
    auc_str = f"{r['roc']:.3f}" if not np.isnan(r['roc']) else "  N/A"
    print(f"  {name:<28} {r['acc']*100:>5.1f}% {r['prec']*100:>5.1f}% "
          f"{r['rec']*100:>6.1f}% {r['f1']*100:>5.1f}% {auc_str:>7} {r['cv_f1']*100:>6.1f}%")
print("-" * 65)
print(f"\n  ★  Best: {best_name}")
print(f"     F1 = {results[best_name]['f1']*100:.2f}%   "
      f"AUC = {results[best_name]['roc']:.4f}")
print(BANNER)
print("  Training complete. Run:  streamlit run app.py")
print(BANNER)
