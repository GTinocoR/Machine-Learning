# -*- coding: utf-8 -*-
"""
Proyecto - Aprendizaje de Máquina (2026-2)
Detección de fallas en circuitos eléctricos (PCB) - Task 3 (RepairLabel)
'NotPossibleToRepair' -> 1   |   'FalseScrap' -> 0

Datos: https://github.com/PHME-Datachallenge/Data-Challenge-2022/tree/main/data
Los datos se descargan directo de ese repositorio (requiere internet). Ejecutar:
    python proyecto_pcb_task3.py
Las figuras se guardan en ./outputs y la predicción en ./outputs/submission_file.csv

Librerías: pandas, numpy, scikit-learn, xgboost, optuna, matplotlib
"""

import fnmatch
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.simplefilter("ignore", pd.errors.DtypeWarning)   # Shape(um) trae algunos textos
warnings.simplefilter("ignore", FutureWarning)            # avisos de versiones nuevas
warnings.simplefilter("ignore", UserWarning)
set_config(transform_output="pandas")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# los datos se leen directo del repositorio de GitHub (pandas descomprime los .zip)
DATA_URL = ("https://raw.githubusercontent.com/PHME-Datachallenge/"
            "Data-Challenge-2022/main/data/")
ARCHIVOS = ["AOI_training.csv.zip", "AOI_testing.csv.zip",
            "SPI_training_0.csv.zip", "SPI_training_1.csv.zip",
            "SPI_training_2.csv.zip", "SPI_training_3.csv.zip",
            "SPI_testing_0.csv.zip", "SPI_testing_1.csv.zip"]
# carpeta "outputs" junto a este archivo .py
CARPETA = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CARPETA, "outputs")
SEED = 42
N_TRIALS = 50                                   # trials de Optuna por modelo
KEYS = ["PanelID", "FigureID", "ComponentID"]
TARGET = "RepairLabel"
TARGET_MAP = {"NotPossibleToRepair": 1, "FalseScrap": 0}
SPI_NO_FEATURES = ["PinNumber", "PadID", "Time", "Date", "PadType"]
os.makedirs(OUT_DIR, exist_ok=True)


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, nombre), dpi=150)
    plt.close()


# =============================================================================
# 1. LECTURA DE DATOS
# =============================================================================
def arreglar_llaves(df):
    # En SPI FigureID viene como 1.0 y en AOI como 1: se dejan como texto igual
    for k in KEYS:
        df[k] = df[k].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return df


def llave(df):
    return df["PanelID"] + "|" + df["FigureID"] + "|" + df["ComponentID"]


def buscar(patron):
    # devuelve las URLs del repositorio cuyo nombre coincide con el patrón
    files = [DATA_URL + n for n in ARCHIVOS if fnmatch.fnmatch(n, patron)]
    if not files:
        raise FileNotFoundError(f"No se encontró {patron} en {DATA_URL}")
    return files


def leer_aoi(patron):
    files = buscar(patron)
    return arreglar_llaves(pd.concat([pd.read_csv(f, dtype={"PanelID": str})
                                      for f in files], ignore_index=True))


def leer_spi(patron, llaves_aoi, muestra=0.0):
    """El SPI tiene millones de filas: se lee por bloques y solo se guardan los
    pines de componentes que están en AOI (y una muestra para el EDA)."""
    rng = np.random.default_rng(SEED)
    datos, muestras, nulos, n = [], [], 0, 0
    for f in buscar(patron):
        print("Leyendo", os.path.basename(f))
        for b in pd.read_csv(f, dtype={"PanelID": str}, chunksize=1_000_000):
            b = arreglar_llaves(b)
            for c in b.columns:
                if c not in KEYS + ["Date", "Time", "Result"]:
                    b[c] = pd.to_numeric(b[c], errors="coerce")
                    if b[c].dtype == "float64":
                        b[c] = b[c].astype("float32")      # ahorra memoria
            n += len(b)
            nulos = nulos + b.isna().sum()
            datos.append(b[llave(b).isin(llaves_aoi)])
            if muestra > 0:
                muestras.append(b[rng.random(len(b)) < muestra])
    muestra_df = pd.concat(muestras) if muestras else None
    return pd.concat(datos, ignore_index=True), muestra_df, nulos, n


aoi_tr = leer_aoi("AOI_training*")
aoi_te = leer_aoi("AOI_testing*")
spi_tr, spi_muestra, nulos_spi, n_spi = leer_spi(
    "SPI_training*", set(llave(aoi_tr)), muestra=0.03)
spi_te, _, _, _ = leer_spi("SPI_testing*", set(llave(aoi_te)))

# =============================================================================
# 2. EDA: tendencias, patrones y valores nulos (AOI y SPI)
# =============================================================================
print(f"\nSPI training: {n_spi:,} filas | AOI training: {len(aoi_tr):,} filas")
print("\n% de nulos SPI training:\n", (nulos_spi / n_spi * 100).round(3))
print("\n% de nulos AOI training:\n", (aoi_tr.isna().mean() * 100).round(3))

rl = aoi_tr[TARGET].fillna("NaN (nulo)").value_counts()
print("\nRepairLabel (nivel pin):\n", rl)
ax = rl.plot.bar(color="#4C72B0", figsize=(7, 4))
for i, v in enumerate(rl.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
plt.title("Distribución de RepairLabel en AOI training (nivel pin)")
plt.ylabel("Filas")
plt.xticks(rotation=20)
guardar("fig_eda_repairlabel.png")

validos = aoi_tr[aoi_tr[TARGET].isin(TARGET_MAP)]
pd.crosstab(validos["AOILabel"], validos[TARGET]).plot.barh(stacked=True, figsize=(8, 5))
plt.title("Tipo de defecto AOI vs RepairLabel")
plt.xlabel("Filas")
guardar("fig_eda_aoilabel.png")

num_cols = [c for c in spi_muestra.select_dtypes("number").columns
            if c not in KEYS + SPI_NO_FEATURES]
print("\nEstadística descriptiva SPI (muestra 3%):\n", spi_muestra[num_cols].describe().T)
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for ax, c in zip(axes.ravel(), num_cols):
    s = spi_muestra[c].dropna()
    s.clip(*s.quantile([0.005, 0.995])).hist(bins=60, ax=ax, color="#55A868")
    ax.set_title(c, fontsize=9)
plt.suptitle("Distribución de variables SPI (nivel pin)")
guardar("fig_eda_histogramas_spi.png")

# =============================================================================
# 3. PREPROCESAMIENTO: feature matrix SPI y AOI + integración
# =============================================================================
def top(serie, k=15):
    return list(serie.dropna().astype(str).value_counts().index[:k])


def matriz_spi(spi):
    grupos = [spi[k] for k in KEYS]
    g = spi.groupby(KEYS, sort=False)
    m = g[num_cols].agg(["mean", "std", "min", "max", "median"])
    m.columns = [f"SPI_{c}_{a}" for c, a in m.columns]
    m["SPI_n_pines"] = g.size()
    for col in ["Result", "PadType"]:
        for v in niveles_spi[col]:
            m[f"SPI_{col}_{v}_prop"] = (spi[col].astype(str) == v).groupby(
                grupos, sort=False).mean()
    return m.reset_index()


def matriz_aoi(aoi):
    grupos = [aoi[k] for k in KEYS]
    m = pd.DataFrame({"AOI_n_filas": aoi.groupby(KEYS, sort=False).size()})
    for col in ["AOILabel", "MachineID"]:
        for v in niveles_aoi[col]:
            m[f"AOI_{col}_{v}_n"] = (aoi[col].astype(str) == v).groupby(
                grupos, sort=False).sum()
    # target: NotPossibleToRepair si al menos un pin lo es
    m[TARGET] = aoi[TARGET].map(TARGET_MAP).groupby(grupos, sort=False).max()
    return m.reset_index()


niveles_spi = {c: top(spi_tr[c]) for c in ["Result", "PadType"]}
niveles_aoi = {c: top(aoi_tr[c]) for c in ["AOILabel", "MachineID"]}

# se descartan NotYetClassified y nulos de RepairLabel
aoi_tr_lab = aoi_tr[aoi_tr[TARGET].isin(TARGET_MAP)]
# test: componentes que llegan al operador de reparación
aoi_te_bad = aoi_te[aoi_te["OperatorLabel"] == "Bad"]

train = matriz_aoi(aoi_tr_lab).merge(matriz_spi(spi_tr), on=KEYS, how="left")
test = matriz_aoi(aoi_te_bad).merge(matriz_spi(spi_te), on=KEYS, how="left")
features = [c for c in train.columns if c not in KEYS + [TARGET]]
for c in features:
    if c not in test.columns:
        test[c] = np.nan

X, y, grupos = train[features], train[TARGET].astype(int), train["PanelID"]
etiquetado = test[TARGET].notna()                       # test con etiqueta real
X_te, y_te = test.loc[etiquetado, features], test.loc[etiquetado, TARGET].astype(int)
print(f"\nTrain: {len(X)} componentes, {len(features)} features | "
      f"Test: {len(X_te)} componentes etiquetados")

vc = y.value_counts().sort_index()
plt.figure(figsize=(5, 4))
plt.bar(["FalseScrap (0)", "NotPossibleToRepair (1)"], vc.values,
        color=["#4C72B0", "#C44E52"])
for i, v in enumerate(vc.values):
    plt.text(i, v, f"{v} ({v / vc.sum():.1%})", ha="center", va="bottom")
plt.title("Distribución del target (nivel componente)")
guardar("fig_eda_target_componente.png")

# =============================================================================
# 4. MODELOS (selección de features + imputación dentro del pipeline)
# =============================================================================
class FiltroFeatures(BaseEstimator, TransformerMixin):
    """Quita columnas con >50% nulos, constantes y con |correlación| > 0.95."""

    def fit(self, X, y=None):
        cols = [c for c in X.columns if X[c].isna().mean() <= 0.5 and X[c].nunique() > 1]
        corr = X[cols].corr().abs()
        sup = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        self.keep_ = [c for c in cols if not (sup[c] > 0.95).any()]
        return self

    def transform(self, X):
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


PARAMS_INICIALES = {
    "XGBoost": dict(n_estimators=300, learning_rate=0.1, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, balance=True),
    "RandomForest": dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
                         max_features="sqrt", class_weight="balanced"),
    "LogisticRegression": dict(C=1.0, penalty="l2", class_weight="balanced"),
}


def crear_modelo(nombre, params, y_train):
    p = dict(params)
    if nombre == "XGBoost":
        spw = (y_train == 0).sum() / (y_train == 1).sum() if p.pop("balance") else 1.0
        modelo = XGBClassifier(**p, scale_pos_weight=spw, tree_method="hist",
                               eval_metric="logloss", random_state=SEED, n_jobs=-1)
        return Pipeline([("filtro", FiltroFeatures()), ("modelo", modelo)])
    if nombre == "RandomForest":
        modelo = RandomForestClassifier(**p, random_state=SEED, n_jobs=-1)
        return Pipeline([("filtro", FiltroFeatures()),
                         ("imputador", SimpleImputer(strategy="median")),
                         ("modelo", modelo)])
    modelo = LogisticRegression(**p, solver="liblinear", max_iter=3000, random_state=SEED)
    return Pipeline([("filtro", FiltroFeatures()),
                     ("imputador", SimpleImputer(strategy="median")),
                     ("escalador", StandardScaler()), ("modelo", modelo)])


def importancia(pipe):
    m = pipe.named_steps["modelo"]
    imp = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(m.coef_[0])
    imp = pd.Series(imp / imp.sum(), index=pipe.named_steps["filtro"].keep_)
    return imp.reindex(features).fillna(0)


# Validación cruzada agrupada por panel (componentes de un mismo panel no se separan)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)


def validacion_cruzada(nombre, params):
    filas, imps = [], []
    for fold, (tr, va) in enumerate(cv.split(X, y, grupos), start=1):
        pipe = crear_modelo(nombre, params, y.iloc[tr]).fit(X.iloc[tr], y.iloc[tr])
        p_tr, p_va = pipe.predict(X.iloc[tr]), pipe.predict(X.iloc[va])
        filas.append({"fold": fold,
                      "train_macro_f1": f1_score(y.iloc[tr], p_tr, average="macro"),
                      "val_macro_f1": f1_score(y.iloc[va], p_va, average="macro"),
                      "val_accuracy": accuracy_score(y.iloc[va], p_va)})
        imps.append(importancia(pipe))
    return pd.DataFrame(filas), pd.concat(imps, axis=1)


# =============================================================================
# 5. BASELINE DE VALOR CONSTANTE
# =============================================================================
clase_mayor = int(y.mode()[0])
bvc_val = np.mean([f1_score(y.iloc[va], np.full(len(va), int(y.iloc[tr].mode()[0])),
                            average="macro") for tr, va in cv.split(X, y, grupos)])
bvc_test = f1_score(y_te, np.full(len(y_te), clase_mayor), average="macro")
print(f"\nBaseline constante (clase {clase_mayor}): Macro F1 val CV = {bvc_val:.4f} | "
      f"test = {bvc_test:.4f} | accuracy test = "
      f"{accuracy_score(y_te, np.full(len(y_te), clase_mayor)):.4f}")

# =============================================================================
# 6. VALIDACIÓN CRUZADA CON HIPERPARÁMETROS INICIALES
# =============================================================================
modelos = ["XGBoost", "RandomForest", "LogisticRegression"]
f1_inicial = {}
for nombre in modelos:
    folds, _ = validacion_cruzada(nombre, PARAMS_INICIALES[nombre])
    f1_inicial[nombre] = folds["val_macro_f1"].mean()
    print(f"CV inicial {nombre}: Macro F1 val = {f1_inicial[nombre]:.4f}")

# =============================================================================
# 7. OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA (maximiza Macro F1 promedio CV)
# =============================================================================
def espacio(trial, nombre):
    if nombre == "XGBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            balance=trial.suggest_categorical("balance", [True, False]))
    if nombre == "RandomForest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_categorical("max_depth", [None, 6, 10, 15, 20, 30]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            class_weight=trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]))
    return dict(
        C=trial.suggest_float("C", 1e-3, 100.0, log=True),
        penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
        class_weight=trial.suggest_categorical("class_weight", ["balanced", None]))


def objetivo(trial, nombre):
    params = espacio(trial, nombre)
    f1s = []
    for tr, va in cv.split(X, y, grupos):
        pipe = crear_modelo(nombre, params, y.iloc[tr]).fit(X.iloc[tr], y.iloc[tr])
        f1s.append(f1_score(y.iloc[va], pipe.predict(X.iloc[va]), average="macro"))
    return np.mean(f1s)


mejores, estudios = {}, {}
for nombre in modelos:
    estudio = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
    estudio.optimize(lambda t: objetivo(t, nombre), n_trials=N_TRIALS)
    mejores[nombre], estudios[nombre] = estudio.best_params, estudio
    print(f"Optuna {nombre}: Macro F1 CV = {estudio.best_value:.4f} "
          f"(inicial {f1_inicial[nombre]:.4f}) | {estudio.best_params}")

# =============================================================================
# 8. COMPARACIÓN DE MODELOS: train (CV), validación (CV) y test
# =============================================================================
resumen, detalle = [], {}
for nombre in modelos:
    folds, imps = validacion_cruzada(nombre, mejores[nombre])
    final = crear_modelo(nombre, mejores[nombre], y).fit(X, y)   # train + validación
    pred = final.predict(X_te)
    detalle[nombre] = (folds, imps, final, pred)
    resumen.append({"modelo": nombre,
                    "train_cv": folds["train_macro_f1"].mean(),
                    "val_cv": folds["val_macro_f1"].mean(),
                    "val_cv_std": folds["val_macro_f1"].std(),
                    "test": f1_score(y_te, pred, average="macro"),
                    "accuracy_test": accuracy_score(y_te, pred)})
resumen = pd.DataFrame(resumen)
print("\nMacro F1 de los modelos optimizados:\n", resumen.round(4).to_string(index=False))

xx, w = np.arange(len(modelos)), 0.27
plt.figure(figsize=(8, 4.8))
plt.bar(xx - w, resumen["train_cv"], w, label="Train (prom. CV)")
plt.bar(xx, resumen["val_cv"], w, yerr=resumen["val_cv_std"], capsize=3,
        label="Validación (prom. CV)")
plt.bar(xx + w, resumen["test"], w, label="Test")
plt.axhline(bvc_test, ls="--", c="gray", label="Baseline constante (test)")
plt.xticks(xx, modelos)
plt.ylabel("Macro F1")
plt.ylim(0, 1)
plt.title("Comparación de modelos")
plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
guardar("fig_comparacion_modelos.png")

# =============================================================================
# 9. MEJOR MODELO
# =============================================================================
mejor = resumen.sort_values("val_cv", ascending=False)["modelo"].iloc[0]
folds, imps, final, pred = detalle[mejor]
print(f"\nMejor modelo: {mejor}")

# 9.1 Métricas por fold (train y validación)
print("\nMétricas por fold:\n", folds.round(4).to_string(index=False))
plt.figure(figsize=(7, 4))
plt.plot(folds["fold"], folds["train_macro_f1"], "o-", label="Train")
plt.plot(folds["fold"], folds["val_macro_f1"], "s-", label="Validación")
plt.xticks(folds["fold"])
plt.xlabel("Fold")
plt.ylabel("Macro F1")
plt.title(f"Macro F1 por fold - {mejor}")
plt.legend()
guardar("fig_folds_mejor_modelo.png")


def grafico_importancia(imp, titulo, archivo, err=None):
    imp = imp.sort_values(ascending=False).head(20)[::-1]
    plt.figure(figsize=(8, 8.5))
    plt.barh(imp.index, imp.values, color="#4C72B0", capsize=3,
             xerr=None if err is None else err.reindex(imp.index).values)
    plt.xlabel("Importancia normalizada")
    plt.title(titulo)
    guardar(archivo)


# 9.2 Feature importance promedio de la CV
imp_cv = imps.mean(axis=1)
print("\nTop 10 feature importance promedio CV:\n",
      imp_cv.sort_values(ascending=False).head(10).round(4))
grafico_importancia(imp_cv, f"Feature importance promedio CV - {mejor}",
                    "fig_feature_importance_cv.png", err=imps.std(axis=1))

# 9.3 Matriz de confusión (test)
cm = confusion_matrix(y_te, pred, labels=[0, 1])
print("\nMatriz de confusión (test):\n", cm)
print(classification_report(y_te, pred, target_names=["FalseScrap", "NotPossibleToRepair"],
                            digits=4))
etiquetas = ["FalseScrap (0)", "NotPossibleToRepair (1)"]
plt.figure(figsize=(5, 4.5))
plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], etiquetas, fontsize=8)
plt.yticks([0, 1], etiquetas, fontsize=8)
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title(f"Matriz de confusión (test) - {mejor}")
guardar("fig_matriz_confusion_test.png")

# 9.4 Feature importance del modelo final (train + validación)
imp_final = importancia(final)
print("\nTop 10 feature importance modelo final:\n",
      imp_final.sort_values(ascending=False).head(10).round(4))
grafico_importancia(imp_final, f"Feature importance modelo final (train+val) - {mejor}",
                    "fig_feature_importance_final.png")

# Evolución de Optuna para el mejor modelo
valores = [t.value for t in estudios[mejor].trials]
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(valores) + 1), valores, "o", alpha=0.5, label="Trial")
plt.plot(range(1, len(valores) + 1), np.maximum.accumulate(valores), "-",
         color="#C44E52", label="Mejor acumulado")
plt.xlabel("Trial")
plt.ylabel("Macro F1 promedio (CV)")
plt.title(f"Optimización con Optuna - {mejor}")
plt.legend()
guardar(f"fig_optuna_{mejor}.png")

# =============================================================================
# 10. SUBMISSION PARA KAGGLE (modelo entrenado con train + validación)
# =============================================================================
submission_file = pd.DataFrame({
    "ID": test["PanelID"] + test["FigureID"] + test["ComponentID"],
    "RepairLabel": final.predict(test[features]).astype(int)})
submission_file.to_csv(os.path.join(OUT_DIR, "submission_file.csv"), index=False)
print(f"\nsubmission_file.csv: {len(submission_file)} filas")
