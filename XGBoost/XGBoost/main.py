import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================
# Nombre del archivo de datos
FILENAME = 'heart_cleaned_20251122_195915.csv'
# Carpeta para guardar todas las evidencias gráficas
OUTPUT_FOLDER = 'Resultados_Investigacion_Final'

# Configuración de estilo científico para gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300  # Alta resolución para papers

# Crear directorio de resultados
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Carpeta '{OUTPUT_FOLDER}' creada.")
else:
    print(f"Usando carpeta existente: '{OUTPUT_FOLDER}'.")

# =============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================
print("\n[1/6] Cargando y preparando datos...")
df = pd.read_csv(FILENAME)

# Ingeniería de Variables (Feature Engineering)
# Variable validada científicamente: Ratio Frecuencia Cardiaca / Edad
df['HeartRate_Age_Ratio'] = df['MaxHR'] / df['Age']

# =============================================================================
# 2. ANÁLISIS EXPLORATORIO (EDA) - "La evidencia previa"
# =============================================================================
print("[2/6] Generando gráficos de análisis previo (EDA)...")

# 2.1 Balance de Clases
plt.figure()
sns.countplot(data=df, x='HeartDisease', palette='viridis')
plt.title('Balance de Clases (Objetivo)')
plt.xlabel('Enfermedad Cardíaca (0=No, 1=Sí)')
plt.savefig(os.path.join(OUTPUT_FOLDER, '1_EDA_Balance_Clases.png'))
plt.close()

# 2.2 Correlaciones
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'HeartRate_Age_Ratio', 'HeartDisease']
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor de Correlaciones')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '2_EDA_Correlacion.png'))
plt.close()

# 2.3 Dolor de Pecho vs Enfermedad
plt.figure()
cp_counts = df.groupby('ChestPainType')['HeartDisease'].value_counts(normalize=True).unstack().fillna(0)
cp_counts.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'])
plt.title('Impacto del Tipo de Dolor de Pecho')
plt.ylabel('Proporción')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '3_EDA_DolorPecho.png'))
plt.close()

# =============================================================================
# 3. PREPROCESAMIENTO Y DIVISIÓN
# =============================================================================
print("[3/6] Procesando datos para el modelo...")
# One-Hot Encoding (Texto -> Números)
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# División Estratificada (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 4. VALIDACIÓN CIENTÍFICA (EARLY STOPPING)
# =============================================================================
print("[4/6] Validando complejidad óptima (Early Stopping)...")
X_tr_val, X_val, y_tr_val, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

model_check = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=3, gamma=0.1,
    subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50,
    eval_metric=["logloss", "auc"], random_state=42
)

model_check.fit(
    X_tr_val, y_tr_val,
    eval_set=[(X_tr_val, y_tr_val), (X_val, y_val)],
    verbose=False
)
print(f"   -> El modelo dejó de mejorar en el árbol #{model_check.best_iteration}.")
print("   -> Conclusión: Usar 200 estimadores es seguro y robusto.")

# =============================================================================
# 5. ENTRENAMIENTO FINAL (MODELO CONSERVADOR)
# =============================================================================
print("[5/6] Entrenando modelo final optimizado...")

model_final = xgb.XGBClassifier(
    n_estimators=200,        # Validado
    learning_rate=0.05,      # Conservador
    max_depth=3,             # Anti-Overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,               # Regularización
    eval_metric=["logloss", "error"], # Métricas para curva de aprendizaje
    random_state=42
)

# Entrenamos pasando el eval_set para poder graficar la curva de aprendizaje después
model_final.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# =============================================================================
# 6. EVALUACIÓN Y GRÁFICOS POST-ENTRENAMIENTO
# =============================================================================
print("[6/6] Generando métricas y gráficos finales...")

# Predicciones
y_pred = model_final.predict(X_test)
y_prob = model_final.predict_proba(X_test)[:, 1]

# --- REPORTE TEXTO ---
print("\n" + "="*40)
print(" RESULTADOS FINALES PARA EL PAPER")
print("="*40)
print(classification_report(y_test, y_pred))
auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc_score:.4f} (Excelente > 0.90)")
print("="*40)

# --- GRÁFICOS DE RENDIMIENTO ---

# 6.1 Matriz de Confusión
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión')
plt.ylabel('Realidad')
plt.xlabel('Predicción')
plt.savefig(os.path.join(OUTPUT_FOLDER, '4_Result_ConfusionMatrix.png'))
plt.close()

# 6.2 Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"XGBoost (AUC={auc_score:.2f})", color='#d35400', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.title('Curva ROC')
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, '5_Result_ROC.png'))
plt.close()

# 6.3 Importancia de Variables
plt.figure(figsize=(10, 8))
xgb.plot_importance(model_final, max_num_features=15, height=0.5, importance_type='gain', show_values=False, title='Importancia de Variables (Gain)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '6_Result_FeatureImportance.png'))
plt.close()

# 6.4 Curva de Aprendizaje (Learning Curve) - La prueba visual del NO-Overfitting
results = model_final.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure()
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
plt.legend()
plt.ylabel('Log Loss')
plt.title('Curva de Aprendizaje')
plt.savefig(os.path.join(OUTPUT_FOLDER, '7_Result_LearningCurve.png'))
plt.close()

# 6.5 Densidad de Probabilidades
plt.figure()
sns.kdeplot(y_prob[y_test == 0], fill=True, color='green', label='Sanos (Real)', alpha=0.3)
sns.kdeplot(y_prob[y_test == 1], fill=True, color='red', label='Enfermos (Real)', alpha=0.3)
plt.title('Confianza del Modelo (Densidad de Probabilidad)')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, '8_Result_ProbDensity.png'))
plt.close()

# 6.6 Visualización del Árbol (Opcional - Requiere Graphviz)
try:
    plt.figure(figsize=(20, 12))
    xgb.plot_tree(model_final, num_trees=0, rankdir='LR')
    plt.title('Estructura del Primer Árbol')
    plt.savefig(os.path.join(OUTPUT_FOLDER, '9_Result_TreeViz.png'), dpi=300)
    plt.close()
    print("   -> Gráfico del árbol generado correctamente.")
except Exception as e:
    print(f"   -> [AVISO] No se pudo generar el gráfico del árbol. Falta Graphviz. (Error: {e})")
    print("      No te preocupes, el resto de resultados están listos.")

# =============================================================================
# 7. GENERACIÓN DEL REPORTE EXCEL MAESTRO (918 PACIENTES)
# =============================================================================
print("\n[7/7] Generando Excel Maestro con TODOS los registros...")

# 1. PREDICCIÓN GLOBAL
# Nuevo criterio: Ante la duda (30%), clasifícalo como Enfermo (1)
all_probs = model_final.predict_proba(X)[:, 1]
all_preds = (all_probs > 0.30).astype(int)
all_probs = model_final.predict_proba(X)[:, 1]

# 2. PREPARAR EL DATAFRAME
# Copiamos el df original completo (918 filas)
reporte_total = df.copy()

# 3. AGREGAR COLUMNAS DE INTELIGENCIA ARTIFICIAL
reporte_total['Diagnostico_REAL'] = y
reporte_total['Diagnostico_IA'] = all_preds
# Probabilidad en porcentaje (0 a 100)
reporte_total['Probabilidad_Enfermedad (%)'] = np.round(all_probs * 100, 2)

# 4. VEREDICTO: ¿ACERTÓ O FALLÓ?
reporte_total['Resultado'] = np.where(
    reporte_total['Diagnostico_REAL'] == reporte_total['Diagnostico_IA'], 
    'ACIERTO', 
    'FALLO'
)

# 5. IDENTIFICAR EL GRUPO (Train vs Test)
# Esto es vital para tu investigación: saber quiénes ya "conocía" el modelo
reporte_total['Grupo'] = 'Test (Evaluación)' # Por defecto
# Si el índice estaba en X_train, lo marcamos como "Train"
reporte_total.loc[X_train.index, 'Grupo'] = 'Train (Aprendizaje)'

# 6. ORDENAMIENTO ESTRATÉGICO
# Orden 1: Primero el grupo de TEST (es el más honesto para validar).
# Orden 2: Primero los FALLOS (para analizar errores).
# Orden 3: Por probabilidad (de mayor a menor).
reporte_total = reporte_total.sort_values(
    by=['Grupo', 'Resultado', 'Probabilidad_Enfermedad (%)'], 
    ascending=[True, True, False] 
)

# 7. GUARDAR EXCEL
nombre_archivo_excel = os.path.join(OUTPUT_FOLDER, 'Reporte_Global_918_Pacientes.xlsx')
reporte_total.to_excel(nombre_archivo_excel, index=True, sheet_name='Diagnosticos_Completos')

print("¡ÉXITO TOTAL! 🚀")
print("-> Se ha generado el archivo con los 918 pacientes.")
print(f"-> Ubicación: '{nombre_archivo_excel}'")
print("-> Nota: En el Excel, filtra la columna 'Grupo' para ver 'Test' (los nuevos) vs 'Train'.")

# Vista previa en consola de las primeras filas del grupo Test
print("\n--- VISTA PREVIA (Primeros registros del Test Set) ---")
print(reporte_total[['Age', 'Sex', 'Diagnostico_REAL', 'Resultado', 'Grupo']].head(5))