# Proiect Rețele Neuronale - Clasificarea Comportamentului Agenților Mobili

## 1. Identificare Proiect

| Câmp | Valoare                                            |
|------|----------------------------------------------------|
| **Student** | Poșchină Liliana Andreea                           |
| **Grupa / Specializare** | 631AB / Informatică Industrială                    |
| **Disciplina** | Rețele Neuronale                                   |
| **Instituție** | POLITEHNICA București – FIIR                       |
| **Link Repository GitHub** | https://github.com/poschinaandreea/proiectRN.git                                 |
| **Acces Repository** | Public                                             |
| **Stack Tehnologic** | Python (TensorFlow/Keras, Streamlit, Scikit-Learn) |
| **Domeniul Industrial de Interes (DII)** | Robotică / Sisteme Autonome                        |
| **Tip Rețea Neuronală** | MLP (Multi-Layer Perceptron)                       |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 5 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 74.40% | 72.90% | -1.50% | ✓ |
| F1-Score (Macro) | ≥0.65 | 0.65 | 0.70 | +0.05 | ✓ |
| Latență Inferență | < 100 ms | ~15 ms | ~12 ms | -3 ms | ✓ |
| Contribuție Date Originale | ≥40% | 100% | 100% | - | ✓ |
| Nr. Experimente Optimizare | ≥4 | - | 5 | - | ✓ |

### Declarație de Originalitate & Politica de Utilizare AI

**Confirmare explicită:**

| Nr. | Cerință | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1 | Modelul RN a fost antrenat **de la zero** | [X] DA |
| 2 | Minimum **40% din date sunt contribuție originală** | [X] DA |
| 3 | Codul este propriu sau sursele externe sunt **citate explicit** | [X] DA |
| 4 | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** | [X] DA |
| 5 | Pot explica și justifica **fiecare decizie importantă** | [X] DA |

**Semnătură student:** Poșchină Liliana Andreea

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz
Proiectul abordează necesitatea monitorizării agenților mobili într-un mediu industrial. Problema principală este identificarea automată a comportamentelor anormale (opriri nejustificate, trasee neregulate) care pot indica defecțiuni tehnice sau obstacole, reducând astfel timpul de nefuncționare.

### 2.2 Beneficii Măsurabile Urmărite
1. Clasificarea tipului de deplasare cu o acuratețe de peste 70%.
2. Reducerea monitorizării manuale prin alertarea automată a operatorului.
3. Detectarea anomaliilor cu un F1-Score echilibrat (0.70) pentru a evita alarmele false.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Detectarea anomaliilor | Clasificare binară a deplasării | RN (MLP) | F1-Score ≥ 0.65 |
| Automatizarea analizei | Procesare automată date senzori | Preprocessing + RN | Accuracy ≥ 70% |
| Interfață operator | Vizualizarea predicției în UI | Web Service (Streamlit) | Latență < 100ms |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Simulare (Generare programatică) |
| **Sursa concretă** | Script propriu `generate_dataset.py` |
| **Număr total observații** | 5,000 |
| **Număr features** | 6 (`avg_speed`, `stop_time`, `direction_changes`, etc.) |
| **Tipuri de date** | Numerice (Float/Int) |
| **Format fișiere** | CSV, NPY |

### 3.2 Contribuția Originală (100%)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 5,000 |
| **Observații originale (M)** | 5,000 |
| **Procent contribuție originală** | 100% |
| **Locație cod generare** | `src/data_acquisition/generate_dataset.py` |

**Descriere metodă:** Datele sunt generate folosind un seed fix (42) pentru reproductibilitate. S-au modelat distribuții Gamma pentru timpii de oprire și Poisson pentru schimbările de direcție.

### 3.3 Preprocesare și Split Date
- **Train (70%)**: 3,500 observații
- **Validation (15%)**: 750 observații
- **Test (15%)**: 750 observații

**Preprocesări:** Standardizare (StandardScaler) aplicată după split (fit doar pe train), tratare outliers prin clipping.

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software
1. **Data Acquisition**: Script Python pentru generarea datelor corelate.
2. **Neural Network**: Model MLP implementat în Keras (Binary Crossentropy).
3. **Web Service / UI**: Interfață Streamlit pentru inferență în timp real.

### 4.2 State Machine
**Flux:** `IDLE` (Așteptare) → `INPUT_DATA` (Input utilizator) → `PREPROCESS` (Scalare) → `RN_INFERENCE` (Predicție) → `DISPLAY_RESULT` (Afișare) → `IDLE`.



**Justificare:** Această structură secvențială asigură că modelul primește întotdeauna date scalate corect înainte de a genera o predicție, prevenind erorile de procesare.

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale


- **Input**: 6 neuroni.
- **Hidden 1**: Dense + ReLU + Dropout (0.35).
- **Hidden 2**: Dense + ReLU.
- **Output**: 1 neuron (Sigmoid).

### 5.2 Hiperparametri Finali (Etapa 6)

| Hiperparametru | Valoare Finală | Justificare |
|----------------|----------------|-------------|
| Optimizer | Adam | Convergență stabilă |
| Batch Size | 32 | Echilibru memorie/performanță |
| Dropout | 0.35 | Prevenirea overfitting-ului |
| Class Weights | Aplicat | Corectarea dezechilibrului de clase (69% vs 31%) |

### 5.3 Experimente de Optimizare

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Observații |
|------|----------------------------|----------|----------|------------|
| Baseline | Configurația Etapa 5 | 74.40% | 0.65 | Referință |
| Exp 1 | LR 0.001 -> 0.0003 | 74.10% | 0.65 | Antrenare prea lentă |
| **Exp 3** | **Class Weights** | **72.90%** | **0.70** | **Echilibru optim între clase (FINAL)** |
| Exp 4 | Threshold 0.5 -> 0.45 | 74.10% | 0.63 | Scade capacitatea de discriminare |

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | 72.90% | ≥70% | ✓ |
| **F1-Score (Macro)** | 0.70 | ≥0.65 | ✓ |

### 6.2 Confusion Matrix
- **Performanță:** Recunoaștere foarte bună a clasei majoritare.
- **Îmbunătățire:** Utilizarea `class_weight` a redus considerabil numărul de cazuri "anormale" ratate de model (False Negatives).

### 6.3 Analiza Top Erori
1. **Input cu zgomot**: Cazuri unde eticheta a fost alterată de `LABEL_NOISE`.
2. **Frontieră**: Date cu scor probabilistic apropiat de 0.5.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6
- Încărcarea automată a `optimized_model.h5`.
- Integrarea scaler-ului salvat (`scaler.joblib`) pentru a evita "data leakage".
- Afișarea probabilității predicției (%) în UI.

### 7.3 Demonstrație Funcțională
1. **Pas 1**: Utilizatorul introduce `avg_speed`, `stop_time`, etc.
2. **Pas 2**: UI apelează funcția de preprocesare.
3. **Pas 3**: Modelul returnează "Normal" sau "Anormal".

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță
Toate obiectivele tehnice au fost atinse. Deși acuratețea a scăzut ușor în favoarea F1-Score-ului, modelul final este mult mai robust în condiții reale de dezechilibru al datelor.

### 10.3 Lecții Învățate
- **Importanța F1-Score**: Pe un dataset cu clase 70/30, acuratețea este înșelătoare.
- **Early Stopping**: A prevenit risipa de timp de calcul în Etapa 5.
- **Documentarea**: Salvarea constantă a metricilor în fișiere JSON a ușurat compararea experimentelor în Etapa 6.

### 10.4 Retrospectivă
Dacă aș reîncepe, aș dedica mai mult timp tehnicii de *Feature Engineering* pentru a crea noi parametri din viteza medie și distanță, posibil crescând acuratețea peste 80%.

---

## 11. Bibliografie
1. Keras Documentation, 2024. [https://keras.io/](https://keras.io/)
2. Scikit-learn, StandardScaler Guide, 2024. [https://scikit-learn.org/](https://scikit-learn.org/)
3. François Chollet, "Deep Learning with Python", 2021.