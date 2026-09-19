# Sistemi i Klasifikimit të Imazheve Ekografike të Gjirit

### Prototip akademik kërkimor — përmbledhje e projektit

> **Vërejtje e rëndësishme:** Ky është një prototip akademik kërkimor. Nuk është
> pajisje mjekësore, nuk është validuar klinikisht dhe nuk duhet përdorur për
> diagnozë ose vendime trajtimi. Nuk pretendohet asnjë përputhshmëri rregullatore
> (HIPAA, GDPR, FDA, CE).

---

## Përmbajtja

1. [Çfarë është projekti](#1-çfarë-është-projekti)
2. [Gjendja fillestare dhe problemet e gjetura](#2-gjendja-fillestare-dhe-problemet-e-gjetura)
3. [Çfarë u implementua](#3-çfarë-u-implementua)
4. [Struktura e projektit](#4-struktura-e-projektit)
5. [Si funksionon sistemi](#5-si-funksionon-sistemi)
6. [Si ta ekzekutoni](#6-si-ta-ekzekutoni)
7. [Rezultatet reale të matura](#7-rezultatet-reale-të-matura)
8. [Vlera akademike për tezën](#8-vlera-akademike-për-tezën)
9. [Çfarë mbetet për t'u bërë](#9-çfarë-mbetet-për-tu-bërë)
10. [Kufizimet](#10-kufizimet)

---

## 1. Çfarë është projekti

Projekti është një sistem i bazuar në inteligjencë artificiale për klasifikimin e
imazheve ekografike të gjirit në dy kategori: **beninje** dhe **malinje**.

Sistemi përbëhet nga dy pjesë:

**a) Aplikacioni web (Flask)** — përdoruesi ngarkon një imazh ekografik dhe merr:
- klasifikimin e modelit,
- probabilitetin e parashikuar,
- nivelin e besueshmërisë,
- një shpjegim vizual (Grad-CAM),
- një raport të strukturuar me informacion mbi modelin dhe kufizimet e tij.

**b) Infrastruktura kërkimore (skriptet)** — një pipeline i riprodhueshëm që
prodhon çdo figurë, tabelë dhe metrikë që do të përdoret në tezë, nga të dhënat
e vërteta dhe jo nga vlera të shpikura.

### Ndryshimi kryesor krahasuar me versionin fillestar

Versioni fillestar bënte:

```
Ngarko imazh  →  Parashikim  →  Tekst: "Yes Breast Cancer"
```

Versioni aktual bën:

```
Ngarko imazh
      ↓
Validim (siguri)
      ↓
Parapërpunim (i njëjtë si në trajnim)
      ↓
Model CNN
      ↓
Klasifikim + probabilitet + besueshmëri
      ↓
Shpjegim vizual (Grad-CAM)
      ↓
Raport i strukturuar + kufizime + deklaratë përgjegjësie
```

---

## 2. Gjendja fillestare dhe problemet e gjetura

Para se të implementohej çfarëdo gjëje, u krye një auditim teknik i kodit dhe i
të dhënave ekzistuese. U gjetën tre probleme thelbësore.

### Problemi 1 — Rrjedhje e të dhënave (*data leakage*)

Dataset-i origjinal me 5.000 imazhe nuk ishte një koleksion origjinal, por një
version i **augmentuar paraprakisht**. Matjet reale mbi 2.500 imazhet beninje:

| Matja | Vlera |
|---|---|
| Grupe dublikatash identike (byte për byte) | 51 grupe (103 skedarë) |
| Imazhe me një "binjak" vizual (korrelacion > 0.95) | **2.053 / 2.500 (82,1%)** |
| Burime vizualisht të dallueshme | **~891, jo 2.500** |
| **Imazhe testi me kuazi-dublikatë në trajnim** | **77,3%** |

Meqenëse augmentimi ishte bërë **para** ndarjes në trajnim/test, rreth tre të
katërtat e setit të testimit ishin parë tashmë gjatë trajnimit.

> **Prandaj saktësia e raportuar prej 99,6% nuk është një vlerësim i vërtetë i
> aftësisë gjeneralizuese të modelit.** Ajo pasqyron kryesisht memorizim.

### Problemi 2 — Mospërputhje midis trajnimit dhe shërbimit

Kodi i trajnimit dhe kodi i aplikacionit e përpunonin imazhin ndryshe:

| Hapi | `MainTrain.py` (trajnimi) | `app.py` (aplikacioni) |
|---|---|---|
| Konvertim BGR → RGB | po | **jo** |
| Normalizim | po | **jo** |

Matja reale mbi 150 imazhe me modelin ekzistues:

> **9 nga 150 parashikime (6,0%) ndryshonin** vetëm për shkak të kësaj
> mospërputhjeje. Diferenca mesatare e probabilitetit: 0,068.

Kjo do të thotë se çdo metrikë e matur jashtë linje nuk përshkruante atë që
aplikacioni bënte realisht.

### Problemi 3 — Dosja `dataset/malignant/` mungon

Gjysma malinje e dataset-it nuk gjendet as në projekt, as në historikun e Git,
as në arkivin rezervë `Diploma.rar` (192 MB). Prandaj `MainTrain.py` nuk mund të
ekzekutohet fare.

**Zgjidhja e miratuar:** kalimi te dataset-i origjinal **BUSI**.

---

## 3. Çfarë u implementua

### 3.1 Rregullimet themelore (parakusht për gjithçka tjetër)

| # | Rregullimi | Përshkrimi |
|---|---|---|
| **T1** | **Parapërpunim i unifikuar** | `src/preprocessing.py` është tani burimi i vetëm i së vërtetës, i përdorur njëlloj nga trajnimi, vlerësimi, Grad-CAM dhe aplikacioni. Profili i modelit ruhet në metadata gjatë trajnimit dhe riprodhohet gjatë parashikimit, kështu që mospërputhja është teknikisht e pamundur. |
| **T2** | **Ndarje e vetëdijshme për grupet** | `src/data/splitting.py` ndan **grupe** kuazi-dublikatash, jo imazhe individuale, duke ruajtur balancën e klasave. Ekziston një kontroll i detyrueshëm që asnjë grup të mos shfaqet në dy ndarje njëkohësisht. |
| **T3** | **Protokoll i korrigjuar eksperimental** | Set validimi i ndarë (seti i testimit nuk përdoret më si `validation_data`), përzierje e aktivizuar, ndalim i hershëm (*early stopping*), `ReduceLROnPlateau`, pesha të balancuara klasash, fara (*seed*) fikse dhe ruajtje e plotë e metadatave. |

#### Krahasimi i protokollit

| Aspekti | Më parë | Tani |
|---|---|---|
| Të dhënat e validimit | seti i testimit | ndarje e veçantë |
| Stratifikimi | asnjë | i stratifikuar sipas klasës |
| Grupimi | asnjë | sipas kuazi-dublikatave |
| Përzierja | `shuffle=False` mbi të dhëna të renditura sipas klasës | e përzier në çdo epokë |
| Epokat | 10, fikse | ndalim i hershëm mbi `val_loss` |
| Peshat e klasave | asnjë | të balancuara |
| Fara (*seed*) | asnjë | Python + NumPy + TensorFlow |

### 3.2 Nëntë veçoritë e miratuara

| # | Veçoria | Ku ndodhet | Çfarë bën |
|---|---|---|---|
| **1** | **Grad-CAM** | `src/explainability/gradcam.py` | Prodhon një hartë nxehtësie që tregon se cilat zona të imazhit ndikuan më shumë në vendimin e modelit |
| **2** | **Krahasim modelesh** | `src/models/registry.py`, `scripts/compare_models.py` | CNN i personalizuar, ResNet50 dhe EfficientNet-B0 nën kushte identike |
| **3** | **Panel vlerësimi** | faqja `/research` | Metrikat e testit, krahasimi i modeleve, analiza e gabimeve |
| **4** | **Matrica e konfuzionit + ROC-AUC** | `src/evaluation/plots.py` | Plus kurbat precision-recall, kalibrimi dhe analiza e pragut |
| **5** | **Vizualizim i besueshmërisë** | `src/inference/predictor.py` | Shirita probabiliteti për çdo klasë dhe brez besueshmërie (i lartë / i moderuar / i ulët) |
| **6** | **Analizë e gabimeve** | `src/evaluation/errors.py` | Identifikon TP/TN/FP/FN dhe **karakterizon sasiorisht** se cilat imazhe dështojnë |
| **7** | **Analizë e dataset-it** | `src/data/analysis.py`, faqja `/dataset` | Shpërndarja e klasave, auditimi i integritetit, kontrolli i shkurtoreve |
| **8** | **Historik parashikimesh** | `src/inference/history.py`, faqja `/history` | Regjistër lokal i parashikimeve (vetëm metadata) |
| **9** | **Raport i strukturuar** | `src/reporting/report.py` | Klasifikim, probabilitet, shpjegim, informacion modeli, kufizime, deklaratë përgjegjësie |

### 3.3 Detaje për veçoritë kryesore

#### Grad-CAM — pse pikërisht ky metod?

Grad-CAM u zgjodh mbi SHAP dhe LIME sepse i përgjigjet pikërisht pyetjes që ka
rëndësi klinike:

> **A e shikoi modeli lezionin, apo shikoi vizoren e thellësisë, tekstin e
> ekografit, ose sfondin e zi?**

- **LIME** ndan imazhin në superpiksela, gjë që është e paqëndrueshme mbi
  teksturën me zhurmë *speckle* të ekografisë.
- **SHAP** është shumë i ngadaltë (KernelSHAP) ose prodhon atribuime me zhurmë
  në imazhe gri me kontrast të ulët (DeepSHAP).

**Kufizimi i rezolucionit** raportohet gjithmonë hapur:

| Modeli | Hyrja | Harta e veçorive | Qelizat |
|---|---|---|---|
| Modeli ekzistues (`.h5`) | 64×64 | 6×6 | **36 — tepër e trashë për kuptim klinik** |
| `custom_cnn` | 224×224 | 52×52 | 2.704 |
| ResNet50 / EfficientNet-B0 | 224×224 | 7×7 | 49 |

> Harta e nxehtësisë është **vizualizim i vëmendjes së modelit**, jo segmentim i
> lezionit dhe jo kufi anatomik i validuar mjekësisht. Kjo thuhet qartë si në API
> ashtu edhe në ndërfaqe.

#### Krahasimi i modeleve — çfarë pyetjeje kërkimore i përgjigjet secili

| Modeli | Parametrat | Pyetja kërkimore |
|---|---|---|
| `custom_cnn` | 33.442 | Bazë krahasimi — arkitektura e vetë tezës, e korrigjuar |
| `resnet50` | 23.591.810 | A ndihmon *transfer learning* nga ImageNet mbi imazhe gri ekografike? |
| `efficientnetb0` | 4.052.133 | A e mund shkallëzimi i përbërë thellësinë reziduale me ~650 imazhe? |
| `custom_cnn_legacy64` | **176.290** | Ablacion rezolucioni — riprodhon **saktësisht** modelin origjinal |

**Pse krahasimi është i drejtë:** çdo model trashëgon `configs/base.json`, pra
optimizuesi, shkalla e mësimit, madhësia e grupit, *callbacks*, peshat e klasave,
politika e augmentimit, fara dhe ndarja janë **identike**. Ndryshon vetëm
arkitektura. Skripti `compare_models.py` e rikontrollon këtë nga metadatat dhe
jep paralajmërim nëse ndonjë parametër ka devijuar.

#### Metrikat — pse jo vetëm saktësia

Për detektimin e kancerit, saktësia është metrika më pak informuese:

- **Ndjeshmëria (*sensitivity*)** është metrika kryesore — një rezultat
  fals-negativ do të thotë kancer i humbur.
- **Specificiteti** është kundërpesha — një fals-pozitiv do të thotë ndjekje e
  panevojshme, e dëmshme por e riparueshme.
- **PR-AUC** është më e ndershme se ROC-AUC kur klasa pozitive është pakicë.
- **Intervalet e besimit 95%** (bootstrap) për çdo metrikë — një vlerësim i
  vetëm nga ~100 imazhe testi nuk është rezultat shkencor.
- **ECE (gabimi i pritshëm i kalibrimit)** — nëse modeli thotë "90%", a ka të
  drejtë në 90% të rasteve?

#### Analiza e gabimeve — pse është kontribut kërkimor

Një galeri imazhesh të gabuara është thjesht një faqe ndërfaqeje. Ky modul
prodhon një **pohim të verifikueshëm**:

> "A janë rastet fals-negative sistematikisht më të vogla, më të errëta ose me
> kontrast më të ulët se rastet e klasifikuara saktë?"

Llogaritet madhësia e efektit (Cohen's *d*) për çdo veti imazhi midis
parashikimeve të sakta dhe atyre të gabuara. Gjithashtu:
- **ndarja e besueshmërisë** — nëse besueshmëria nuk i dallon rastet e sakta nga
  ato të gabuara, atëherë nuk mund të përdoret për triazh (dhe kjo është vetë një
  gjetje e vlefshme);
- **kurba e abstenimit** — çfarë saktësie arrihet nëse sistemi vendos vetëm kur
  është i sigurt dhe i referon rastet e tjera te specialisti.

> Nuk raportohen vlera *p*. Me ~100 imazhe testi dhe disa veti të shqyrtuara,
> madhësitë e efektit janë përmbledhja e ndershme dhe shmangin problemin e
> krahasimeve të shumëfishta.

### 3.4 Siguria dhe privatësia

**Të implementuara:**
- Verifikim i formatit me *magic bytes* (prapashtesa e skedarit nuk besohet)
- Kufi i madhësisë së ngarkimit (10 MB)
- Kontroll dekodimi dhe përmasash (refuzon skedarë të korruptuar dhe *decompression bombs*)
- Imazhet **nuk shkruhen në disk** nga rruga e API-t
- `debug=False` si parazgjedhje
- Gabimet kthejnë mesazhe të sigurta; detajet shkojnë vetëm në regjistrin e serverit
- Shërbimi i figurave është i kufizuar brenda `results/figures` (bllokohet *path traversal*)
- Historiku ruan **vetëm metadata**; ruajtja e imazheve është opsionale dhe vetëm si miniaturë

**Të paimplementuara — nuk duhet pretenduar e kundërta:** autentikim, autorizim,
HTTPS, regjistër auditimi, politikë ruajtjeje të dhënash, enkriptim në qetësi,
ose çfarëdo përputhshmërie rregullatore.

---

## 4. Struktura e projektit

```
app.py                      Vetëm shtresa e rrugëve Flask (pa logjikë ML)
configs/                    Konfigurime JSON; modelet trashëgojnë base.json
  base.json                 cilësimet e përbashkëta
  custom_cnn.json           arkitektura e tezës në 224×224
  custom_cnn_legacy64.json  riprodhim besnik i modelit origjinal 64×64
  resnet50.json             ResNet50 i paratrajnuar në ImageNet
  efficientnetb0.json       EfficientNet-B0 i paratrajnuar në ImageNet
src/
  config.py                 ngarkimi i konfigurimit + farat globale
  preprocessing.py          BURIMI I VETËM I SË VËRTETËS për parapërpunimin
  augmentation.py           augmentim vetëm për setin e trajnimit
  data/
    loader.py               zbulimi i dataset-it
    integrity.py            zbulimi i dublikatave dhe matja e rrjedhjes
    splitting.py            ndarje e stratifikuar e vetëdijshme për grupet
    analysis.py             statistika dhe figura të dataset-it
  models/registry.py        arkitekturat + shtresat e synuara për Grad-CAM
  training/trainer.py       orkestrimi i trajnimit
  evaluation/
    metrics.py              metrikat + intervalet e besimit + kalibrimi
    plots.py                gjenerimi i figurave
    errors.py               analiza e gabimeve
  explainability/gradcam.py Grad-CAM
  inference/
    validation.py           validimi i ngarkimeve
    predictor.py            regjistri i modeleve + parashikimi
    history.py              historiku (SQLite)
  reporting/
    report.py               raporti i strukturuar
    results_index.py        ngarkon rezultatet e ruajtura për panelin
scripts/                    prepare_data, analyze_dataset, train, evaluate, compare_models
tests/                      97 teste
models/                     modelet e trajnuara + metadata
results/                    metrikat, figurat, ndarja
notebooks/train_colab.ipynb trajnim në Google Colab (GPU falas)
docs/THESIS_MAPPING.md      harta: implementim → kapituj të tezës
```

### Faqet e aplikacionit

| Faqja | Çfarë tregon |
|---|---|
| `/service` | Ngarkim → parashikim → besueshmëri → Grad-CAM → raport |
| `/dataset` | Shpërndarja e klasave, auditimi i integritetit, ndarja, kontrolli i shkurtoreve |
| `/research` | Metrikat e testit me IB 95%, krahasimi i modeleve, analiza e gabimeve |
| `/history` | Regjistri lokal i parashikimeve |

> **Parim i rëndësishëm:** nuk shfaqet asgjë që nuk është prodhuar nga një
> ekzekutim real. Seksionet pa rezultate shfaqin *"Eksperimenti nuk është
> ekzekutuar ende"* së bashku me komandën që do t'i prodhonte.

---

## 5. Si funksionon sistemi

### Rrjedha e parashikimit

```
Shfletuesi (service.html)
      │  POST /api/predict
      ↓
Validimi           → magic bytes, kufi madhësie, kontroll dekodimi
      ↓
Parapërpunimi      → i njëjti kod si në trajnim (profili nga metadatat)
      ↓
Modeli CNN         → probabilitete softmax
      ↓
Grad-CAM           → hartë nxehtësie mbi imazhin origjinal
      ↓
Raporti            → JSON i strukturuar
      ↓
Ndërfaqja          → klasifikim, shirita probabiliteti, hartë, model, kufizime
```

### Rrjedha kërkimore (jashtë linje)

```
dataset BUSI
      ↓
[1] Auditim integriteti     → dublikata, kuazi-dublikata, grupe
      ↓
[2] Ndarje sipas grupeve    → trajnim / validim / test (asnjë lezion në dy ndarje)
      ↓
[3] Parapërpunim            ← I NJËJTI MODUL si aplikacioni
      ↓
[4] Augmentim               → VETËM seti i trajnimit
      ↓
[5] Trajnim                 → i drejtuar nga konfigurimi, me fara fikse
      ↓
[6] Vlerësim                → metrika + IB + kalibrim + prag
      ↓
[7] Artefakte               → models/*.keras + results/*.json + figura
```

---

## 6. Si ta ekzekutoni

### Instalimi

Kërkohet **Python 3.10** (TensorFlow 2.15 nuk mbështet 3.12+).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/ -q          # duhet të kalojnë 97 teste
```

### Përgatitja e dataset-it

Dataset-i i përdorur është **BUSI**:

> Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast ultrasound
> images.* Data in Brief. 2020 Feb;28:104863. doi:10.1016/j.dib.2019.104863

780 imazhe — 437 beninje, 210 malinje, 133 normale, me maska lezionesh.

```bash
python scripts/prepare_data.py --source "C:/Downloads/Dataset_BUSI_with_GT"
```

### Riprodhimi i rezultateve

```bash
# 1. Analiza e dataset-it dhe ndarja e përbashkët
python scripts/analyze_dataset.py
python scripts/analyze_dataset.py --legacy-dataset   # audito edhe dataset-in e vjetër

# 2. Trajnimi (të gjitha modelet lexojnë TË NJËJTËN ndarje)
python scripts/train.py --config custom_cnn
python scripts/train.py --config resnet50
python scripts/train.py --config efficientnetb0

# 3. Vlerësimi mbi setin e testimit
python scripts/evaluate.py --model custom_cnn
python scripts/evaluate.py --model resnet50
python scripts/evaluate.py --model efficientnetb0

# 4. Tabela e krahasimit
python scripts/compare_models.py
```

### Ablacionet

```bash
# A ndihmon augmentimi?
python scripts/train.py --config custom_cnn --no-augmentation
python scripts/evaluate.py --model custom_cnn_noaug

# A ka rëndësi rezolucioni? (64×64 — cilësimi origjinal i tezës)
python scripts/train.py --config custom_cnn_legacy64
python scripts/evaluate.py --model custom_cnn_legacy64
```

### Trajnimi në Google Colab

Trajnimi i ResNet50 / EfficientNet-B0 në 224×224 është i ngadaltë në CPU (kjo
makinë nuk ka GPU). Përdorni `notebooks/train_colab.ipynb`, i cili ekzekuton
**të njëjtat skripte** mbi një GPU falas, pastaj shkarkoni dosjet `models/` dhe
`results/` në projektin lokal.

### Ekzekutimi i aplikacionit

```bash
python app.py
# http://127.0.0.1:5000
```

---

## 7. Rezultatet reale të matura

> **Nuk ekziston ende asnjë model i trajnuar mbi të dhëna reale**, sepse BUSI nuk
> është shkarkuar ende. Prandaj nuk raportohet asnjë metrikë performance.

Këto janë matjet reale të kryera gjatë punës:

| Matja | Vlera |
|---|---|
| Mospërputhja trajnim/shërbim (modeli ekzistues, 150 imazhe) | **9/150 parashikime ndryshuan (6,0%)**, diferenca mesatare 0,068 |
| Kuazi-dublikata në dataset-in e vjetër (2.500 imazhe, korr. > 0,95) | **82,1%**; ~891 grupe të dallueshme |
| Rrjedhja në setin e testimit me ndarjen origjinale | **77,3%** |
| Ndarja sipas grupeve kundrejt ndarjes naive (nënset 800 imazhe) | **0,0% kundrejt 66,7%** |
| Parametrat e `custom_cnn_legacy64` | 176.290 — përputhje e saktë me `BreastCancer10Epochs.h5` |
| Metrikat kundrejt llogaritjeve me dorë të Kap. 12 | përputhje e saktë në të 7 vlerat |
| Testet automatike | **97 kaluan, 0 dështuan** |

Një ekzekutim provë i të gjithë pipeline-it (analizë → trajnim → vlerësim →
krahasim) u krye mbi një dataset **sintetik** vetëm për të verifikuar që skriptet
funksionojnë nga fillimi në fund. **Të gjitha ato artefakte u fshinë** — dosjet
`results/` dhe `models/` janë bosh.

---

## 8. Vlera akademike për tezën

Skedari [`docs/THESIS_MAPPING.md`](docs/THESIS_MAPPING.md) përmban hartën e plotë:
çdo komponent → seksioni i tezës → eksperimenti → rezultati → figura → tabela →
diskutimi.

### Kapitujt që forcohen

| Kapitulli | Çfarë fiton |
|---|---|
| **Kap. 7** (Dataset-i) | Një kontribut origjinal: auditimi i integritetit dhe matja e rrjedhjes |
| **Kap. 8** (Parapërpunimi) | Ablacione të matura në vend të pohimeve të pambështetura |
| **Kap. 10** (Zhvillimi i modelit) | Krahasim real arkitekturash — e bën të vërtetë pretendimin e abstraktit për ResNet/VGG |
| **Kap. 12** (Vlerësimi) | Intervale besimi, kalibrim, analizë pragu, PR-AUC |
| **Kap. 13.1** (Kufizimet) | Pohimi "Grad-CAM nuk u integrua" tani është i vjetëruar |

### Kapituj të rinj të mundshëm

1. **Integriteti i dataset-it dhe analiza e rrjedhjes** — kontributi më i fortë
2. **Vlerësim krahasues i arkitekturave**
3. **Shpjegueshmëria**
4. **Analiza e gabimeve**
5. **Kalibrimi dhe zgjedhja e pikës së punës**
6. **Riprodhueshmëria** (shtojcë)

### Eksperimentet e mundshme tani

| ID | Eksperimenti | Komanda |
|---|---|---|
| E1 | **Ablacioni i rrjedhjes** (rezultati kryesor) | `analyze_dataset.py --legacy-dataset` |
| E2 | Krahasimi i arkitekturave | `train.py` × 3 → `compare_models.py` |
| E3 | Ablacioni i rezolucionit (64 kundrejt 224) | `train.py --config custom_cnn_legacy64` |
| E4 | Ablacioni i augmentimit | `train.py --config custom_cnn --no-augmentation` |
| E6 | Analiza e pragut / pika e punës | automatike në `evaluate.py` |
| E8 | Kalibrimi | automatike në `evaluate.py` |
| E9 | Karakterizimi i gabimeve + kurba e abstenimit | automatike në `evaluate.py` |

### Një këshillë për mbrojtjen

Prisni që saktësia e ndershme të jetë rreth **80–90%**, jo 99,6%. Kjo është vlera
**e saktë**.

> Një 85% i ndershëm është një rezultat i mbrojtshëm. Një 99,6% me rrjedhje nuk
> është.

Zbulimi dhe korrigjimi i gabimit në të dhënat e veta është pikërisht lloji i
gjykimit akademik që një tezë synon të certifikojë. Kjo duhet paraqitur si pikë e
fortë, jo si turp.

---

## 9. Çfarë mbetet për t'u bërë

| # | Detyra | Prioriteti |
|---|---|---|
| 1 | **Shkarkoni BUSI** — bllokon çdo eksperiment | 🔴 Thelbësore |
| 2 | Ekzekutoni trajnimin në Colab | 🔴 Thelbësore |
| 3 | Rishikoni tekstin e tezës (13 pretendime të listuara në `THESIS_MAPPING.md`) | 🔴 Thelbësore |
| 4 | Skript për vleftësim të kryqëzuar (funksioni ekziston, skripti jo) | 🟠 I dobishëm |
| 5 | Kalibrim me *temperature scaling* (ECE matet; korrigjimi nuk implementohet) | 🟠 I dobishëm |
| 6 | Fshini skedarin e mbetur `README.mdgit` | 🟡 Kozmetike |

### Hapi i menjëhershëm i rekomanduar

```bash
python scripts/prepare_data.py --source <shtegu-drejt-BUSI>
python scripts/analyze_dataset.py --legacy-dataset
```

Kjo prodhon **E1 — rezultatin tuaj për rrjedhjen** para çdo trajnimi, sepse
auditon dataset-in e vjetër dhe ndërton ndarjen e pastër në një hap të vetëm.

---

## 10. Kufizimet

- **Klasifikim binar** beninj/malinj. Klasa `normal` e BUSI-t përjashtohet si
  parazgjedhje (ndryshohet nga `data.include_normal` në `configs/base.json`).
- **Një dataset i vetëm publik** — gjeneralizimi te ekografë, cilësime dhe
  popullata të tjera nuk është testuar.
- **Asnjë metadatë klinike** (mosha, pajisja, BI-RADS), prandaj analiza e
  nëngrupeve ose e drejtësisë algoritmike **nuk është e mundur** mbi këto të
  dhëna. Kjo deklarohet si punë e ardhshme, nuk simulohet.
- **Probabilitetet nuk janë të kalibruara** — janë dalje të papërpunuara softmax.
  ECE raportohet që shkalla e mospërputhjes të jetë e dukshme.
- Dosja `pred/` (546 skedarë të përzier) ruhet nga projekti origjinal, nuk është
  pjesë e asnjë eksperimenti dhe nuk ka etiketa.

---

## Tre çështje që kërkojnë vendimin tuaj

1. **Dëshmitë e shpikura të pacientëve** në `home.html` (rreshtat 335–375) janë
   ende aktive, si dhe faqja `/appointment` që nënkupton një shërbim klinik
   joekzistues. Nuk u hoqën sepse janë veçori ekzistuese dhe fshirja kërkon
   miratimin tuaj. **Rekomandohet heqja e të dyjave.**

2. **Funksioni `gradcam.mask_agreement()`** ekziston por nuk është i lidhur me
   pipeline-in. Ai mat përputhjen e hartës së nxehtësisë me maskat reale të
   lezioneve të BUSI-t (*pointing game* dhe IoU). Kjo do ta shndërronte
   shpjegueshmërinë nga një figurë në një **matje** — një rezultat i fortë për
   tezën. Thoni nëse dëshironi ta aktivizoni.

3. **Teksti i përgjigjes së `/predict`** ndryshoi nga `"Yes Breast Cancer"` në
   `"Model classification: benign (predicted probability 100.0%)"`. Kontrata e
   endpoint-it (tekst i thjeshtë, e njëjta URL) ruhet, por formulimi i vjetër
   pohonte një diagnozë që sistemi nuk ka të drejtë ta bëjë.
