# Automatinis šviesų perjungimas pagal priešais atvažiuojančius automobilius

Grupinis individualus darbas, P176B041 „Dirbtinis intelektas“, Autotransporto elektronika (AE 5/1 gr.), Kaunas 2026.
Dėstytojas: Laimonas Kairiūkštis.

Pagal priekinės kameros kadrą sistema nusprendžia, ar kelias laisvas (**0 – tolimosios šviesos**), ar priekyje
yra priešais atvažiuojantis automobilis (**1 – artimosios šviesos**). Prototipas veikia tik **patariamuoju režimu**
(ekranas / LED), realių automobilio šviesų savarankiškai nejungia, o vairuotojas visada gali perimti valdymą.
Viskas vyksta lokaliai, be interneto ir be vaizdų įrašymo.

## Projekto struktūra

| Failas / aplankas | Paskirtis |
|---|---|
| `README.md` | Projekto aprašymas, paleidimo instrukcija ir darbų pasidalijimo lentelė |
| `requirements.txt` | Visų būtinų Python bibliotekų sąrašas su versijomis |
| `code/generate_dataset.py` | 1600 sintetinių naktinių kadrų generatorius (70 / 15 / 15 padalijimas) |
| `code/train_and_eval.py` | Bandymo 1, bazinio (LogisticRegression) ir CompactHeadlightCNN mokymas, testavimas ir greičio matavimas |
| `app_demo.py` | Paprasta Streamlit demonstracinė sąsaja (kadras → sprendimas, tikimybė, LED indikacija) |

Programos sukuria aplankus `data/` (duomenų rinkinys), `models/` (išsaugoti modeliai) ir `outputs/`
(`results.json`, mokymo kreivės, konfuzijos matricos, sėkmės / klaidų pavyzdžiai, dėmesio žemėlapiai).

## Paleidimas

```bash
pip install -r requirements.txt
python code/train_and_eval.py      # sugeneruoja duomenis, apmoko modelius, išveda metrikas ir grafikus
streamlit run app_demo.py          # paleidžia grafinę demonstraciją naršyklėje
```

Papildomai:

```bash
python code/generate_dataset.py          # tik duomenų rinkinys (data/dataset.npz ir data/preview.png)
python code/train_and_eval.py --skip-cnn # be PyTorch: tik ryškumo slenkstis ir bazinis modelis
```

Visi atsitiktiniai procesai naudoja `seed = 42`, todėl rezultatai atkartojami.

## Modeliai

1. **Bandymas 1 (atmestas):** viso kadro vidutinio ryškumo slenkstis.
2. **Bazinis:** 8 požymiai iš kelio srities (ROI) → `StandardScaler` → `LogisticRegression (C = 1,0, random_state = 42)`.
3. **Patobulintas – CompactHeadlightCNN:** 3 blokai (`Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`),
   `AdaptiveAvgPool2d`, `Dropout (0,25)` ir du pilnai sujungti sluoksniai, iš viso 43 826 parametrai.
   Adam (lr = 0,0012, weight_decay = 1e-4), StepLR (step = 8, gamma = 0,5), batch_size = 32, 22 epochos.

## Darbų pasidalijimo lentelė

| Komandos narys | Atsakomybės sritis | Skyriai |
|---|---|---|
| Matas Kostik | Sintetinių duomenų generavimas, duomenų paruošimas, bazinis modelis | 2, 5 |
| Simonas Mauricas, Ernestas Gaivenis | CompactHeadlightCNN kūrimas, treniravimas, XAI žemėlapiai, grafikai | 3, 4, 5, Paveikslai |
| Tautvydas Gečionis, Matas Jancevičius | Problemos analizė, pokalbiai su vairuotojais, etika / sauga, ataskaita ir priedai | Santrauka, 1, 6, Literatūra, Priedai |
