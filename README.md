# Detector Acústico de Eventos Impulsivos em Bioacústica Marinha

Pipeline em Python para detecção de eventos impulsivos (cliques de odontocetos) em gravações de monitoramento passivo (PAM). Desenvolvido inicialmente para baleias-piloto (_Globicephala_) e expandido para dados oceânicos reais do projeto PAMA.

## Estrutura do repositório

```
.
├── event_detection.py           # Pipeline principal de detecção
├── detection_spectrograms.py    # Geração de espectrogramas diagnósticos
├── compute_sxx_fbins.py         # Representações espectrais compactas
├── core/
│   ├── signal_processing.py
│   └── onset_detection_functions.py
├── plot_setup.py
├── click_event_visualization.ipynb
├── globicephala_click_EDA.ipynb
└── docs/                        # Documentação detalhada
```

## Uso rápido

**Detecção com threshold fixo:**

```bash
python event_detection.py \
  --input ./data/wav_files \
  --output ./data/detections \
  --thr-mode fixed \
  --low-thr 0.6 \
  --high-thr 1.2 \
  --flow 10000 \
  --fhig 30000
```

**Visualização diagnóstica:**

```bash
python detection_spectrograms.py \
  --csv ./data/detections/<arquivo>.csv \
  --wav ./data/wav_files/<arquivo>.wav \
  --output ./data/detections/spectrograms
```

## Documentação

A documentação detalhada está em [`docs/`](docs/):

- [`docs/pipeline.md`](docs/pipeline.md) — Fluxo completo do detector
- [`docs/parameters.md`](docs/parameters.md) — Referência de parâmetros
- [`docs/outputs.md`](docs/outputs.md) — Formato das saídas CSV
- [`docs/visualization.md`](docs/visualization.md) — Uso do `detection_spectrograms.py`
- [`docs/methodology.md`](docs/methodology.md) — Decisões metodológicas e evolução do detector
