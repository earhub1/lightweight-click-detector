# Visualização diagnóstica

`detection_spectrograms.py` gera imagens de espectrograma por janela para análise qualitativa do detector. Ele recalcula o Z usando exatamente a mesma lógica de `event_detection.py` — incluindo `z_mode`, `contrast_win_ms` e `contrast_alpha` — e exibe os limiares reais usados na detecção como linhas horizontais nos subplots do Z.

> **Nota:** desde a consolidação das saídas do detector, o CSV frame-level (`<arquivo>.csv`) e o window manifest (`<arquivo>_window_manifest.csv`) não são mais gerados automaticamente. Para usar `detection_spectrograms.py`, é necessário re-executar `event_detection.py` sobre o arquivo de interesse com saída temporária individual, ou adaptar o script para leitura direta do `all_events.csv` filtrado por `audio_filepath`.

## Uso

### Modo fixo com thresholds explícitos

```bash
python detection_spectrograms.py \
  --csv ./data/detections/<arquivo>.csv \
  --wav ./data/wav_files/<arquivo>.wav \
  --output ./data/detections/spectrograms \
  --band-low-hz 20000 \
  --band-high-hz 50000 \
  --short-term-duration 0.001 \
  --w-hfc 0.45 --w-sf 0.30 --w-wpd 0.20 --w-cd 0.05 \
  --norm-percentile 95 \
  --z-mode contrast \
  --contrast-win-ms 100 \
  --thr-low 1.5 \
  --thr-high 3.0
```

### Modo adaptativo com thresholds por janela (window manifest)

```bash
# Modo adaptativo com thresholds por janela
# Requer CSV frame-level e window manifest gerados individualmente.
# Esses arquivos não são gerados automaticamente pelo pipeline consolidado —
# execute event_detection.py separadamente sobre o arquivo de interesse para obtê-los.
python detection_spectrograms.py \
  --csv ./data/detections/<arquivo>.csv \
  --wav ./data/wav_files/<arquivo>.wav \
  --output ./data/detections/spectrograms \
  --band-low-hz 20000 \
  --band-high-hz 50000 \
  --short-term-duration 0.001 \
  --w-hfc 0.45 --w-sf 0.30 --w-wpd 0.20 --w-cd 0.05 \
  --norm-percentile 95 \
  --z-mode contrast \
  --window-manifest ./data/detections/<arquivo>_window_manifest.csv
```

No modo com `--window-manifest`, os thresholds exibidos em cada janela são os valores reais que o detector aplicou naquela janela (`low_thr_used` / `high_thr_used` do manifest), refletindo fielmente o comportamento adaptativo.

## Parâmetros principais

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--csv` | — | CSV frame-level do arquivo de áudio (não gerado pelo pipeline consolidado; requer execução individual) |
| `--wav` | — | Arquivo `.wav` correspondente |
| `--output` | — | Diretório de saída para imagens |
| `--window-sec` | `10.0` | Duração da janela de visualização (s) |
| `--hop-sec` | `10.0` | Passo entre janelas (s) |
| `--only-with-events` | `True` | Exporta apenas janelas com eventos |
| `--include-empty-windows` | — | Exporta todas as janelas |
| `--dpi` | `200` | Resolução das imagens |
| `--format` | `png` | Formato das imagens |
| `--band-low-hz` | `20000` | Limite inferior da banda focada (Hz) |
| `--band-high-hz` | `50000` | Limite superior da banda focada (Hz) |
| `--short-term-duration` | `0.001` | Deve ser igual ao usado no detector |
| `--z-mode` | `raw` | Modo do score Z: `raw`, `contrast` ou `hybrid` — deve ser igual ao detector |
| `--contrast-win-ms` | `100.0` | Janela de contraste local (ms) — deve ser igual ao detector |
| `--contrast-alpha` | `0.5` | Peso do Z raw no modo `hybrid` — deve ser igual ao detector |
| `--z-smooth-mode` | `none` | Suavização visual do Z: `none`, `mean`, `median` |
| `--z-smooth-size` | `5` | Tamanho da janela de suavização (frames) |
| `--thr-low` | `None` | Threshold low fixo para exibir como linha nos plots |
| `--thr-high` | `None` | Threshold high fixo para exibir como linha nos plots |
| `--window-manifest` | `None` | Caminho para window manifest individual — lê thresholds reais por janela (não gerado pelo pipeline consolidado) |

### Regra de precedência dos thresholds plotados

1. `--window-manifest` fornecido → usa `low_thr_used`/`high_thr_used` do CSV por janela (**threshold adaptativo real**)
2. `--thr-low`/`--thr-high` fornecidos → usa esses valores fixos em todas as janelas
3. Nenhum dos dois → não desenha linhas de threshold

Os pesos das ODFs e `--norm-percentile` devem ser idênticos aos usados na detecção.

## Subplots gerados

Cada imagem contém 9 subplots com eixo X compartilhado (tempo local da janela):

| # | Subplot | Cor | Descrição |
|---|---------|-----|-----------|
| 1 | **Espectrograma focado** | magma | Banda `[band-low-hz, band-high-hz]` em dB |
| 2 | **dHFC** | azul | Variação positiva de alta frequência |
| 3 | **SF** | laranja | Spectral flux |
| 4 | **WPD** | verde | Weighted phase deviation |
| 5 | **CD** | vermelho | Complex domain |
| 6 | **Z raw** | cinza | Score Z da fusão aditiva (sempre Z raw) |
| 7 | **Z `<z_mode>`** | roxo | Score Z efetivo do detector conforme `--z-mode` |
| 8 | **Z smooth** | marrom | Z efetivo suavizado (apenas visual) |
| 9 | **Eventos** | —  | Linhas verticais: vermelho = `high`, verde = `low` |

Linhas de threshold são desenhadas nos subplots 7 e 8:
- **Dourada (---)** → `low_thr`
- **Carmesim (---)** → `high_thr`

Linhas verticais de eventos são sobrepostas ao espectrograma (subplot 1), ao Z efetivo (subplot 7) e ao Z smooth (subplot 8).

O subplot 6 (Z raw) serve como referência sempre em escala absoluta, independente do `z_mode`, facilitando comparar o Z bruto com o Z transformado pelo contraste local.

## Saídas

As imagens são salvas em `<output>/<stem_do_wav>/` com nome:

```
<stem>__wstart_<ms>ms__wend_<ms>ms.<format>
```

Um arquivo `manifest.csv` é gerado na mesma pasta com colunas:

| Coluna | Descrição |
|--------|-----------|
| `image_path` | Caminho absoluto da imagem |
| `wav_path` | Caminho absoluto do WAV |
| `window_start_sec` | Início da janela (s) |
| `window_end_sec` | Fim da janela (s) |
| `num_high` | Número de eventos `high` na janela |
| `num_low` | Número de eventos `low` na janela |
| `high_times_sec` | JSON com instantes dos eventos `high` |
| `low_times_sec` | JSON com instantes dos eventos `low` |
| `rms` | RMS do áudio |
| `peak` | Pico do áudio |
| `duration_sec_real` | Duração real do áudio lido (s) |
| `spec_db_p95` | Percentil 95 do espectrograma em dB |
| `spec_db_median` | Mediana do espectrograma em dB |
| `snr_db_approx` | SNR aproximado (p95 - p5) em dB |
| `z_raw_max` | Valor máximo do Z raw na janela |
| `z_raw_p95` | Percentil 95 do Z raw na janela |
