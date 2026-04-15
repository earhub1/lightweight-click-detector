# Formato das saídas

`event_detection.py` gera um único arquivo CSV consolidado no diretório de saída (`--output`), acumulando as detecções de **todos os arquivos `.wav`** processados em lote. Nenhum CSV individual por arquivo é criado.

## Chain-level — `all_chains.csv`

Uma linha por cadeia de cliques detectada. Cada linha identifica o arquivo de origem pela coluna `audio_filepath`.

### Identificação

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `audio_filepath` | str | Caminho absoluto do `.wav` de origem |
| `filepath` | str | Caminho do arquivo processado (herdado internamente, igual a `audio_filepath`) |
| `chain_id` | int | Identificador da cadeia (reinicia por arquivo) |

### Temporal

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `start_time` | float | Instante do primeiro clique da cadeia (s) |
| `end_time` | float | Instante do último clique da cadeia (s) |
| `duration_sec` | float | Duração total da cadeia (s) |
| `n_clicks` | int | Número de cliques na cadeia |
| `click_rate_hz` | float | Taxa de cliques (Hz) — `n_clicks / duration_sec` |

### ICI (Inter-Click Interval)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `mean_ici_ms` | float | ICI médio entre cliques consecutivos (ms) |
| `std_ici_ms` | float | Desvio padrão do ICI (ms) |
| `cv_ici` | float | Coeficiente de variação do ICI — `std / mean` |
| `median_ici_ms` | float | Mediana do ICI (ms) |
| `min_ici_ms` | float | ICI mínimo (ms) |
| `max_ici_ms` | float | ICI máximo (ms) |
| `p25_ici_ms` | float | 1° quartil do ICI (ms) |
| `p75_ici_ms` | float | 3° quartil do ICI (ms) |
| `ici_trend_slope` | float | Coeficiente angular da regressão linear do ICI ao longo da cadeia — positivo indica ICI crescente (cliques espaçando), negativo indica aceleração |

### Intensidade (score Z)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `mean_peak_z` | float | Média do pico Z de cada clique da cadeia |
| `std_peak_z` | float | Desvio padrão do pico Z |
| `cv_peak_z` | float | Coeficiente de variação do pico Z |
| `min_peak_z` | float | Pico Z mínimo entre os cliques |
| `max_peak_z` | float | Pico Z máximo entre os cliques |

### Frequência dominante

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `mean_dominant_freq_hz` | float | Frequência dominante média dos cliques da cadeia (Hz) |
| `std_dominant_freq_hz` | float | Desvio padrão da frequência dominante (Hz) |
| `cv_dominant_freq_hz` | float | Coeficiente de variação da frequência dominante |
| `min_dominant_freq_hz` | float | Frequência dominante mínima observada na cadeia (Hz) |
| `max_dominant_freq_hz` | float | Frequência dominante máxima observada na cadeia (Hz) |

### Amplitude espectral

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `mean_dominant_amp` | float | Amplitude média no bin dominante dos cliques (magnitude STFT) |
| `std_dominant_amp` | float | Desvio padrão da amplitude dominante |

### Largura de banda

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `mean_bw_3db_hz` | float | Largura de banda média a −3 dB em torno do pico espectral (Hz) |
| `std_bw_3db_hz` | float | Desvio padrão da largura de banda a −3 dB (Hz) |
| `mean_bw_10db_hz` | float | Largura de banda média a −10 dB em torno do pico espectral (Hz) |
| `std_bw_10db_hz` | float | Desvio padrão da largura de banda a −10 dB (Hz) |

### Energia por subfaixa (10 bandas uniformes dentro de `[flow, fhig]`)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `mean_band_energy_1` … `mean_band_energy_10` | float | Energia média absoluta (soma de potência) em cada subfaixa, calculada sobre todos os cliques da cadeia |
| `mean_band_energy_rel_1` … `mean_band_energy_rel_10` | float | Energia média relativa em cada subfaixa (normalizada pela soma das 10 bandas) — representa o perfil espectral médio da cadeia |

As 10 subfaixas são distribuídas uniformemente entre `flow` e `fhig`. Com os padrões `flow=75 kHz` e `fhig=125 kHz`, cada subfaixa tem 5 kHz de largura: 75–80, 80–85, …, 120–125 kHz.

### Centróide espectral

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `spectral_centroid_hz` | float | Centro de gravidade espectral da cadeia (Hz), calculado como média ponderada dos centros das subfaixas pela energia média de cada banda |

## Comportamento de append (streaming)

- As linhas são escritas em `all_chains.csv` imediatamente após cada arquivo ser processado (`mode='a'`).
- O cabeçalho é escrito apenas quando o arquivo ainda não existe ou está vazio.
- Para reiniciar o processamento do zero, apague `all_chains.csv` antes de executar — caso contrário, os áudios reprocessados gerarão linhas duplicadas.

## Estrutura de saída esperada

```
output_dir/
└── all_chains.csv   ← cadeias de todos os áudios
```
