# Referência de parâmetros

Todos os parâmetros são passados via linha de comando para `event_detection.py`.

## Entrada / saída

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--input` | — | Diretório com arquivos `.wav` |
| `--output` | — | Diretório de saída para os CSVs |
| `--recursive` | `True` | Busca recursiva de arquivos |
| `--pattern` | `*.wav` | Padrão glob para os arquivos |

## Segmentação temporal

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--short-term-duration` | `0.001` | Resolução temporal dos frames STFT (s) |
| `--mid-term-duration` | `1.0` | Duração das janelas de médio prazo (s) |

## Banda de frequência

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--flow` | `75000` | Frequência mínima da banda de análise (Hz) |
| `--fhig` | `125000` | Frequência máxima da banda de análise (Hz) |

## Pesos das ODFs

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--w-hfc` | `0.35` | Peso do dHFC no score Z |
| `--w-sf` | `0.35` | Peso do SF no score Z |
| `--w-wpd` | `0.15` | Peso do WPD no score Z |
| `--w-cd` | `0.15` | Peso do CD no score Z |

## Normalização

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--norm-percentile` | `95.0` | Percentil usado como referência de escala de cada ODF |

## Score Z

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--z-mode` | `raw` | Modo do score: `raw`, `contrast` ou `hybrid` |
| `--contrast-win-ms` | `100.0` | Tamanho da janela de contraste local (ms) |
| `--contrast-alpha` | `0.5` | Peso do `raw` no modo `hybrid` |

## Limiares

### Modo fixo (`--thr-mode fixed`)

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--thr-mode` | `fixed` | Modo de threshold |
| `--low-thr` | `0.05` | Limiar inferior (evento `low`) |
| `--high-thr` | `0.2` | Limiar superior (evento `high`) |

### Modo adaptativo por janela (`--thr-mode adaptive`)

Calcula os limiares individualmente para cada janela de médio prazo usando mediana + MAD sobre o score Z local (com remoção da cauda superior para robustez a cliques densos). Os thresholds são suavizados exponencialmente entre janelas e penalizados ou recompensados conforme o regime acústico detectado na janela anterior.

**Estimativa do ruído de fundo:**

```
z_bg = percentil inferior 98% de z_used_chunk  (remove top 2%)
bg_med = median(z_bg)
bg_mad = 1.4826 * median(|z_bg - bg_med|)

low_thr  = max(bg_med + k_low  * bg_mad, low_thr_floor)
high_thr = max(bg_med + k_high * bg_mad, high_thr_floor)
```

**Suavização exponencial entre janelas:**

```
low_thr[i]  = beta * low_thr[i-1]  + (1 - beta) * low_thr_local
high_thr[i] = beta * high_thr[i-1] + (1 - beta) * high_thr_local
```

**Penalização (janela suspeita de ruído):** se `event_rate_prev > penalty_event_rate_limit` e `peak_to_p95_ratio_prev < penalty_ratio_limit`, os thresholds são multiplicados pelos fatores de penalização.

**Recompensa (janela limpa):** se `rms_prev < rms_clean_limit` e `peak_to_p95_ratio_prev > ratio_good_limit`, os thresholds são multiplicados pelo `reward_factor`.

**Regra de proporção (`low_thr_ratio`):** após o cálculo via MAD e antes da suavização, aplica `low_thr_local = max(low_thr_local, low_thr_ratio * high_thr_local)`. Impede que o `low_thr` descole demais do `high_thr`, evitando que a zona cinzenta do `low` capture ruído difuso.

**Gate duplo para `low`:** no modo `adaptive`, além de cruzar `low_thr`, um frame só é classificado como `low` se também satisfazer `z_contrast >= contrast_low_thr`. Isso exige saliência local além de amplitude, diferenciando cliques fracos reais de ruído que cruzou a régua baixa.

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--low-thr-floor` | `1e-5` | Valor mínimo absoluto para o limiar `low` |
| `--high-thr-floor` | `5e-5` | Valor mínimo absoluto para o limiar `high` |
| `--bg-k-low` | `3.0` | Multiplicador do MAD para o limiar `low` |
| `--bg-k-high` | `6.0` | Multiplicador do MAD para o limiar `high` |
| `--smooth-beta` | `0.8` | Fator de suavização exponencial entre janelas (0–1) |
| `--penalty-event-rate-limit` | `50.0` | Taxa de eventos (ev/s) acima da qual a janela seguinte é penalizada |
| `--penalty-ratio-limit` | `1.5` | `peak_to_p95_ratio` abaixo do qual a janela seguinte é penalizada |
| `--penalty-factor-high` | `1.15` | Multiplicador de `high_thr` em janelas suspeitas de ruído |
| `--penalty-factor-low` | `1.10` | Multiplicador de `low_thr` em janelas suspeitas de ruído |
| `--reward-factor` | `0.95` | Multiplicador dos thresholds em janelas limpas |
| `--rms-clean-limit` | `0.01` | RMS máximo para classificar uma janela como limpa |
| `--ratio-good-limit` | `3.0` | `peak_to_p95_ratio` mínimo para classificar uma janela como limpa |
| `--low-thr-ratio` | `0.75` | Fração mínima de `high_thr` que `low_thr` deve atingir (regra de proporção) |
| `--contrast-low-thr` | `2.5` | Z-contrast local mínimo exigido para classificar um frame como `low` (gate duplo) |

> Os parâmetros `--low-quantile` e `--high-quantile` são mantidos no CLI por compatibilidade, mas não têm efeito em `thr_mode=adaptive`.

## Pós-processamento temporal

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--min-event-duration-ms` | `1.0` | Duração mínima de um evento (ms) |
| `--merge-gap-ms` | `1.0` | Gap máximo para fusão de eventos adjacentes (ms) |
| `--refractory-ms` | `2.0` | Período refratário entre eventos (ms) |

## Cadeias de cliques

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--min-clicks-in-chain` | `5` | Mínimo de cliques para validar uma cadeia |
| `--chain-gap-factor` | `2.0` | Fator multiplicativo do ICI médio para quebra de cadeia |

## Log

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--log-stats` | `False` | Ativa logging detalhado por janela e arquivo |

## Descritores espectrais por evento

Calculados automaticamente a partir do espectro STFT no frame de pico de cada evento (média dos 3 frames centrados no pico). Não requerem parâmetros CLI adicionais — dependem de `--flow`, `--fhig` e `--short-term-duration`.

| Descritor gerado | Descrição |
|------------------|-----------|
| `dominant_freq_hz` | Frequência com maior magnitude dentro de `[flow, fhig]` |
| `dominant_amp` | Magnitude absoluta nessa frequência |
| `bw_3db_hz` | Largura de banda a −3 dB em torno do pico |
| `bw_10db_hz` | Largura de banda a −10 dB em torno do pico |
| `band_energy_1…10` | Energia (soma de potência) em cada uma das 10 subfaixas uniformes de `[flow, fhig]` |
| `band_energy_rel_1…10` | Energia relativa em cada subfaixa (normalizada pela soma total das 10 bandas) |

Esses descritores por evento são agregados em estatísticas por cadeia e salvos em `all_chains.csv`. Ver [`outputs.md`](outputs.md) para o schema completo.

## Controle do evento `low`

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--disable-low` | `False` | Zera toda a saída `low` antes do pós-processamento. Útil para analisar o `high` isoladamente (ambos os modos) |
