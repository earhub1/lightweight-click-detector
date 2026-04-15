# Pipeline de detecção

## Visão geral

O detector processa arquivos `.wav` em lotes e produz um único CSV consolidado (`all_chains.csv`) no diretório de saída, acumulando as cadeias de cliques detectadas em todos os arquivos processados.

## Fluxo detalhado

### 1. Localização dos arquivos

`event_detection.py` varre o diretório de entrada (`--input`) de forma recursiva (padrão) ou plana, buscando arquivos `.wav`.

### 2. Segmentação em janelas de médio prazo

Cada arquivo é lido em blocos de `mid_term_duration` segundos (padrão: 1 s). Esse parâmetro controla o tamanho da janela usada para estatísticas de qualidade acústica.

### 3. Extração de ODFs por STFT

Em cada janela de médio prazo, o sinal é analisado via STFT com janela de Hann de tamanho `short_term_duration * fs` (sem sobreposição entre frames). Quatro funções de detecção de onset (ODFs) são extraídas:

| ODF | Descrição |
|-----|-----------|
| `dHFC` | Variação positiva do conteúdo de alta frequência na banda `[flow, fhig]` |
| `SF` | Spectral Flux — variação positiva da magnitude espectral na banda |
| `WPD` | Weighted Phase Deviation — desvio de fase ponderado por magnitude |
| `CD` | Complex Domain — distância complexa entre frames consecutivos |

### 4. Combinação em score Z

As ODFs são combinadas por fusão aditiva após normalização robusta por percentil:

```
x_norm = x / (percentile(x, 95) + eps)

Z = w_hfc * dHFC_n + w_sf * SF_n + w_wpd * WPD_n + w_cd * CD_n
```

Pesos padrão: `dHFC=0.35`, `SF=0.35`, `WPD=0.15`, `CD=0.15`.

O score Z pode operar em três modos (`--z-mode`):
- `raw` — Z direto da fusão aditiva
- `contrast` — contraste local via mediana/MAD em janela deslizante
- `hybrid` — combinação linear entre `raw` e `contrast`

### 5. Aplicação de limiares

Dois limiares definem três zonas:

- `z >= high_thr` → evento `high`
- `low_thr <= z < high_thr` → evento `low`
- `z < low_thr` → silêncio

Os limiares podem ser **fixos** (`--thr-mode fixed`) ou **adaptativos** (`--thr-mode adaptive`), calculados como quantis de Z sobre o arquivo inteiro.

### 6. Pós-processamento temporal (event-level)

As máscaras binárias frame-a-frame passam por três etapas:

1. **Filtro de duração mínima** (`--min-event-duration-ms`): remove eventos muito curtos.
2. **Fusão de eventos próximos** (`--merge-gap-ms`): une eventos separados por gap menor que o limiar.
3. **Período refratário** (`--refractory-ms`): suprime disparos redundantes próximos, mantendo o mais forte.

### 7. Agrupamento em cadeias (chain-level)

Eventos `high` consecutivos são agrupados em cadeias de cliques com base no ICI (inter-click interval) médio da cadeia. Uma cadeia é válida se contiver ao menos `--min-clicks-in-chain` cliques.

### 8. Saídas

Para cada arquivo processado, as cadeias de cliques detectadas são escritas via append em um CSV consolidado no diretório de saída:

- **`all_chains.csv`** — cadeias de cliques (chain-level), com coluna `audio_filepath` como primeira coluna

Ver [`outputs.md`](outputs.md) para o schema completo.
