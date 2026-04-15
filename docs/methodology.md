# Metodologia e evolução do detector

## Formulação original

A versão inicial combinava as ODFs por produto logarítmico:

```
Z = log10(dHFC * SF * WPD * CD + 1)
```

Essa abordagem apresentou limitações críticas em dados reais:

- Colapso numérico (`Z ≈ 0`) quando qualquer ODF era próxima de zero
- Thresholds efetivos extremamente baixos (~1e-8), sem significado interpretável
- Baixa sensibilidade em gravações ruidosas

## Fusão aditiva normalizada (formulação atual)

A substituição pelo modelo aditivo resolveu o colapso numérico:

### 1. Normalização robusta por percentil

```python
x_norm = x / (percentile(x, 95) + eps)
```

Cada ODF é escalada individualmente em relação ao seu próprio percentil 95. Isso torna os pesos comparáveis entre ODFs com ordens de magnitude diferentes.

### 2. Combinação ponderada

```python
Z = 0.35 * dHFC_n + 0.35 * SF_n + 0.15 * WPD_n + 0.15 * CD_n
```

Os pesos refletem a maior relevância do dHFC e SF para detecção de cliques de odontocetos, com WPD e CD como contribuições auxiliares.

### Impacto

- Eliminação do colapso do score
- Distribuição de Z com amplitude significativa
- Thresholds interpretáveis na faixa ~0.3–4.0
- Viabilidade de uso em dados massivos

## Modos de score Z

O parâmetro `--z-mode` controla como o score final é calculado:

### `raw`
Z direto da fusão aditiva. Sensível à energia absoluta do sinal.

### `contrast`
Contraste local calculado via mediana e MAD (Median Absolute Deviation) em janela deslizante:

```
z_contrast = (Z - median_local(Z)) / (MAD_local(Z) + eps)
```

Reduz a influência de regimes de ruído elevado, aumentando o contraste entre fundo e evento.

### `hybrid`
Combinação linear entre `raw` e `contrast`:

```
z_hybrid = alpha * z_raw + (1 - alpha) * z_contrast
```

Permite ajuste fino entre sensibilidade absoluta e contraste local.

> Para primeiros testes com `contrast` ou `hybrid`, use `--thr-mode adaptive` para evitar comparações de escala incompatíveis com thresholds fixos calibrados para `raw`.

## Limitação em dados oceânicos reais

Em ambientes com ruído ambiental intenso (chuva, ruído de embarcações, biofonia densa):

- O score Z responde a qualquer descontinuidade energética, não apenas a cliques
- Ruído impulsivo ambiental eleva sistematicamente as ODFs
- Resultado: aumento de falsos positivos e perda de contraste evento/fundo

O detector deve ser interpretado como um **detector de descontinuidades energéticas na banda de análise**, não exclusivamente como um detector de cliques de odontocetos.

## Métricas auxiliares para qualificação de janelas

Durante o processamento, métricas acústicas por janela de médio prazo são computadas internamente (RMS, crest factor, peak-to-p95 ratio, estatísticas do Z) e utilizadas para guiar o mecanismo de penalização e recompensa adaptativa dos thresholds. Essas métricas **não são mais salvas em CSV**; servem exclusivamente como sinal de controle do threshold adaptativo dentro de cada execução.

| Métrica interna | Uso |
|---------|---------------|
| `event_rate` alto | Aciona penalização do threshold na janela seguinte |
| `peak_to_p95_ratio` baixo | Aciona penalização do threshold na janela seguinte |
| `rms` baixo + `peak_to_p95_ratio` alto | Aciona recompensa (afrouxamento) do threshold |
