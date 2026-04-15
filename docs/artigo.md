## Introdução

A detecção de sinais acústicos impulsivos em ambientes marinhos constitui um dos principais desafios no contexto do monitoramento acústico passivo (PAM), especialmente quando se trata de espécies de odontocetos. Esses animais utilizam sinais do tipo _click_ como mecanismo central de ecolocalização, permitindo a percepção do ambiente, a localização de presas e a navegação em meios de baixa visibilidade. Do ponto de vista acústico, tais sinais caracterizam-se por apresentarem curta duração temporal, elevada concentração de energia e natureza altamente impulsiva, frequentemente distribuída em faixas de frequência específicas. Essas propriedades tornam os _clicks_ biologicamente relevantes e, ao mesmo tempo, detectáveis por meio de técnicas adequadas de processamento de sinais.

No âmbito do projeto PAMA, a aquisição contínua de dados acústicos resulta em um volume expressivo de gravações, cuja análise manual torna-se operacionalmente inviável. Nesse cenário, a identificação de momentos de atividade de odontocetos passa a demandar o uso de ferramentas automatizadas capazes de localizar, de forma eficiente, trechos de interesse dentro de grandes conjuntos de dados. No entanto, a complexidade do ambiente acústico marinho — caracterizado pela presença de ruídos ambientais, interferências antrópicas e sobreposição de diferentes fontes sonoras — impõe restrições adicionais ao desenvolvimento dessas soluções, exigindo abordagens que conciliem sensibilidade à impulsividade dos sinais com robustez frente ao ruído.

Diante desse contexto, emerge a necessidade de desenvolvimento de ferramentas computacionais leves, que sejam capazes de operar de forma eficiente sobre grandes volumes de dados, sem depender de arquiteturas complexas ou alto custo computacional. Tais ferramentas devem ser capazes não apenas de detectar eventos impulsivos compatíveis com _clicks_ de odontocetos, mas também de fornecer descritores quantitativos que representem características relevantes desses sinais, como sua impulsividade e variabilidade temporal. Esses descritores podem, futuramente, subsidiar análises mais aprofundadas sobre padrões comportamentais e dinâmicas acústicas associadas à atividade dos animais.

Assim, o presente trabalho insere-se no esforço de desenvolvimento de uma abordagem baseada em processamento de sinais para detecção automática de eventos impulsivos em dados do projeto PAMA, buscando estabelecer uma base metodológica que permita, de forma eficiente e interpretável, a identificação de momentos de atividade acústica e a extração de índices representativos do comportamento dos _clicks_ de odontocetos.

## Objetivos do Trabalho

O presente trabalho tem como objetivo principal o desenvolvimento de uma ferramenta computacional leve, baseada em técnicas de processamento de sinais, capaz de detectar automaticamente momentos de atividade acústica impulsiva compatíveis com _clicks_ de odontocetos em grandes volumes de dados provenientes de monitoramento acústico passivo.

Como objetivos específicos, destacam-se:

- Desenvolver um método de detecção automática fundamentado na extração de índices espectro-temporais associados à impulsividade do sinal acústico;
- Implementar uma estratégia de fusão desses índices em um escore composto, capaz de representar de forma integrada diferentes evidências de ocorrência de eventos impulsivos;
- Permitir a identificação temporal de eventos candidatos a _clicks_, gerando saídas estruturadas que possibilitem a análise posterior dos dados;
- Investigar a influência dos pesos atribuídos aos diferentes índices na qualidade da detecção, buscando uma calibração coerente com a morfologia acústica dos sinais de interesse;
- Avaliar estratégias de definição de thresholds, incluindo abordagens adaptativas, visando lidar com a variabilidade acústica entre diferentes trechos e arquivos;
- Construir uma base de descritores que possam, futuramente, ser utilizados na caracterização quantitativa da impulsividade e da dinâmica temporal dos eventos acústicos;
- Preparar a integração de dados anotados que permitam a avaliação quantitativa do desempenho da ferramenta por meio de métricas apropriadas.

## Referencial Teórico

A análise e detecção de sinais acústicos impulsivos em bioacústica marinha fundamenta-se, primordialmente, em técnicas de processamento de sinais capazes de representar, de forma adequada, a variação temporal e espectral do áudio. No caso dos _clicks_ de odontocetos, essa necessidade torna-se ainda mais evidente, uma vez que tais sinais apresentam curta duração, elevada energia concentrada em intervalos temporais reduzidos e distribuição espectral característica, frequentemente associada a faixas de frequência específicas. Essas propriedades tornam os _clicks_ eventos altamente transitórios, cuja detecção exige ferramentas sensíveis a mudanças abruptas no sinal.

### Representação tempo-frequência: STFT e espectrograma

Uma das ferramentas fundamentais para a análise desses sinais é a Transformada de Fourier de Curto Prazo (_Short-Time Fourier Transform_ — STFT), que permite decompor o sinal acústico em componentes espectrais ao longo do tempo. Matematicamente, a STFT de um sinal ( x(t) ) é definida como:

[
X(t, f) = \sum\_{n=-\infty}^{\infty} x[n] \cdot w[n - t] \cdot e^{-j 2\pi f n}
]

onde:

- ( w[n] ) é uma função janela (por exemplo, janela de Hann),
- ( t ) representa o deslocamento temporal,
- ( f ) corresponde à frequência.

A partir da STFT, obtém-se o espectrograma, definido como a magnitude ao quadrado da transformada:

[
S(t, f) = |X(t, f)|^2
]

O espectrograma constitui uma ferramenta essencial para a inspeção visual do sinal, permitindo identificar eventos impulsivos como estruturas verticais de alta energia, típicas dos _clicks_ de odontocetos. Essa representação também serve como base para o cálculo de descritores quantitativos utilizados na detecção automática.

### Índices de detecção de eventos impulsivos

Para capturar as características transitórias dos _clicks_, são utilizados diferentes índices derivados do espectrograma, cada um sensível a aspectos específicos da dinâmica do sinal.

#### 1. High Frequency Content (HFC)

O índice de conteúdo de alta frequência mede a concentração de energia em regiões de maior frequência, sendo particularmente sensível a eventos impulsivos. Uma formulação típica é:

[
HFC(t) = \sum_{k = k_{min}}^{k_{max}} f_k \cdot |X(t, f_k)|^2
]

onde ( f_k ) representa as frequências discretizadas dentro da banda de interesse.

**Interpretação:**
Eventos impulsivos, como _clicks_, tendem a apresentar maior contribuição em altas frequências, resultando em valores elevados de HFC. Esse índice reflete diretamente a presença de energia concentrada e abrupta no espectro.

#### 2. Spectral Flux (SF)

O fluxo espectral quantifica a variação da magnitude espectral entre quadros consecutivos:

[
SF(t) = \sum_{k} \left( |X(t, f_k)| - |X(t-1, f_k)| \right)^2
]

Em algumas formulações, considera-se apenas a variação positiva.

**Interpretação:**
Como os _clicks_ são eventos abruptos, eles provocam mudanças rápidas no conteúdo espectral, resultando em picos no fluxo espectral. Assim, o SF é um indicador de transições súbitas no sinal.

#### 3. Weighted Phase Deviation (WPD)

O desvio de fase ponderado avalia a consistência da fase entre quadros consecutivos, ponderando essa informação pela magnitude espectral:

[
WPD(t) = \sum_{k} |X(t, f_k)| \cdot \left| \phi(t, f_k) - 2\phi(t-1, f_k) + \phi(t-2, f_k) \right|
]

onde ( \phi(t, f_k) ) representa a fase do sinal.

**Interpretação:**
Eventos impulsivos introduzem descontinuidades na evolução da fase, aumentando o valor do WPD. Esse índice é particularmente útil para capturar irregularidades não evidentes apenas na magnitude.

#### 4. Complex Domain (CD)

O domínio complexo combina magnitude e fase para medir a diferença entre o espectro atual e uma previsão baseada no comportamento passado:

[
CD(t) = \sum_{k} \left| X(t, f_k) - \hat{X}(t, f_k) \right|
]

onde ( \hat{X}(t, f_k) ) é uma estimativa do espectro baseada em quadros anteriores.

**Interpretação:**
Esse índice captura desvios inesperados no sinal, sendo sensível a eventos transitórios que não seguem a dinâmica previsível do fundo acústico.

### Combinação dos índices: escore composto

Considerando que cada índice captura uma dimensão distinta do comportamento impulsivo, adota-se uma estratégia de fusão para integrar essas informações em um único escore:

[
z_{raw}(t) = w_{HFC} \cdot \tilde{HFC}(t) + w_{SF} \cdot \tilde{SF}(t) + w_{WPD} \cdot \tilde{WPD}(t) + w_{CD} \cdot \tilde{CD}(t)
]

onde:

- ( \tilde{\cdot} ) indica versões normalizadas dos índices (por exemplo, via percentil),
- ( w_i ) são pesos atribuídos a cada componente.

**Interpretação:**
O escore ( z\_{raw} ) representa uma medida integrada da impulsividade do sinal. Valores elevados indicam alta probabilidade de ocorrência de eventos impulsivos compatíveis com _clicks_ de odontocetos.

### Relação com a natureza dos clicks de odontocetos

Os _clicks_ de odontocetos apresentam características acústicas que justificam a escolha dos índices utilizados:

- **Impulsividade temporal:** capturada por SF e CD;
- **Alta concentração de energia:** refletida no HFC;
- **Descontinuidade estrutural do sinal:** evidenciada por WPD e CD;
- **Localização espectral:** considerada na seleção da banda de frequência analisada.

Assim, a combinação desses descritores permite construir uma representação robusta do fenômeno acústico, capaz de diferenciar eventos biológicos de ruídos ambientais e interferências externas.

### Considerações finais do referencial

O uso de técnicas baseadas em STFT e índices espectro-temporais configura uma abordagem computacionalmente eficiente e interpretável, adequada para o processamento de grandes volumes de dados acústicos. Ao evitar modelos excessivamente complexos, essa estratégia permite maior controle sobre os parâmetros do detector e facilita a análise crítica dos resultados, aspecto fundamental em aplicações científicas e bioacústicas.

## Metodologia (versão detalhada)

A metodologia adotada neste trabalho foi estruturada de forma incremental, combinando conhecimento prévio sobre o fenômeno bioacústico de interesse com etapas sucessivas de calibração orientadas por análise diagnóstica. Essa abordagem foi concebida de modo a garantir não apenas a aplicação da ferramenta em larga escala, mas também a interpretabilidade dos resultados obtidos, aspecto essencial em estudos científicos baseados em processamento de sinais.

### Aquisição e caracterização dos dados

Os dados utilizados neste estudo foram obtidos no âmbito do projeto PAMA, conduzido pelo laboratório EAR HUB da Universidade Federal do Rio Grande do Norte (UFRN), em parceria com as empresas CLS e TGS, que atuam no ambiente oceânico da costa do Maranhão. As gravações foram realizadas por meio de hidrofones, em regime de monitoramento acústico passivo (PAM), permitindo a captura contínua da paisagem sonora submarina sem interferência ativa no ambiente.

Os sinais foram adquiridos com taxa de amostragem de 128 kHz, o que implica uma frequência de Nyquist de 64 kHz. Essa configuração foi adotada considerando o conhecimento prévio de que os _clicks_ de odontocetos apresentam componentes relevantes em altas frequências. Atualmente, o acervo disponível compreende mais de um ano de gravações contínuas, armazenadas em servidor institucional, o que impõe desafios computacionais e metodológicos relacionados ao processamento em larga escala.

### Definição da faixa espectral de análise

A etapa inicial da metodologia consistiu na incorporação do conhecimento biológico e acústico do sinal de interesse. Com base na literatura e na observação preliminar dos dados, definiu-se uma faixa de frequência de análise compreendida entre 20 kHz e 50 kHz, considerada representativa para a detecção dos _clicks_ no conjunto de dados estudado. Essa seleção é fundamental, pois orienta diretamente o cálculo dos índices espectro-temporais, restringindo a análise às regiões do espectro onde os eventos impulsivos apresentam maior evidência.

### Execução inicial com parâmetros _default_

Uma etapa central da metodologia — e frequentemente subestimada — consiste na execução inicial da ferramenta `event_detection.py` utilizando parâmetros _default_, incluindo pesos dos índices e thresholds de detecção predefinidos. Essa etapa não tem como objetivo produzir resultados finais ou diretamente utilizáveis, mas sim cumprir uma função exploratória e diagnóstica essencial para a calibração do sistema.

Ao executar o detector com valores padrão sobre uma amostra representativa dos áudios, obtém-se uma primeira visão do comportamento dos índices acústicos (HFC, SF, WPD e CD) e do escore composto ( z\_{raw} ) em relação ao sinal real. Essa execução inicial permite observar, em dados reais, como cada descritor responde a diferentes estruturas acústicas presentes no ambiente, incluindo _clicks_, ruídos de fundo, interferências antrópicas e outros eventos biológicos.

Do ponto de vista metodológico, essa etapa desempenha três funções principais:

1. **Caracterização empírica do comportamento dos índices**
   Embora os índices possuam definições matemáticas bem estabelecidas, sua resposta prática depende fortemente do contexto acústico específico. A execução com parâmetros _default_ permite verificar, por exemplo, quais índices apresentam maior sensibilidade à presença de _clicks_ e quais são mais suscetíveis a ruídos ou artefatos.

2. **Identificação de padrões e discrepâncias**
   A análise dos resultados iniciais possibilita identificar situações em que o detector responde de forma adequada (detectando eventos compatíveis com _clicks_) e situações em que há excesso de falsos positivos ou falsos negativos. Essas discrepâncias fornecem indícios sobre ajustes necessários nos parâmetros do modelo.

3. **Estabelecimento de uma referência inicial de calibração**
   Os resultados obtidos com os parâmetros padrão funcionam como uma linha de base (_baseline_), a partir da qual são realizadas as etapas subsequentes de ajuste. Essa referência é importante para avaliar se as modificações introduzidas no sistema de fato produzem melhorias consistentes.

É importante ressaltar que, para que essa etapa seja eficaz, a amostra de áudio utilizada deve conter ocorrências reais de _clicks_. A ausência desses eventos compromete a capacidade de interpretar a resposta dos índices e inviabiliza a construção de uma “régua” adequada para a detecção.

### Análise diagnóstica e calibração dos parâmetros

Após a execução inicial, procede-se à análise detalhada dos resultados por meio da ferramenta de visualização `detection_spectrograms.py`. Essa ferramenta permite a inspeção simultânea do espectrograma, dos sinais individuais dos índices e do escore composto ao longo do tempo, em janelas específicas do áudio.

Essa etapa é metodologicamente crucial, pois possibilita a correlação direta entre:

- a morfologia do sinal no espectrograma (estrutura visual dos _clicks_);
- o comportamento dos índices individuais;
- a resposta do escore ( z\_{raw} ).

A partir dessa análise, são realizados ajustes iterativos nos pesos atribuídos a cada índice, de modo a privilegiar aqueles que apresentam maior aderência ao fenômeno de interesse. De forma complementar, são ajustados os thresholds de detecção, buscando uma separação mais consistente entre eventos impulsivos reais e ruído de fundo.

Essa calibração não é conduzida de forma arbitrária, mas orientada pela interpretação conjunta dos dados visuais e numéricos. Em termos práticos, busca-se alinhar o comportamento do escore composto com a percepção visual dos eventos no espectrograma, garantindo que picos relevantes correspondam a valores elevados de ( z\_{raw} ).

### Execução em lote e geração de saídas

Uma vez definida uma configuração inicial adequada, a ferramenta é aplicada em lote sobre diretórios contendo arquivos `.wav`. O processamento gera arquivos de saída no formato `.csv`, contendo os instantes temporais em que eventos candidatos foram detectados.

Essas saídas permitem:

- a identificação automática de trechos com atividade acústica;
- a redução do esforço de inspeção manual;
- a organização dos dados para análises posteriores;
- a construção futura de métricas e descritores associados à impulsividade dos eventos.

### Reprodutibilidade da metodologia

A metodologia proposta foi estruturada de forma a ser replicável por outros pesquisadores. O fluxo de aplicação pode ser descrito como um procedimento sistemático composto pelas seguintes etapas:

1. Conhecimento prévio do sinal biológico de interesse;
2. Definição da faixa espectral relevante;
3. Execução inicial do detector com parâmetros _default_;
4. Análise diagnóstica dos índices e do espectrograma;
5. Ajuste dos pesos e thresholds com base na resposta observada;
6. Aplicação da ferramenta em lote e geração dos resultados.

### Limitações e estágio atual

Até o presente momento, a versão consolidada da metodologia baseia-se no uso de thresholds fixos, definidos a partir da calibração orientada por inspeção diagnóstica. Embora estratégias de threshold adaptativo estejam em investigação, elas ainda não foram incorporadas à versão final disponibilizada ao leitor.

Adicionalmente, a etapa de avaliação quantitativa, baseada em dados anotados e métricas de desempenho, encontra-se em desenvolvimento. A recente obtenção desses dados permitirá, em etapas futuras, validar de forma objetiva a eficácia da ferramenta proposta.

## Integração de novos dados anotados e avanço para a etapa de validação

No estágio atual do trabalho, uma frente particularmente relevante tem sido a incorporação de novos sinais provenientes de um laboratório de Portugal, os quais apresentam características acústicas compatíveis com o tipo de dado utilizado no projeto PAMA, embora tenham sido obtidos em uma região geográfica distinta. Essa expansão do conjunto de análise possui importância metodológica significativa, pois permite submeter a ferramenta desenvolvida a uma condição mais próxima de um teste de robustez, no qual o detector deixa de ser observado apenas em função do ambiente original de desenvolvimento e passa a ser confrontado com outro contexto acústico real, ainda que preserve propriedades bioacústicas semelhantes.

A relevância desse novo material não reside apenas na disponibilidade de áudios adicionais, mas principalmente no fato de que esses dados encontram-se acompanhados de identificações manuais dos _clicks_. Esse aspecto representa um avanço decisivo para o trabalho, uma vez que, até então, o processo de desenvolvimento vinha sendo conduzido predominantemente com base em análise qualitativa, inspeção visual e calibração orientada pelo comportamento dos índices e do espectrograma. Embora essa etapa tenha sido essencial para a construção metodológica da ferramenta, a presença de anotações manuais passa agora a oferecer uma referência externa concreta para a avaliação do desempenho do detector.

Do ponto de vista científico, a disponibilidade de dados anotados permite a transição de uma fase de desenvolvimento orientada por evidências qualitativas para uma etapa de validação quantitativa e qualitativa integrada. Em outras palavras, deixa-se de avaliar a coerência da ferramenta apenas com base na inspeção visual dos eventos detectados e passa-se a verificar, de forma mensurável, em que grau as detecções automáticas correspondem aos eventos efetivamente identificados por especialistas humanos. Isso modifica substancialmente o nível de maturidade do estudo, pois torna possível examinar objetivamente se os ajustes de pesos e thresholds estão, de fato, produzindo maior acurácia na identificação dos _clicks_.

Nesse contexto, a principal contribuição dos novos dados portugueses é possibilitar a construção de uma etapa de avaliação metodologicamente mais sólida. A presença simultânea dos áudios e das marcações manuais torna viável comparar os instantes detectados automaticamente com os instantes anotados por especialistas, permitindo verificar não apenas se o detector está localizando os eventos corretos, mas também se os parâmetros atualmente em calibração estão coerentes com a realidade do fenômeno acústico. Assim, pesos mais adequados e thresholds mais bem ajustados deixam de ser definidos apenas pela aparência do escore ( z\_{raw} ) ou pela resposta visual no espectrograma, passando a ser avaliados também pela sua capacidade de maximizar a correspondência entre a detecção automática e a referência manual.

Essa nova etapa é particularmente importante porque o estudo, até o momento, vem demonstrando que a definição dos pesos dos índices e dos limiares de detecção exerce influência direta sobre a qualidade dos resultados. Observou-se, por exemplo, que determinadas configurações podem tornar o detector excessivamente sensível ao ruído, especialmente em eventos classificados como de baixa intensidade, enquanto outras podem favorecer a identificação dos eventos mais evidentes, mas ainda deixar lacunas em situações menos contrastantes. Sem uma base anotada, essas observações permanecem restritas ao campo da interpretação qualitativa; com os novos dados, passa a ser possível verificar quantitativamente quais configurações efetivamente produzem melhor desempenho.

Além disso, a utilização de dados provenientes de outra região adiciona um componente importante de generalização metodológica. Como os sinais possuem propriedades comparáveis às dos dados do PAMA, mas não pertencem exatamente ao mesmo cenário acústico, sua incorporação permite investigar até que ponto a ferramenta mantém comportamento consistente fora do conjunto original de desenvolvimento. Esse aspecto é relevante porque, em aplicações reais de monitoramento acústico passivo, a variabilidade ambiental constitui uma das principais dificuldades para a construção de detectores confiáveis. Dessa forma, se a ferramenta demonstrar desempenho satisfatório também nesse novo conjunto, haverá evidência mais robusta de que a abordagem proposta não está excessivamente ajustada a um contexto único, mas possui potencial de aplicação mais ampla em dados de natureza semelhante.

A incorporação desses novos sinais também viabiliza uma análise dupla, qualitativa e quantitativa, que se mostra especialmente adequada ao estágio do projeto. A análise qualitativa continuará sendo indispensável, pois permite interpretar a relação entre a morfologia do evento no espectrograma, o comportamento dos índices individuais e a resposta do escore composto. Já a análise quantitativa permitirá medir a aderência do detector aos eventos anotados, conferindo maior rigor às decisões de calibração. A combinação dessas duas perspectivas é metodologicamente valiosa: a primeira preserva a interpretabilidade do processo, enquanto a segunda oferece objetividade à avaliação.

Sob esse prisma, o uso dos dados anotados de Portugal representa não apenas uma ampliação do material empírico disponível, mas a consolidação de uma etapa decisiva do estudo. É a partir desse novo conjunto que se torna possível avançar da calibração exploratória para uma validação formal da ferramenta, com base em evidências mensuráveis. Consequentemente, essa etapa tende a desempenhar papel central no fechamento do trabalho, pois permitirá verificar se os ajustes realizados até o presente momento conduzem, de fato, a um detector mais preciso, mais robusto e mais coerente com o fenômeno bioacústico de interesse.

Em termos práticos, essa fase abre caminho para a definição de métricas de desempenho aplicadas à detecção de _clicks_, bem como para a seleção mais criteriosa das configurações finais de pesos e thresholds. Com isso, será possível concluir o estudo com uma base de avaliação mais completa, sustentada tanto pela análise visual diagnóstica quanto por resultados quantitativos obtidos a partir de referências manuais. Assim, a integração desses novos dados constitui um marco metodológico no desenvolvimento da ferramenta, ao permitir que a fase de experimentação e ajuste evolua para uma etapa de validação capaz de sustentar, com maior rigor, as conclusões finais do trabalho.
