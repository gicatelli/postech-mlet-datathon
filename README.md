# Datathon 7MLET - Grupo 13

## Visão do Problema

Uma instituição financeira digital precisa decidir, em diferentes canais, qual oferta, mensagem ou próximo passo apresentar para cada cliente elegível. Regras fixas e testes A/B longos desperdiçam tráfego, demoram para reagir a mudanças de contexto e dificultam a personalização responsável.

Este projeto implementa uma **plataforma de experimentação adaptativa** usando Multi-Armed Bandit (Thompson Sampling) para identificar comportamentos distintos, equilibrar exploração e explotação e aprender com respostas observadas — sem congelar a decisão em regras estáticas.

A solução demonstra o ciclo completo de Machine Learning Engineering: formulação do problema, construção de baselines, versionamento de dados, serviço de componentes, avaliação de qualidade, monitoramento e documentação de limitações.

---

## Base de Dados

- **Dataset**: [bank-marketing (henriqueyamahata)](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
- **Fonte**: Kaggle
- **Licença**: CC BY 4.0
- **Registros**: 41.188
- **Features**: 20 atributos (demográficos, de campanha e socioeconômicos)
- **Target**: `y` (subscreveu depósito a prazo: yes/no)
- **Coluna descartada**: `duration` (vazamento temporal — só conhecida após o contato)

### Colunas do Dataset

| Tipo | Colunas |
|------|---------|
| Perfil do cliente | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Campanha atual | `contact`, `month`, `day_of_week`, `campaign` |
| Campanha anterior | `pdays`, `previous`, `poutcome` |
| Indicadores socioeconômicos | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |
| Target | `y` |

---

## Como Executar

### No Kaggle (recomendado para avaliadores):
1. Acesse o notebook no Kaggle: [link do notebook]
2. Clique em **"Copy & Edit"**
3. Verifique que o dataset "Bank Marketing" está adicionado no Input
4. Clique em **"Run All"**
5. Todas as seções executam sequencialmente sem necessidade de configuração

### Localmente:
```bash
git clone https://github.com/gicatelli/postech-mlet-datathon.git
cd postech-mlet-datathon
pip install -r requirements.txt
jupyter notebook notebook/datathon-7mlet.ipynb
```
> Nota: Para execução local, coloque o arquivo `bank-additional-full.csv` na pasta `data/`.

---

## Estrutura do Projeto

```
postech-mlet-datathon/
├── docs/
│   └── DOCUMENTACAO.md       # Documentação técnica detalhada
├── notebook/
│   └── datathon-7mlet.ipynb  # Notebook principal (executar no Kaggle)
├── requirements.txt          # Dependências Python
├── LICENSE
└── README.md
```

---

## Algoritmos

### Baseline (Regra Fixa)
Política determinística que sempre recomenda a oferta com maior taxa de conversão histórica (melhor braço fixo). Serve como referência comparativa.

### Thompson Sampling Contextual (Algoritmo Adaptativo)
Abordagem bayesiana que modela a incerteza sobre a taxa de conversão de cada braço usando distribuição Beta. A cada decisão, amostra valores de cada distribuição e seleciona o braço com maior amostra. Utiliza segmentação contextual por perfil do cliente.

**Braços (ofertas)**:
| Braço | Oferta | Critério de Favorecimento |
|-------|--------|--------------------------|
| 0 | Depósito a prazo (padrão) | Oferta base, sem ajuste |
| 1 | Depósito com taxa premium | Clientes contatados 1x (alta qualificação) |
| 2 | Produto alternativo (empréstimo) | Clientes jovens (age < 35) |
| 3 | Controle (não contatar) | Reward sempre 0 |

**Segmentos contextuais**:
| Segmento | Critério |
|----------|----------|
| `historico_positivo` | `sucesso_anterior == 1` (poutcome anterior foi success) |
| `jovem` | `age < 30` |
| `senior` | `age > 55` |
| `alta_qualificacao` | `campaign == 1` (converteu com apenas 1 contato) |
| `padrao` | Demais clientes |

---

## Resultados

| Métrica | Baseline | Thompson Sampling |
|---------|----------|-------------------|
| Taxa de Conversão (CTR) | - | - |
| Regret Acumulado | - | - |
| Reward Acumulado | - | - |
| Regret Médio | - | - |

*Tabela preenchida automaticamente ao executar o notebook.*

---

## Golden Set (5 Casos de Teste)

| # | Perfil | Segmento Esperado | Oferta Esperada | Justificativa |
|---|--------|-------------------|-----------------|---------------|
| 1 | Jovem, estudante, primeiro contato | jovem | Produto alternativo | Jovens tendem a preferir empréstimos a depósitos |
| 2 | Adulto, gerente, alta qualificação | alta_qualificacao | Depósito premium | Contatado 1x, perfil qualificado que converte com pouco esforço |
| 3 | Aposentado, múltiplos contatos | senior | Depósito padrão | Perfil conservador, busca segurança |
| 4 | Desempregado, muitos contatos | padrao | Controle | Baixa propensão, evitar contato desnecessário |
| 5 | Técnico, sucesso anterior | historico_positivo | Depósito padrão/premium | Histórico positivo indica alta propensão |

---

## Serviço Demonstrável

O notebook inclui duas funções que simulam uma API REST:

```python
# Recomendar oferta para um cliente
resultado = recomendar_oferta({
    'age': 35, 'job': 'technician', 'campaign': 1,
    'sucesso_anterior': 1
})
# Retorna: oferta_recomendada, braco_selecionado, segmento_cliente,
#          probabilidade_conversao, confianca, requer_revisao_humana

# Registrar feedback (atualiza o modelo online)
registrar_feedback(segmento='historico_positivo', braco=0, converteu=True)
```

---

## Arquitetura Cloud (AWS)

```
Cliente -> API Gateway -> ECS/Fargate (FastAPI) -> Modelo (Thompson Sampling)
                                                        |
                                                  S3 (logs de decisão)
                                                        |
                                          Step Functions (retreino periódico)
```

| Serviço | Função |
|---------|--------|
| Amazon S3 | Armazenamento de dados, logs de decisão e artefatos |
| Amazon ECS + Fargate | API de recomendação containerizada |
| API Gateway | Exposição da API para canais digitais |
| AWS Step Functions | Orquestração do pipeline de retreino |
| CloudWatch | Monitoramento de latência, erros e drift |
| SageMaker Feature Store | Features centralizadas dos clientes |
| MLflow (EC2) | Tracking de experimentos |

---

## MLflow

Experimentos registrados no notebook:
- **baseline_regra_fixa**: métricas do baseline determinístico
- **thompson_sampling_contextual**: Thompson Sampling com segmentação por features

Parâmetros e métricas rastreados:
- `ctr`, `reward_acumulado`, `regret_acumulado`, `regret_medio`
- `ganho_vs_baseline_pct`
- Segmentos e configuração dos priors

---

## Limitações

- Dataset de uma única instituição portuguesa — pode não generalizar para outros mercados
- Apenas 1 produto real no dataset; braços são simulados para demonstração
- Dados estáticos, não captura mudanças em tempo real (streaming)
- Desbalanceamento (~11% positivos) pode enviesar o bandit
- Sem dados de custo real por contato — usa reward binário como proxy
- Cold start: novos segmentos sem histórico partem de prior uniforme
- Thompson Sampling assume independência entre decisões (sem efeito de saturação)

---

## Governança e Ética

- **Base legal**: Legítimo interesse para otimização de campanhas (LGPD Art. 10)
- **Finalidade**: Recomendar ofertas personalizadas maximizando conversão e minimizando contatos desnecessários
- **Minimização**: Apenas features relevantes utilizadas. Sem identificadores pessoais, patrimônio, renda direta, gênero ou raça
- **Retenção**: Dados de decisão retidos 12 meses para auditoria, depois anonimizados
- **Humano no loop**: Casos de baixa confiança (probabilidade < 0.1) sinalizados para revisão manual
- **Monitoramento de viés**: Acompanhamento da distribuição de ofertas por segmento

---

## Equipe

- **Giovanna Catelli** — Grupo 13

---

## Vídeo de Apresentação

[Link do vídeo YouTube — até 5 minutos]
