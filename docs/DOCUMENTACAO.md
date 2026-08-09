# Documentação Técnica — Datathon 7MLET

## Plataforma de Experimentação Adaptativa para Ofertas com Multi-Armed Bandit

**Grupo 13 — Giovanna Catelli**

---

## 1. Introdução

### 1.1 Contexto do Problema

Instituições financeiras enfrentam o desafio de decidir qual oferta apresentar a cada cliente em campanhas de marketing direto. Abordagens tradicionais como regras fixas ("oferecer o mesmo produto para todos") ou testes A/B clássicos (que exigem amostras grandes e tempo prolongado) são ineficientes: desperdiçam contatos com clientes de baixa propensão e demoram para se adaptar a mudanças no comportamento do consumidor.

### 1.2 Proposta de Solução

Implementar um sistema de **Multi-Armed Bandit com Thompson Sampling contextual** que:
- Aprende continuamente qual oferta funciona melhor para cada perfil de cliente
- Equilibra exploração (testar ofertas menos conhecidas) e explotação (usar o que já funciona)
- Personaliza decisões por segmento sem necessidade de testes A/B longos
- Converge para a política ótima ao longo do tempo

### 1.3 Escopo

O projeto utiliza o dataset Bank Marketing (UCI/Kaggle) como base para simular um ambiente de decisão com múltiplas ofertas. Embora o dataset original contenha apenas uma variável binária de conversão, criamos um cenário realista com 4 braços (ofertas) e 5 segmentos contextuais.

---

## 2. Dataset

### 2.1 Origem e Características

| Atributo | Valor |
|----------|-------|
| Nome | bank-additional-full.csv |
| Fonte | [Kaggle - henriqueyamahata](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) |
| Registros | 41.188 |
| Colunas | 21 (20 features + 1 target) |
| Período | Campanhas de marketing de uma instituição bancária portuguesa |
| Target | `y` — cliente subscreveu depósito a prazo (yes/no) |
| Desbalanceamento | ~11% positivos (yes), ~89% negativos (no) |

### 2.2 Descrição das Colunas

#### Perfil do Cliente
| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `age` | numérica | Idade do cliente |
| `job` | categórica | Tipo de emprego (admin., technician, student, retired, etc.) |
| `marital` | categórica | Estado civil (married, single, divorced) |
| `education` | categórica | Nível de escolaridade (basic.4y, high.school, university.degree, etc.) |
| `default` | categórica | Tem crédito em default? (yes, no, unknown) |
| `housing` | categórica | Tem empréstimo habitacional? (yes, no, unknown) |
| `loan` | categórica | Tem empréstimo pessoal? (yes, no, unknown) |

#### Campanha Atual
| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `contact` | categórica | Tipo de comunicação (cellular, telephone) |
| `month` | categórica | Mês do último contato |
| `day_of_week` | categórica | Dia da semana do último contato |
| `duration` | numérica | Duração do contato em segundos (**removida** — vazamento temporal) |
| `campaign` | numérica | Número de contatos realizados nesta campanha |

#### Campanha Anterior
| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `pdays` | numérica | Dias desde o último contato da campanha anterior (999 = nunca contatado) |
| `previous` | numérica | Número de contatos antes desta campanha |
| `poutcome` | categórica | Resultado da campanha anterior (success, failure, nonexistent) |

#### Indicadores Socioeconômicos
| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `emp.var.rate` | numérica | Taxa de variação do emprego (trimestral) |
| `cons.price.idx` | numérica | Índice de preços ao consumidor (mensal) |
| `cons.conf.idx` | numérica | Índice de confiança do consumidor (mensal) |
| `euribor3m` | numérica | Taxa Euribor 3 meses (diária) |
| `nr.employed` | numérica | Número de empregados (trimestral) |

### 2.3 Pré-processamento

1. **Remoção de `duration`**: Coluna removida por representar vazamento temporal — a duração do contato só é conhecida após a ligação
2. **Tratamento de `unknown`**: Valores "unknown" mantidos e codificados como categoria própria
3. **Feature Engineering**:
   - `faixa_etaria`: Categorização da idade em jovem (≤30), adulto (31-45), senior (46-55), idoso (>55)
   - `teve_contato_anterior`: Flag binária derivada de `pdays` (1 se pdays ≠ 999)
   - `sucesso_anterior`: Flag binária derivada de `poutcome` (1 se poutcome == "success")
4. **Encoding**: LabelEncoder aplicado em variáveis categóricas para compatibilidade com o modelo

---

## 3. Modelagem

### 3.1 Formulação como Multi-Armed Bandit

O problema é modelado como um **Contextual Multi-Armed Bandit** onde:
- **Agente**: Sistema de recomendação de ofertas
- **Contexto**: Features do cliente (age, campaign, sucesso_anterior, etc.)
- **Ações (braços)**: 4 ofertas possíveis
- **Recompensa**: Conversão binária (0 ou 1)

### 3.2 Definição dos Braços

| Braço | Oferta | Lógica de Reward |
|-------|--------|-----------------|
| 0 | Depósito a prazo (padrão) | `reward = target` (sem ajuste) |
| 1 | Depósito com taxa premium | `reward = target × 1.3` se `campaign == 1`, senão `target × 0.7` |
| 2 | Produto alternativo (empréstimo) | `reward = target × 1.2` se `age < 35`, senão `target × 0.5` |
| 3 | Controle (não contatar) | `reward = 0` (sempre) |

**Justificativa dos critérios:**
- **Braço 1 (premium)**: Clientes contatados apenas 1 vez (`campaign == 1`) indicam alta qualificação — convertem com menos esforço comercial, portanto são candidatos ideais para a oferta premium
- **Braço 2 (empréstimo)**: Jovens (< 35 anos) tendem a ter demandas por crédito maiores que por investimentos conservadores
- **Braço 3 (controle)**: Simula a decisão de não contatar, evitando custo em clientes de baixa propensão

### 3.3 Segmentação Contextual

A segmentação permite que o Thompson Sampling aprenda políticas diferentes para perfis distintos:

| Segmento | Critério | Racional |
|----------|----------|----------|
| `historico_positivo` | `sucesso_anterior == 1` | Clientes com conversão anterior têm alta propensão |
| `jovem` | `age < 30` | Perfil com preferências distintas de produto |
| `senior` | `age > 55` | Perfil conservador, propenso a depósitos |
| `alta_qualificacao` | `campaign == 1` | Contatados 1x e com dados na base = qualificados |
| `padrao` | Nenhum critério acima | Segmento default |

### 3.4 Thompson Sampling — Algoritmo

```
Para cada cliente t:
  1. Identificar segmento s(t) do cliente
  2. Para cada braço k = 0, ..., K-1:
     Amostrar θ_k ~ Beta(α_s[k], β_s[k])
  3. Selecionar braço a(t) = argmax_k θ_k
  4. Observar reward r(t) ∈ {0, 1}
  5. Atualizar:
     α_s[a(t)] += r(t)
     β_s[a(t)] += (1 - r(t))
```

**Prior**: Beta(1, 1) = Distribuição Uniforme (ignorância completa)

**Propriedades**:
- Convergência garantida para o braço ótimo em cada segmento
- Exploração natural: braços com mais incerteza (poucos dados) geram amostras mais variáveis
- Sem hiperparâmetros de exploração (diferente de ε-greedy ou UCB)

---

## 4. Baseline

### 4.1 Política de Regra Fixa

O baseline implementa a estratégia mais simples: calcular a taxa de conversão média de cada braço no conjunto de treino e sempre recomendar o braço com maior taxa.

**Limitações do baseline:**
- Não personaliza por segmento
- Não se adapta ao longo do tempo
- Ignora o contexto do cliente

### 4.2 Métricas de Comparação

| Métrica | Definição |
|---------|-----------|
| CTR | Média dos rewards obtidos |
| Reward Acumulado | Soma de todos os rewards |
| Regret | `max_k reward(k) - reward(a)` para cada decisão |
| Regret Acumulado | Soma dos regrets ao longo do tempo |

---

## 5. Avaliação

### 5.1 Protocolo

- **Split**: 70% treino (simulação/aprendizado) / 30% teste (avaliação)
- **Estratificação**: Mantém proporção do target em ambos os conjuntos
- **Seed**: `np.random.seed(42)` para reprodutibilidade

### 5.2 Resultados Esperados

O Thompson Sampling deve demonstrar:
1. **CTR superior** ao baseline (aprende por segmento)
2. **Regret sublinear** (converge ao longo do tempo)
3. **Distribuição não-uniforme** de braços por segmento (personalização efetiva)

### 5.3 Golden Set

5 perfis representativos são testados para validar a coerência das recomendações:

| # | Perfil | Segmento | Oferta Esperada |
|---|--------|----------|-----------------|
| 1 | Jovem, estudante, primeiro contato | jovem | Empréstimo (braço 2) |
| 2 | Adulto, gerente, 1 contato | alta_qualificacao | Premium (braço 1) |
| 3 | Aposentado, 4 contatos | senior | Padrão (braço 0) |
| 4 | Desempregado, 5 contatos | padrao | Controle (braço 3) |
| 5 | Técnico, sucesso anterior | historico_positivo | Padrão/Premium (braço 0/1) |

---

## 6. Serviço Demonstrável

### 6.1 Interface

Duas funções simulam o comportamento de uma API REST:

#### `recomendar_oferta(cliente: dict) -> dict`

**Input:**
```python
{
    'age': 35,
    'job': 'technician',
    'campaign': 1,
    'sucesso_anterior': 1
}
```

**Output:**
```python
{
    'oferta_recomendada': 'Deposito a prazo (padrao)',
    'braco_selecionado': 0,
    'segmento_cliente': 'historico_positivo',
    'probabilidade_conversao': 0.7823,
    'confianca': 'alta',
    'requer_revisao_humana': False
}
```

#### `registrar_feedback(segmento, braco, converteu) -> dict`

Atualiza os priors do modelo com o resultado real da interação. Simula o loop de feedback em produção.

### 6.2 Nível de Confiança

| Nível | Critério | Ação |
|-------|----------|------|
| Alta | > 50 observações no segmento/braço | Decisão automática |
| Média | 20-50 observações | Decisão automática com monitoramento |
| Baixa | < 20 observações | Flag para revisão |

### 6.3 Humano no Loop

Quando `probabilidade_conversao < 0.1`, o campo `requer_revisao_humana = True` sinaliza que a decisão deve ser revisada por um operador antes de executar.

---

## 7. MLflow Tracking

### 7.1 Configuração

```python
mlflow.set_tracking_uri('file:///kaggle/working/mlruns')
mlflow.set_experiment('datathon-bandit')
```

### 7.2 Experimentos Registrados

| Run | Parâmetros | Métricas |
|-----|-----------|----------|
| `baseline_regra_fixa` | algoritmo, melhor_braco_fixo, n_bracos | ctr, reward_acumulado, regret_acumulado, regret_medio |
| `thompson_sampling_contextual` | algoritmo, contextual, n_bracos, n_segmentos, prior_alpha, prior_beta | ctr, reward_acumulado, regret_acumulado, regret_medio, ganho_vs_baseline_pct |

---

## 8. Arquitetura Cloud (AWS)

### 8.1 Diagrama

```
┌─────────┐     ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Cliente │────>│ API Gateway │────>│ ECS/Fargate      │────>│ Thompson        │
│ (App)   │<────│             │<────│ (FastAPI)        │     │ Sampling Model  │
└─────────┘     └─────────────┘     └──────────────────┘     └────────┬────────┘
                                            │                          │
                                            v                          v
                                     ┌─────────────┐          ┌──────────────┐
                                     │ S3          │          │ Feature      │
                                     │ (logs)      │          │ Store        │
                                     └──────┬──────┘          └──────────────┘
                                            │
                                            v
                                     ┌─────────────────┐
                                     │ Step Functions  │
                                     │ (retreino)      │
                                     └────────┬────────┘
                                              │
                                              v
                                     ┌─────────────────┐
                                     │ CloudWatch      │
                                     │ (monitoramento) │
                                     └─────────────────┘
```

### 8.2 Componentes

| Serviço | Função | Justificativa |
|---------|--------|---------------|
| **API Gateway** | Exposição da API REST | Gerenciamento de throttling, autenticação e versionamento |
| **ECS + Fargate** | Container da API (FastAPI) | Serverless containers, escala automática, sem gerenciar EC2 |
| **S3** | Armazenamento de logs de decisão e artefatos | Durável, barato, integra com todos os serviços |
| **Step Functions** | Orquestração de retreino | Workflow visual, retry automático, baixo custo |
| **CloudWatch** | Métricas e alarmes | Latência, erros, drift de modelo |
| **SageMaker Feature Store** | Features centralizadas | Consistência entre treino e inferência |
| **MLflow (EC2)** | Tracking de experimentos | Comparação de versões, reprodutibilidade |

### 8.3 Fluxo de Dados

1. Cliente faz requisição via app/web
2. API Gateway roteia para ECS (FastAPI)
3. Serviço consulta Feature Store para enriquecer contexto
4. Thompson Sampling seleciona braço e retorna recomendação
5. Log de decisão gravado no S3
6. Feedback (conversão ou não) chega via endpoint separado
7. Step Functions executa retreino periódico (diário) usando logs acumulados
8. CloudWatch monitora drift e alerta se métricas degradarem

---

## 9. Governança e Ética

### 9.1 LGPD

| Princípio | Implementação |
|-----------|---------------|
| **Base legal** | Legítimo interesse (Art. 10) para otimização de campanhas |
| **Finalidade** | Recomendar ofertas personalizadas, minimizando contatos desnecessários |
| **Minimização** | Sem identificadores pessoais, gênero, raça ou renda direta |
| **Retenção** | Logs de decisão retidos 12 meses para auditoria, depois anonimizados |
| **Transparência** | Cliente pode solicitar explicação da decisão |

### 9.2 Fairness

- Segmentação baseada em comportamento (campaign, poutcome), não em atributos sensíveis
- Monitoramento da distribuição de ofertas por segmento para detectar viés emergente
- Braço "Controle" garante que nenhum segmento é sobre-contatado

### 9.3 Humano no Loop

- Decisões com probabilidade < 0.1 são sinalizadas para revisão manual
- Nível de confiança (alta/média/baixa) disponível para operadores
- Dashboard de monitoramento permite intervenção manual em tempo real

---

## 10. Limitações e Trabalhos Futuros

### 10.1 Limitações

1. **Generalização**: Dataset de uma única instituição portuguesa
2. **Simulação**: Braços são simulados, não representam ofertas reais distintas
3. **Estático**: Dados batch, sem captura de mudanças em tempo real
4. **Desbalanceamento**: ~11% positivos podem enviesar aprendizado inicial
5. **Custo**: Sem dados de custo real por contato (reward binário como proxy)
6. **Cold start**: Novos segmentos partem de prior uniforme (incerteza máxima)
7. **Independência**: Assume decisões independentes (sem efeito de saturação)

### 10.2 Melhorias Futuras

- **Streaming**: Integrar com Kinesis/Kafka para atualização em tempo real
- **Contextual Bandits com Deep Learning**: LinUCB ou Neural Thompson Sampling para features contínuas
- **Reward shaping**: Incorporar custo de contato e lifetime value
- **Multi-objetivo**: Otimizar conversão + satisfação + retenção simultaneamente
- **A/B testing integrado**: Shadow mode para validar antes de deploy completo

---

## 11. Referências

- Russo, D. et al. (2018). "A Tutorial on Thompson Sampling". Foundations and Trends in Machine Learning.
- Chapelle, O. & Li, L. (2011). "An Empirical Evaluation of Thompson Sampling". NeurIPS.
- Moro, S. et al. (2014). "A data-driven approach to predict the success of bank telemarketing". Decision Support Systems.
- Dataset: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
