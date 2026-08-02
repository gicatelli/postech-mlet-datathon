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
- **Registros**: ~45.211
- **Features**: 16 atributos (demográficos, financeiros e de campanha)
- **Target**: `y` (subscreveu depósito a prazo: yes/no)
- **Coluna descartada**: `duration` (vazamento temporal — só conhecida após o contato)

---

## Como Executar

### No Kaggle (recomendado para avaliadores):
1. Acesse o notebook no Kaggle: [link do notebook]
2. Clique em **"Copy & Edit"**
3. Clique em **"Run All"**
4. Todas as seções executam sequencialmente sem necessidade de configuração

### Localmente:
```bash
git clone https://github.com/gicatelli/postech-mlet-datathon.git
cd postech-mlet-datathon
pip install -r requirements.txt
jupyter notebook notebook/datathon-7mlet.ipynb
```

---

## Algoritmos

### Baseline (Regra Fixa)
Política determinística que sempre recomenda a oferta com maior taxa de conversão histórica (melhor braço fixo). Serve como referência comparativa.

### Thompson Sampling (Algoritmo Adaptativo)
Abordagem bayesiana que modela a incerteza sobre a taxa de conversão de cada braço usando distribuição Beta. A cada decisão, amostra valores de cada distribuição e seleciona o braço com maior amostra. Com segmentação contextual por perfil do cliente.

**Braços (ofertas)**:
| Braço | Oferta |
|-------|--------|
| 1 | Depósito a prazo (padrão) |
| 2 | Depósito com taxa premium |
| 3 | Produto alternativo (empréstimo) |
| 4 | Controle (não contatar) |

---

## Resultados

| Métrica | Baseline | Thompson Sampling |
|---------|----------|-------------------|
| Taxa de Conversão | - | - |
| Regret Acumulado | - | - |
| Reward Acumulado | - | - |

*Tabela será preenchida após execução do notebook.*

---

## Golden Set (5 Casos de Teste)

| # | Perfil | Oferta Recomendada | Justificativa |
|---|--------|--------------------|---------------|
| 1 | Jovem, estudante, baixo saldo | - | - |
| 2 | Adulto, gerente, alto saldo | - | - |
| 3 | Aposentado, saldo médio | - | - |
| 4 | Desempregado, sem contato anterior | - | - |
| 5 | Técnico, contato anterior bem-sucedido | - | - |

*Tabela será preenchida após execução do notebook.*

---

## Arquitetura Cloud (AWS)

Para colocar este projeto em produção, utilizaríamos os seguintes serviços AWS:

**Amazon S3** para armazenamento dos dados brutos e processados, junto com logs de decisão do modelo. **Amazon ECS com Fargate** para hospedar a API de recomendação em container Docker, exposta via **API Gateway** para receber requisições dos canais digitais. O modelo Thompson Sampling seria atualizado periodicamente via **AWS Step Functions** orquestrando um pipeline de retreino que lê os feedbacks acumulados no S3.

Para monitoramento, **CloudWatch** coleta métricas de latência, taxa de erro e drift do modelo. **SageMaker Feature Store** armazena as features dos clientes de forma centralizada, e os experimentos de ML são rastreados via **MLflow hospedado em EC2** ou **SageMaker Experiments**.

---

## MLflow

Experimentos registrados:
- **baseline_regra_fixa**: métricas do baseline determinístico
- **thompson_sampling_v1**: Thompson Sampling sem contexto
- **thompson_sampling_contextual**: Thompson Sampling com segmentação por features

Parâmetros e métricas rastreados via MLflow inline no notebook Kaggle.

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
