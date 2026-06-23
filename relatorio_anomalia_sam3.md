# Relatório Técnico: Resolução de Anomalia de Avaliação do SAM3 Zero-Shot (BamaPig2D e FaroPigSeg)

Este relatório detalha a identificação, a causa raiz e a resolução definitiva de uma anomalia observada nos testes dos datasets **BamaPig2D** e **FaroPigSeg**, onde a performance de detecção zero-shot do modelo professor (SAM3) parecia significativamente inferior à dos modelos alunos (YOLOv8) treinados a partir dele.

---

## 1. A Anomalia Identificada
Ao avaliar o desempenho do **SAM3 Zero-Shot** contra anotações humanas utilizando métricas padrão COCO (mAP), os resultados obtidos inicialmente foram discrepantes:

* **No BamaPig2D:** O SAM3 Zero-shot registrava **73,7% mAP**, enquanto o YOLOv8m (treinado em suas pseudo-anotações) alcançava **86,0% mAP**.
* **No FaroPigSeg:** O SAM3 Zero-shot registrava apenas **45,6% mAP**, enquanto o YOLOv8m alcançava **84,8% mAP**.

> [!WARNING]
> **A incoerência científica:** Embora modelos alunos possam superar o professor em cenários ruidosos devido ao efeito de *denoising* (regularização e viés indutivo de redes especializadas como YOLO), a margem de diferença — especialmente de quase **40% de mAP** no dataset Faro — indicava que o baseline do SAM3 estava severamente subavaliado por uma falha técnica.

---

## 2. Análise da Causa Raiz: O Bug de Pontuação de Confiança (Scores)
A biblioteca oficial de métricas do COCO (`pycocotools`) calcula a Precisão Média (mAP) gerando curvas de Precisão-Revocação (PR). Para traçar essa curva de maneira ideal, a biblioteca precisa **ordenar as predições por pontuação de confiança (score)**. Dessa forma, ela avalia a precisão de caixas muito confiantes primeiro e filtra falsos positivos com scores baixos.

O problema ocorreu devido a uma inconsistência de formatos na extração das inferências do SAM3:

1. **PigLife (Funcionando Corretamente):** 
   A inferência do SAM3 no PigLife era lida a partir de um arquivo de predições plano (`teacher/predictions.json`), o qual continha os scores de confiança reais de cada detecção (ex: `"score": 0.9167`).

2. **Bama e Faro (Com Bug):** 
   Para os datasets Bama e Faro, o script [calculate_metrics.py](file:///hd1/marcos/repos/sam3-pseudo-label-pig-yolo/reports/scripts/calculate_metrics.py) avaliava o zero-shot carregando diretamente o arquivo COCO de anotações (`instances_test.json`) gerado por [teacher/label.py](file:///hd1/marcos/repos/sam3-pseudo-label-pig-yolo/teacher/label.py).
   * O formato COCO padrão de anotações de dataset **não possui** por padrão um campo para armazenar a confiança das inferências.
   * O script `teacher/label.py` gerava o arquivo omitindo o valor de confiança (`score`) calculado pelo SAM3.
   * Ao processar o JSON, o script de métricas tentava recuperar o score de cada detecção. Na ausência deste campo, ele definia o score de **todas** as caixas de forma genérica como **`1.0`** (através da linha `"score": ann.get("score", 1.0)`).

### Impacto na Métrica (COCOeval)
Com todas as detecções do SAM3 configuradas com score de `1.0`:
* A biblioteca `pycocotools` não conseguia ordenar as caixas por ordem de confiança.
* Qualquer detecção ruidosa ou falso positivo gerado pelo modelo (mesmo com confiança baixa de inferência original, como `0.41`) ganhava peso máximo (`1.0`) na avaliação.
* Isso impedia o avaliador de descartar predições de baixa confiança ao calcular a precisão, penalizando bruscamente a curva Precision-Recall e abaixando artificialmente os valores de mAP.

---

## 3. Resolução Estrutural e Arquitetural
Para resolver a anomalia de forma definitiva e manter o repositório limpo, o projeto foi refatorado para separar as responsabilidades de **anotação de treino** e **predição de teste** em scripts independentes:

1. **Geração de Rótulos de Treino (`teacher/label.py`):**
   Gera o dataset de treino e validação no formato estrito de anotações COCO (`instances_*.json`), **sem** conter chaves proprietárias como `"score"`. Isso garante a padronização oficial dos rótulos.
   
2. **Geração de Predições de Teste (`teacher/predict.py`):**
   Criamos um script dedicado para rodar inferência de teste e gerar um arquivo de predições COCO plano (`predictions_test.json`) contendo as confianças reais de cada caixa (`score`).
   
3. **Task Runner (`tasks.py`) e Nomenclaturas:**
   * A tarefa de teste do aluno foi renomeada de `evaluate` para `predict` (que chama `student/predict.py`).
   * Adicionamos a nova tarefa `predict-teacher` (que chama `teacher/predict.py` para gerar as predições do SAM3 zero-shot).
   * O pipeline `all` foi atualizado para automatizar a geração desses arquivos.

4. **Cálculo de Métricas (`reports/scripts/calculate_metrics.py`):**
   O script agora aponta para o novo arquivo de predições (`predictions_test.json`) para avaliar o baseline zero-shot do SAM3, unificando a leitura de predições reais de todos os modelos e datasets.

---

## 4. Resultados Finais e Conclusões
Com a correção aplicada, as métricas de baseline zero-shot do SAM3 subiram consideravelmente e agora refletem a realidade:

### Tabela Comparativa de Métricas (mAP)
| Dataset | Modelo | mAP [50-95] (Antes) | mAP [50-95] (Corrigido) | $AP_{50}$ (Antes) | $AP_{50}$ (Corrigido) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BamaPig2D** | SAM3 Zero-Shot | 73.7% | **84.0%** | 87.7% | **96.6%** |
| | YOLOv8m (Student) | 86.0% | **86.0%** | 97.8% | **97.8%** |
| **FaroPigSeg** | SAM3 Zero-Shot | 45.6% | **82.6%** | 51.9% | **90.5%** |
| | YOLOv8m (Student) | 84.8% | **84.8%** | 91.2% | **91.2%** |

### Conclusões Científicas Atualizadas:
1. **O Potencial de Anotação do SAM3:** O SAM3 possui um altíssimo potencial de anotação zero-shot, ultrapassando os **90% de mAP@50** tanto no Bama quanto no Faro.
2. **Comparativo Professor vs. Aluno:** O aluno (YOLOv8m) performa muito próximo do professor em ambos os cenários (86.0% vs 84.0% no Bama; 84.8% vs 82.6% no Faro). Essa ligeira superioridade do aluno treinado sobre o baseline zero-shot do professor se justifica cientificamente pelo **viés indutivo da rede YOLOv8** e pelo **efeito de denoising / especialização de domínio** ao ser treinado no conjunto completo de imagens daquele farm/ambiente específico.
