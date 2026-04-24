# 🐝 MiteScan AI

MiteScan AI é um módulo de Inteligência Artificial desenvolvido para auxiliar apicultores na identificação de possíveis ameaças às colmeias, como a presença do ácaro varroa e deformações nas asas das abelhas.

Este projeto implementa uma **Rede Neural Convolucional (CNN)** desenvolvida manualmente utilizando PyTorch, sem o uso de modelos prontos como YOLO, com o objetivo de compreender e controlar todo o processo de aprendizado da IA.

---

## 🧠 Objetivo

Identificar, a partir de imagens, se uma abelha está:

- 🟢 Normal  
- 🔴 Com presença de varroa  
- 🟡 Com asas deformadas  

Este módulo complementa o sistema principal do MiteScan, que utiliza sensores IoT (temperatura e umidade) para detectar condições de risco na colmeia.

---

## 🏗️ Estrutura do Projeto

```
mitescan-ai/
│
├── dataset/
│ ├── normal/
│ ├── varroa/
│ └── deformada/
│
├── model/
│ └── cnn.py
│
├── train.py
├── predict.py
├── model.pth
└── README.md
```

---

## ⚙️ Como Funciona

O fluxo da IA é dividido em três partes principais:

### 📥 Treinamento (`train.py`)

- Carrega as imagens do dataset
- Redimensiona e transforma em **tensores (matrizes numéricas)**
- Executa o treinamento da rede neural
- Calcula o erro da previsão
- Ajusta os pesos com **backpropagation**
- Salva o modelo treinado em `model.pth`

---

### 🧠 Arquitetura da Rede (`model/cnn.py`)

Define a estrutura da CNN:

- Camadas convolucionais (extração de características)
- Funções de ativação (ReLU)
- Camadas de pooling
- Camadas totalmente conectadas

A rede aprende padrões como:

- presença do ácaro varroa
- deformações nas asas
- características de abelhas saudáveis

---

### 🔍 Inferência (`predict.py`)

- Recebe uma imagem nova
- Aplica o mesmo pré-processamento do treino
- Carrega o modelo treinado (`model.pth`)
- Retorna a classificação com base nos padrões aprendidos

---

## 🔄 Fluxo do Sistema


Imagem → Tensor → CNN → Probabilidades → Classe

---

## 🧪 Tecnologias Utilizadas

- Python 3.x  
- PyTorch  
- Torchvision  
- Pillow  

---

## ▶️ Como Executar o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/mitescan-ai.git
cd mitescan-ai
```

2. Instale as dependências
```pip install torch torchvision pillow```
3. Prepare o dataset

Organize as imagens da seguinte forma:

```
dataset/
 ├── normal/
 ├── varroa/
 └── deformada/
```

Cada pasta deve conter imagens correspondentes à classe.

4. Treinar o modelo
```python train.py```

Isso irá gerar o arquivo: model.pth

5. Fazer previsão

Coloque uma imagem de teste (ex: teste.jpg) na raiz do projeto e execute:

```python predict.py```

Saída esperada:

Classe: varroa
Confiança: 0.87

--- 

### ⚠️ Observações
O modelo trabalha com probabilidades, não certezas absolutas
Dataset pequeno pode gerar baixa precisão
Mais dados = melhor desempenho
Evitar overfitting aumentando exageradamente o número de épocas

---

### 🚀 Próximos Passos
Aumentar o dataset (mais imagens)
Aplicar técnicas de data augmentation
Melhorar a arquitetura da CNN
Integrar com backend (API)
Conectar com frontend (dashboard React)
🔗 Integração com o Sistema MiteScan

O fluxo completo do sistema será:
```
IoT (temperatura/umidade) → alerta de risco  
→ usuário captura imagem  
→ envio para backend  
→ IA (CNN) analisa  
→ retorno da classificação  
→ exibição no dashboard
```

---

### 📌 Conclusão

Este projeto demonstra a implementação de uma IA de forma manual, permitindo total controle sobre:

- arquitetura da rede neural
- processo de treinamento
- inferência
