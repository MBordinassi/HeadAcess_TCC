# HeadAccess

HeadAccess e um software de acessibilidade que permite controlar o cursor do mouse com movimentos da cabeca usando webcam.

O projeto foi desenvolvido como prototipo de TCC, com arquitetura modular, foco em escalabilidade e separacao clara de responsabilidades.

## Principais funcionalidades

- Captura de video via webcam com OpenCV
- Rastreamento facial com MediaPipe
- Uso do ponto do nariz como referencia de movimento
- Mapeamento proporcional camera -> tela
- Controle de mouse com PyAutoGUI
- Suavizacao de movimento (media movel configuravel)
- Dead zone para reduzir tremores involuntarios
- Calibracao inicial e recalibracao manual (`C`)
- Deteccao de piscadas:
  - piscada esquerda -> clique esquerdo
  - piscada direita -> clique direito
- Overlay de debug com:
  - landmarks faciais
  - centro do rosto
  - vetor de movimento
  - FPS
- Encerramento rapido com `ESC`

## Arquitetura do projeto

```text
headaccess/
├── main.py
├── config.py
├── requirements.txt
├── core/
│   ├── camera.py
│   ├── face_tracker.py
│   ├── movement_processor.py
│   ├── blink_detector.py
│   └── mouse_controller.py
├── utils/
│   ├── smoothing.py
│   ├── calibration.py
│   └── screen_mapping.py
├── ui/
│   └── debug_overlay.py
└── models/
    └── face_landmarker.task
```

## Requisitos

- Windows 10/11
- Python 3.11 ou 3.12 (recomendado)
- Webcam funcional

> Observacao: Python 3.13 pode gerar incompatibilidades de wheels em algumas maquinas.

## Instalacao

No PowerShell, dentro da pasta do projeto:

```powershell
cd "headaccess"
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se a ativacao do ambiente virtual for bloqueada por policy, rode sem ativar:

```powershell
.\venv\Scripts\python.exe main.py
```

## Execucao

```powershell
cd "headaccess"
.\venv\Scripts\python.exe main.py
```

## Controles

- `C`: recalibrar posicao neutra da cabeca
- `ESC`: encerrar o aplicativo

## Como funciona (resumo tecnico)

1. A webcam captura frames continuamente em thread separada.
2. O `FaceTracker` extrai landmarks faciais e o ponto do nariz.
3. O `MovementProcessor` converte deslocamento da cabeca em coordenadas de tela.
4. Um filtro de media movel suaviza o movimento.
5. A dead zone ignora micro-oscilacoes involuntarias.
6. O `BlinkDetector` calcula EAR para detectar piscadas unilaterais.
7. O `MouseController` move o cursor e dispara cliques.

## Configuracao

Ajustes centralizados em `headaccess/config.py`:

- sensibilidade (`sensitivity_x`, `sensitivity_y`)
- janela de suavizacao (`smoothing_window`)
- dead zone (`dead_zone_px`)
- limiar e cooldown de piscada (`ear_threshold`, `cooldown_frames`)
- modo debug e nivel de log

## Troubleshooting

### 1) Erro de ExecutionPolicy no PowerShell

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Ou execute sempre com `venv\Scripts\python.exe` sem ativar.

### 2) `AttributeError: module 'mediapipe' has no attribute 'solutions'`

O projeto ja possui fallback para MediaPipe Tasks quando `mp.solutions` nao existe.

### 3) Erro compilando `numpy` (meson / compiler not found)

Use Python 3.11/3.12 e recrie o ambiente virtual.

## Roadmap (expansao futura)

- Interface grafica para configuracao em tempo real
- Perfis de usuario (sensibilidade/calibracao)
- Gestos extras para duplo clique e scroll
- Suporte a multiplos monitores
- Persistencia de configuracoes em arquivo

## Licenca

Defina a licenca desejada para o repositorio (ex.: MIT).

---

Se este projeto te ajudou, considere deixar uma estrela no GitHub.
