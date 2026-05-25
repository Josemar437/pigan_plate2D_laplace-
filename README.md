# PI-GAN para Laplace 2D em Placa Plana

Implementação em PyTorch de uma Physics-Informed GAN para resolver a equação de Laplace
estacionária em uma placa retangular 2D. O projeto combina restrição física explícita,
referência numérica FDM-SOR e treinamento adversarial com dois discriminadores para avaliar
campo térmico e consistência do residual.

O repositório foi organizado para experimentos reprodutíveis de TCC/pesquisa: o treino grava
resultados, métricas, figuras e checkpoints em `runs/`, salva histórico completo, compara a
solução neural contra uma referência FDM e expõe scripts para busca de hiperparâmetros com
Optuna.

## Problema Físico

O domínio implementado é:

```text
Omega = [0, LX] x [0, LY]
nabla² T = 0 no interior
T(0, y) = T_LEFT
T(LX, y) = T_RIGHT
dT/dn = 0 nas bordas inferior e superior
```

Por padrão:

```text
LX = LY = 1.0
T_LEFT = 200.0
T_RIGHT = 100.0
malha = 32 x 32
```

A condição de Dirichlet lateral é imposta por restrição forte:

```text
T_theta(x, y, z) = g(x, y) + phi(x, y) * N_theta(x, y, z)
```

Isso faz `T_theta` respeitar exatamente as laterais de Dirichlet quando
`hard_constraint_bc=True`. Por isso `lambda_bc=0.0` no treino padrão. As bordas inferior e
superior são tratadas como Neumann homogêneo via `lambda_neumann`.

## Como O Método Está Montado

O caminho principal de execução é:

```text
main.py / train.py
  -> src/pipeline.py
  -> src/model/*
  -> src/losses/*
  -> src/trainer.py
  -> runs/
```

Componentes centrais:

| Componente | Papel |
|---|---|
| `src/fdm.py` | Resolve uma referência FDM-SOR para comparação e supervisão auxiliar. |
| `src/model/generator.py` | Define o gerador U-Net 2D com parametrização física forte. |
| `src/model/operators.py` | Calcula o Laplaciano discreto no interior da malha. |
| `src/model/discriminator.py` | Implementa `D1` físico e `D2` de referência. |
| `src/losses/physical.py` | Perdas físicas: residual PDE e Neumann. |
| `src/losses/adversarial.py` | Perdas WGAN-GP, drift e controle de gap dos críticos. |
| `src/trainer.py` | Orquestra treino, críticos, GradNorm, checkpoint e refinamento opcional. |
| `src/pipeline.py` | Monta malha, modelos, referência, treino, avaliação e arquivos de saída. |

> [!IMPORTANT]
> `D2` usa a referência FDM como regularizador supervisionado em formato adversarial. Ele ajuda
> a estabilizar o campo, mas não deve ser descrito como evidência isolada de aprendizado
> adversarial puro. A consistência física principal vem do residual Laplaciano, de `D1` e da
> restrição forte de contorno.

## Instalação

Use Python com PyTorch, NumPy, SciPy, Matplotlib e Optuna:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se CUDA não estiver disponível e você quiser permitir execução em CPU:

```powershell
$env:PIGAN_ALLOW_CPU = "1"
```

## Treinamento Rápido

Treino padrão definido por `ExperimentConfig`:

```powershell
python main.py
```

Entrada alternativa equivalente:

```powershell
python train.py
```

Treino final usando o launcher reprodutível:

```powershell
python start.py
```

Forçar GPU no launcher final:

```powershell
python start.py --use-gpu
```

Retomar um checkpoint:

```powershell
python main.py --resume-checkpoint runs/checkpoints/checkpoint_epoch_1000.pt
```

Permitir retomada com diferenças pequenas de arquitetura/configuração:

```powershell
python main.py --resume-checkpoint runs/checkpoints/checkpoint_epoch_1000.pt --no-strict-checkpoint
```

Selecionar modo do gerador:

```powershell
python main.py --generator-mode stochastic_pigan --latent-dim 8
python main.py --generator-mode deterministic_adversarial
```

Controlar determinismo:

```powershell
python main.py --deterministic
python main.py --no-deterministic
python main.py --deterministic --deterministic-warn-only
```

## Configuração Atual

Os valores padrão vivem em `src/config.py`.

| Parâmetro | Valor |
|---|---:|
| `generator_mode` | `stochastic_pigan` |
| `latent_dim` | `8` |
| `grid_size_x`, `grid_size_y` | `32`, `32` |
| `epochs` | `4000` |
| `steps_per_epoch` | `1` |
| `batch_size` | `16` |
| `gen_lr` | `1.15e-4` |
| `disc_lr` | `8.625e-5` |
| `n_critic` | `3` |
| `lambda_pde` | `37.0` |
| `lambda_pde_min`, `lambda_pde_max` | `50.0`, `500.0` |
| `lambda_adv1`, `lambda_adv2` | `5.0e-2`, `2.0e-2` |
| `lambda_neumann` | `10.0` |
| `lambda_gp` | `8.0` |
| `hard_constraint_bc` | `True` |
| `physics_refine_enable` | `False` |
| `deterministic_run` | `True` |

Também é possível passar um JSON customizado:

```powershell
python main.py --config caminho\para\config.json
```

Somente chaves reconhecidas por `ExperimentConfig` e `SystemConfig` são aplicadas.

## Saídas Geradas

Depois do treino, o pipeline grava resultados em `runs/`:

```text
runs/
  results/
    metrics.json
    training_history.json
    adversarial_health.json
    baseline_metrics.json
    training_gain.json
    temperature_pred.npy
    temperature_ref_fdm.npy
    pde_residual.npy
    l2_relative_vs_fdm.txt
  plots/
    field_comparison.png
    training_curves.png
    ensemble_predictions.png
    gan_quality_metrics.png
    physics_consistency.png
    training_history.png
    uncertainty_analysis.png
  checkpoints/
    checkpoint_epoch_<N>.pt
  logs/
    training.log
```

Leitura rápida dos principais indicadores:

```powershell
Get-Content runs/results/metrics.json
Get-Content runs/results/training_gain.json
Get-Content runs/results/adversarial_health.json
```

Métricas importantes:

| Métrica | Interpretação |
|---|---|
| `relative_l2_error_vs_fdm` | Erro relativo do campo neural contra a solução FDM. |
| `pde_residual_mean` | Média do residual discreto `nabla²T` no interior. |
| `pde_residual_max` | Pior residual encontrado na malha. |
| `boundary_error` | Erro nas condições de contorno avaliadas. |
| `adv_gate_open_ratio` | Fração do treino em que o gate adversarial atuou. |
| `d1_nonzero_ratio`, `d2_nonzero_ratio` | Indicam participação efetiva dos discriminadores. |

## Busca De Hiperparâmetros

Busca principal com Optuna:

```powershell
python scripts/optunaSearch.py --trials 40 --epochs 180 --steps-per-epoch 1
```

Busca physics-first:

```powershell
python scripts/tuneOptunaPhysicsFirst.py --trials 60 --epochs 160 --steps-per-epoch 3 --use-gpu
```

Varredura ativa em duas fases:

```powershell
python scripts/tuneActivePigan.py --search-epochs 40 --confirm-epochs 80 --top-k 3 --fast-profile --use-gpu
```

Treino final a partir de um `best_trial.json`:

```powershell
python scripts/runFinalBest.py `
  --best-trial runs_optuna_physics_first_60t/best_trial.json `
  --runs-dir runs_final_4000_best `
  --epochs 4000 `
  --use-gpu
```

Presets PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/runOptunaOvernight3050.ps1
powershell -ExecutionPolicy Bypass -File scripts/runOptunaFinalRefine3050.ps1
powershell -ExecutionPolicy Bypass -File scripts/runOptunaFixedBestRefine3050.ps1
```

## Testes E Auditoria

Rodar a suíte:

```powershell
python -m pytest -q --basetemp .pytest_tmp
```

Auditar estrutura e convenções de nomes:

```powershell
python scripts/auditProject.py
```

A auditoria verifica a árvore oficial, nomes `lowerCamelCase` para scripts e módulos,
testes em padrão `testCamelCase`, ausência de módulos legados em `snake_case` e remoção de
diretórios antigos fora do contrato atual.

## Estrutura Do Projeto

```text
src/
  config.py                  configurações de experimento e sistema
  pipeline.py                execução ponta a ponta
  trainer.py                 laço de treino PI-GAN
  fdm.py                     referência FDM-SOR
  evaluation.py              métricas do campo
  models.py                  reexports de compatibilidade
  utils.py                   malha, contorno e máscaras
  model/
    generator.py
    discriminator.py
    operators.py
  losses/
    physical.py
    adversarial.py
  physics/
    domainMetrics.py
    pdeResidual.py
  training/
    adaptiveSchemes.py
    constants.py
    lossFunctions.py
scripts/
  optunaSearch.py
  tuneOptunaPhysicsFirst.py
  tuneActivePigan.py
  runFinalBest.py
  auditProject.py
tests/
  test*.py
main.py
train.py
start.py
inference.py
```

## Notas Para Interpretação Científica

- A solução FDM é referência numérica e regularizador auxiliar; ela não substitui a perda física.
- `D1` opera sobre o residual Laplaciano e é o discriminador mais diretamente ligado à física.
- `D2` compara pares envolvendo `T_ref`; relate isso como supervisão auxiliar/adversarial.
- `physics_refine_enable=False` por padrão. Quando ativado, reporte como etapa de refinamento ou
  ablação separada.
- Para validar uma execução, confira sempre `metrics.json`, `training_history.json` e
  `adversarial_health.json`, não apenas as figuras.
