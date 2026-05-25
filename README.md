# PI-GAN 2D Laplace - contrato tecnico da versao atual

Ultima revisao: 2026-05-24.

Este repositorio treina uma PI-GAN para a equacao de Laplace 2D estacionaria em placa retangular. O caminho de execucao oficial e:

```text
main.py (ou train.py) -> src/pipeline.py -> src/model/* + src/losses/* + src/trainer.py -> runs/
```

O treino usa uma solucao FDM-SOR interna como referencia numerica. Essa referencia alimenta o discriminador de dados (`D2`) e as metricas finais; ela nao substitui a penalizacao explicita do residuo `nabla^2 T`.

Nota: na configuracao atual, `D2` deve ser lido como regularizador auxiliar supervisionado por FDM, nao como prova de aprendizado adversarial puro. A consistencia fisica vem principalmente do residual discreto e do discriminador fisico (`D1`). O refinamento fisico final existe, mas fica desligado por padrao e deve ser tratado como ablation declarada.

## Problema Implementado

Dominio:

```text
Omega = [0, LX] x [0, LY]
nabla^2 T = 0 no interior
T = g na fronteira de Dirichlet
```

O gerador usa restricao forte de contorno:

```text
T_theta(x, y, z) = g(x, y) + phi(x, y) * N_theta(x, y, z)
```

`src/utils.py` constroi `g` como extensao linear das bordas laterais de Dirichlet e constroi `phi` com valor zero nas laterais. Quando `hard_constraint_bc=True`, `lambda_bc=0.0` porque a condicao de Dirichlet lateral ja esta embutida na parametrizacao. As bordas inferior e superior usam Neumann homogeneo, imposto por `lambda_neumann` no treino e pela referencia FDM mista.

## Modulos Ativos No Treino

`src/pipeline.py`

- monta a malha cartesiana;
- cria `base_field`, `phi_mask`, mascaras de interior/fronteira e coordenadas;
- resolve a referencia FDM-SOR com `src/fdm.py`;
- instancia modelos e `FieldPIGANTrainer`;
- salva metricas, arrays, figuras e checkpoints.

`src/model/generator.py`

- `UNetGenerator2D`: `T0 = g + phi * N0` (nomes `g`, `phi`, `N0`, `T0`); coordenadas e latente opcionais;
- `HardConstraintLayer`: camada final explicita da imposicao forte de Dirichlet.

`src/model/operators.py`

- `LaplacianOperator` (`laplacian_kernel` em buffer): stencil 3x3 centrado, residuo apenas no interior;
- `build_g_field`, `build_phi_mask`: wrappers de `src/utils.py`.

`src/model/discriminator.py`

- `PhysicsDiscriminator2D` (`D1`): critica `R0` (residuo Laplaciano);
- `DataDiscriminator2D` (`D2`): critica pares `[T0, T_ref]`;
- `FieldDualDiscriminator`: agrupa `D1` e `D2`.

`src/losses/physical.py` e `src/losses/adversarial.py`

- `loss_pde`, `neumann_loss`, `discriminator_loss_wgan_gp`, `generator_adversarial_loss` (WGAN-GP + drift + gap).

`src/models.py` reexporta os modulos acima para compatibilidade.

`src/trainer.py`

- calcula `loss_pde` / `L_PDE_raw = mean(abs(R0))` com `R0 = Laplacian(T0)` no interior;
- treina `D1` e `D2` com WGAN-GP, drift e penalidade de gap;
- atualiza `lambda_pde` dinamicamente por lei log-linear ancorada em `config.lambda_pde`;
- aplica balanceamento GradNorm do termo adversarial;
- controla warmup, gate adversarial, pausa de criticos, queda de learning rate por drift e checkpoint/resume.
- opcionalmente executa uma fase final `refine_physics()` que congela discriminadores e reduz diretamente o residual discreto.

## Revisao Critica Da Dinamica Adversarial

Pontos importantes para leitura de resultados:

- `D1` era quase irrelevante quando `lambda_adv1 << lambda_adv2`. A configuracao padrao foi revisada para favorecer `D1` (`lambda_adv1=5.0e-2`) e deixar `D2` como auxiliar (`lambda_adv2=2.0e-2`).
- `D2` compara `[pred, ref]` contra `[ref, ref]`; portanto, ele e aprendizado supervisionado por referencia FDM em roupagem adversarial. Isso e util como regularizacao, mas deve ser declarado assim em texto tecnico/artigo.
- WGAN-GP com `n_critic=1` tende a subtreinar os criticos. O padrao agora usa `n_critic=3`, com `precision_refine_n_critic=3`.
- `lambda_bc` nao participa do treino padrao porque `hard_constraint_bc=True` garante Dirichlet exatamente. O default foi zerado para evitar parametro morto.
- `physics_refine_enable=False` por padrao. Runs com refinamento fisico final devem reportar `physics_refine_steps` e comparar contra uma run sem refinamento; caso contrario, a atribuicao de credito ao framework GAN fica comprometida.
- O gate adversarial deve ser auditado em `adversarial_health.json`: `adv_gate_ever_opened`, `adv_gate_first_open_epoch`, `adv_gate_open_ratio`, `adv_gate_closed_ratio`, `d1_nonzero_ratio` e `d2_nonzero_ratio` dizem se D1/D2 participaram de fato.
- O `best_trial.json` antigo de `runs_optuna_rebalance_32t` nao documenta `steps_per_epoch`; como `scripts/optunaSearch.py` usa default `1`, o launcher `start.py` assume `search_steps_per_epoch=1` e bloqueia mudancas sem `--allow-steps-mismatch`.
- Os modulos refatorados em `src/physics/` e `src/training/` ainda nao sao o caminho canonico da loss no runtime. Eles devem ser citados como componentes auxiliares/testados, nao como origem ativa das perdas de treino.

## Modulos Refatorados Ainda Nao Integrados Ao Runtime

Os seguintes componentes existem, possuem testes unitarios e sao instanciados no `FieldPIGANTrainer`, mas a logica efetiva do treino ainda esta nos metodos internos de `src/trainer.py`:

```text
src/physics/pdeResidual.py        PDEResidualComputer (metricas/testes)
src/physics/domainMetrics.py      DomainMetricsComputer
src/training/lossFunctions.py     shim -> src/losses/*
src/training/adaptiveSchemes.py   AdaptiveLambdaPDE, GradNormBalancer,
                                  StagnationDetector, DivergenceDetector
```

Eles devem ser tratados como biblioteca preparada para uma proxima etapa de integracao, nao como caminho ativo do treinamento atual. Para auditoria, o contrato oficial de runtime permanece `src/pipeline.py` + `src/trainer.py` + `src/models.py`.

## Execucao Local

Use a pasta canonica do projeto:

```powershell
cd D:\TCC\PyTorch\2D_PlacaPlana\pigans_plate2D_lapace_d1
```

Se voce estiver na pasta antiga `pigans_plate2D_lapace`, o `start.py` dessa pasta deve redirecionar para a pasta canonica `_d1`. Ainda assim, para desenvolvimento, testes e edicao de arquivos, prefira trabalhar diretamente em `pigans_plate2D_lapace_d1`.

Instalacao:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Treino com a configuracao padrao:

```powershell
python main.py
```

Treino final rebalanceado usando o melhor trial Optuna versionado em `runs_optuna_rebalance_32t`:

```powershell
python start.py
```

Treino final usando GPU explicitamente:

```powershell
python start.py --use-gpu
```

Forcar CPU quando CUDA nao estiver disponivel:

```powershell
$env:PIGAN_ALLOW_CPU = "1"
python main.py
```

Retomar checkpoint:

```powershell
python main.py --resume-checkpoint runs/checkpoints/checkpoint_epoch_1000.pt
```

Retomar checkpoint aceitando campos ausentes/novos:

```powershell
python main.py --resume-checkpoint runs/checkpoints/checkpoint_epoch_1000.pt --no-strict-checkpoint
```

Modos de gerador expostos pela CLI:

```powershell
python main.py --generator-mode stochastic_pigan --latent-dim 8
python main.py --generator-mode deterministic_adversarial
```

Determinismo:

```powershell
python main.py --deterministic
python main.py --no-deterministic
python main.py --deterministic --deterministic-warn-only
```

## Configuracao Padrao Relevante

Fonte: `ExperimentConfig` e `SystemConfig` em `src/config.py`.

| Campo | Valor atual |
|---|---:|
| `generator_mode` | `stochastic_pigan` |
| `latent_dim` | `8` |
| `grid_size_x`, `grid_size_y` | `32`, `32` |
| `epochs`, `batch_size` | `4000`, `16` |
| `gen_lr`, `disc_lr` | `1.15e-4`, `8.625e-5` |
| `n_critic`, `disc_update_every` | `3`, `1` |
| `lambda_pde` | `37.0` |
| `lambda_bc` | `0.0` com hard constraint |
| `lambda_adv1`, `lambda_adv2` | `5.0e-2`, `2.0e-2` |
| `lambda_gp` | `8.0` |
| `max_grad_norm` | `1.85` |
| `adaptive_lambda_pde` | `True` |
| `lambda_pde_min`, `lambda_pde_max` | `19.0`, `98.0` |
| `residual_scale_reference` | `1.0e-2` |
| `lambda_pde_growth_exponent` | `0.60` |
| `gradnorm_target_adv_to_pde` | `0.35` |
| `adv_warmup_epochs` | `120` |
| `adv_residual_gate_target` | `0.01` |
| `precision_refine_enable` | `True` |
| `precision_refine_n_critic` | `3` |
| `physics_refine_enable` | `False` |
| `physics_refine_steps` | `0` |
| `deterministic_run` | `True` |

## Saidas Geradas

Diretorio `runs/results/`:

```text
metrics.json
training_history.json
adversarial_health.json
baseline_metrics.json
training_gain.json
temperature_pred.npy
temperature_ref_fdm.npy
pde_residual.npy
l2_relative_vs_fdm.txt
```

Diretorio `runs/plots/`, quando plotagem esta habilitada:

```text
field_comparison.png
training_curves.png
ensemble_predictions.png
gan_quality_metrics.png
physics_consistency.png
training_history.png
uncertainty_analysis.png
```

Diretorio `runs/checkpoints/`:

```text
checkpoint_epoch_<N>.pt
```

Leitura rapida apos treino:

```powershell
Get-Content runs/results/metrics.json
Get-Content runs/results/training_gain.json
Get-Content runs/results/adversarial_health.json
```

## Scripts De Busca

Optuna principal:

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

Rodar testes usando um `basetemp` local evita falhas de permissao no Temp do Windows:

```powershell
python -m pytest -q --basetemp .pytest_tmp
```

Auditoria de estrutura e nomes:

```powershell
python scripts/auditProject.py
```

O script de auditoria valida a arvore oficial, nomes `lowerCamelCase` para scripts/modulos, nomes `testCamelCase` para testes, ausencia de nomes legados `snake_case` e ausencia de diretorios antigos fora do contrato atual.

## Estrutura Contratada

```text
src/
  config.py
  pipeline.py
  trainer.py
  models.py
  evaluation.py
  fdm.py
  utils.py
  physics/
    domainMetrics.py
    pdeResidual.py
  training/
    adaptiveSchemes.py
    lossFunctions.py
scripts/
  auditProject.py
  optunaCommon.py
  optunaSearch.py
  tuneOptunaPhysicsFirst.py
  tuneActivePigan.py
  runFinalBest.py
tests/
  testCheckpointCompatibility.py
  testCheckpointResume.py
  testCpuFallback.py
  testFieldTrainingModes.py
  testInferencePoints.py
  testLaplacianFieldOperator.py
  testMainConfig.py
  testPipelineExecution.py
  testTrainingModules.py
main.py
inference.py
```
