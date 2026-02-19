# PI-GAN 2D (Laplace) com referencia numerica no treino

Ultima revisao: 2026-02-19

Implementacao de Physics-Informed GAN (PI-GAN) para Laplace 2D estacionaria, com:
- hard constraint de Dirichlet;
- dois criticos WGAN-GP (`D1` fisico e `D2` de referencia);
- referencia numerica usada no treino e nas metricas.

A referencia atual e gerada por FDM-SOR em `src/fdm.py`. Se voce usar FVM/FEM externo, a semantica permanece: `T_ref` alimenta `D2` e a avaliacao fisica.

## Visao geral do problema

Dominio e equacao:
- `Omega = [0, LX] x [0, LY]`
- PDE: `nabla^2 T = 0` em `Omega`
- BC: Dirichlet

Parametrizacao com hard constraint:

`T_theta(x,y,z) = g(x,y) + phi(x,y) * N_theta(x,y,z)`

onde:
- `g(x,y)` estende as condicoes de contorno;
- `phi(x,y)=0` na borda (`partial Omega`);
- `N_theta` e a saida da rede.

Perdas (resumo):
- `L_PDE_raw = mean(|nabla^2 T_theta|)`
- `L_adv1 = -E[D1(fake)]`
- `L_adv2 = -E[D2(fake)]`
- `L_G = lambda_pde_dyn*L_PDE_raw + lambda_bc*L_BC + adv_grad_scale*(lambda_adv1_eff*L_adv1 + lambda_adv2_eff*L_adv2)`
- `L_D1`, `L_D2`: WGAN-GP com drift e penalidade de gap

Observacao importante:
- com `hard_constraint_bc=True`, o pipeline zera `lambda_bc` para evitar redundancia.

## Arquitetura e modulos

- `src/models.py`
  - `UNetGenerator2D`
  - `PhysicsDiscriminator2D` (`D1`)
  - `DataDiscriminator2D` (`D2`)
  - `FieldDualDiscriminator`
  - `LaplacianLayer`
- `src/trainer.py`
  - `FieldPIGANTrainer` com WGAN-GP, gradnorm, gate adversarial, controle de criticos e scheduler
- `src/pipeline.py`
  - prepara malha/campos, gera referencia, treina, avalia e salva artefatos
- `src/config.py`
  - `ExperimentConfig` e `SystemConfig`

## Instalacao

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencias opcionais recomendadas:

```powershell
pip install psutil structlog
```

## Execucao

Treino padrao:

```powershell
python main.py
```

Determinismo na CLI:
- `python main.py` ja roda deterministico por padrao (`deterministic_run=True`);
- `python main.py --deterministic` explicita o mesmo comportamento;
- `python main.py --no-deterministic` desativa determinismo estrito;
- `python main.py --deterministic --deterministic-warn-only` nao interrompe em operadores CUDA sem caminho deterministico.

Modo do gerador:

```powershell
python main.py --generator-mode stochastic_pigan --latent-dim 8
python main.py --generator-mode deterministic_adversarial
```

Retomar checkpoint:

```powershell
python main.py --resume-checkpoint runs/checkpoints/checkpoint_epoch_1000.pt
python main.py --resume-checkpoint runs/checkpoints/checkpoint_epoch_1000.pt --no-strict-checkpoint
```

CPU fallback (quando necessario):

```powershell
$env:PIGAN_ALLOW_CPU = "1"
python main.py
```

## Defaults principais (estado atual)

Fonte: `src/config.py` (`ExperimentConfig`/`SystemConfig`).

| Parametro | Default |
|---|---|
| `generator_mode` | `stochastic_pigan` |
| `latent_dim` | `8` |
| `grid_size_x`, `grid_size_y` | `32`, `32` |
| `epochs`, `batch_size` | `4000`, `16` |
| `gen_lr`, `disc_lr` | `1.15e-4`, `8.625e-5` |
| `n_critic`, `disc_update_every` | `1`, `1` |
| `lambda_adv1`, `lambda_adv2` | `5.0e-1`, `2.0e-1` |
| `lambda_pde`, `lambda_bc` | `37.0`, `20.0` |
| `lambda_gp` | `8.0` |
| `max_grad_norm` | `1.85` |
| `adv_warmup_epochs` | `120` |
| `adv_residual_gate_target/min/power` | `0.01`, `0.09`, `1.2` |
| `adv_residual_gate_hysteresis` | `True` |
| `max_critic_gap` | `11.0` |
| `adaptive_lambda_pde` | `True` |
| `lambda_pde_min/max` | `19.0`, `98.0` |
| `pde_corner_sampling_ratio` | `0.10` |
| `pde_corner_band_points` | `2` |
| `precision_refine_enable` | `True` |
| `precision_refine_n_critic` | `3` |
| `precision_refine_lambda_pde_max_scale` | `0.72` |
| `deterministic_run` (SystemConfig) | `True` |

## Optuna e tuning

Busca base (controle adversarial):

```powershell
python scripts/optuna_search.py --trials 40 --epochs 180 --steps-per-epoch 1
```

Preset overnight (RTX 3050):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_optuna_overnight_3050.ps1
```

Preset final-refine (gate mais aberto + `precision_refine_*`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_optuna_final_refine_3050.ps1
```

Preset fixed-best-refine (fixa `best_params` e busca apenas refino final):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_optuna_fixed_best_refine_3050.ps1
```

Busca physics-first (foco primario em residual da PDE):

```powershell
python scripts/tune_optuna_physics_first.py --trials 60 --epochs 160 --steps-per-epoch 3 --use-gpu
```

Varredura ativa em 2 fases (candidatos curados):

```powershell
python scripts/tune_active_pigan.py --search-epochs 40 --confirm-epochs 80 --top-k 3 --fast-profile --use-gpu
```

Treino final de 4000 epocas a partir de `best_trial.json`:

```powershell
python scripts/run_final_4000_best.py `
  --best-trial runs_optuna_physics_first_60t/best_trial.json `
  --runs-dir runs_final_4000_best `
  --epochs 4000 `
  --use-gpu
```

Saidas principais de estudos Optuna:
- `best_trial.json`
- `study_summary.json`
- `trials/trial_XXXX/results/optuna_trial_summary.json`

## Recursos

Em `runs/results/`:
- `metrics.json`
- `training_history.json`
- `adversarial_health.json`
- `baseline_metrics.json` (quando nao resume de checkpoint)
- `training_gain.json` (quando ha baseline pre-treino)
- `temperature_pred.npy`
- `temperature_ref_fdm.npy`
- `pde_residual.npy`
- `l2_relative_vs_fdm.txt`

Em `runs/plots/` (quando habilitado):
- `field_comparison.png`
- `training_curves.png`
- `ensemble_predictions.png`
- `gan_quality_metrics.png`
- `physics_consistency.png`
- `training_history.png`
- `uncertainty_analysis.png`

Em `runs/checkpoints/`:
- `checkpoint_epoch_<N>.pt`

## Metricas para relatorio

Minimas recomendadas:
- campo: `mae`, `rmse`, `relative_l2_error`, `max_error`
- fisica: `pde_residual_mean`, `pde_residual_l2`, `pde_residual_max`
- contorno: `boundary_error`
- adversarial: `adv_gate_end`, `adv_over_pde_end`, `d1_nonzero_ratio`, `d2_nonzero_ratio`

Chaves uteis de `adversarial_health.json`:
- `critics_paused_ratio`
- `critics_pause_flag_ratio`
- `critics_reduced_ratio`
- `disc_update_ratio`
- `disc_updates_mean`

Leitura rapida:

```powershell
Get-Content runs/results/metrics.json
Get-Content runs/results/training_gain.json
Get-Content runs/results/adversarial_health.json
```

## Estrutura atual do projeto

```text
src/
  config.py
  pipeline.py
  trainer.py
  models.py
  evaluation.py
  fdm.py
  utils.py
scripts/
  optuna_common.py
  optuna_search.py
  tune_optuna_physics_first.py
  tune_active_pigan.py
  run_optuna_overnight_3050.ps1
  run_optuna_final_refine_3050.ps1
  run_optuna_fixed_best_refine_3050.ps1
  run_final_4000_best.py
  final_4000_best_config.json
tests/
  test_pipeline_execution.py
  test_checkpoint_resume.py
  test_field_training_modes.py
  test_laplacian_field_operator.py
  test_cpu_fallback.py
main.py
orphaned/  (codigo e docs legados, fora do fluxo principal)
```

## Testes

```powershell
python -m pytest -q
```

Cobertura atual:
- fluxo end-to-end em CPU;
- resume de checkpoint;
- modos de treino (sem adversarial / sem PDE);
- operador laplaciano discreto;
- fallback CPU.

## Limitacoes

- foco em Laplace 2D estacionaria;
- referencia interna atual e FDM-SOR;
- qualidade final depende de calibracao conjunta de `lambda_pde`, gate adversarial e dinamica dos criticos.

## Referencias

- Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-Informed Neural Networks.
- Goodfellow, I. et al. (2014). Generative Adversarial Nets.
- Arjovsky, M., Chintala, S., Bottou, L. (2017). Wasserstein GAN.
- Gulrajani, I. et al. (2017). Improved Training of Wasserstein GANs (WGAN-GP).

## Citacao (modelo)

```bibtex
@misc{pigan_laplace_2d,
  title        = {PI-GAN 2D para Laplace com referencia numerica},
  author       = {Autores do Projeto},
  year         = {2026},
  howpublished = {\url{https://github.com/Josemar437/pigan_plate2D_laplace-.git}},
  note         = 
}
```
