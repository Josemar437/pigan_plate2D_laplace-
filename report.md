# Relatorio Tecnico do Projeto PI-GAN 2D (Laplace)

## 1. Metadados da analise

- Projeto: `pigans_plate2D_lapace`
- Caminho analisado: `d:\TCC\PyTorch\2D_PlacaPlana\pigans_plate2D_lapace`
- Data da analise: `2026-02-24`
- Escopo analisado:
  - Codigo ativo: `main.py`, `src/`, `scripts/`, `tests/`
  - Artefatos experimentais: `runs/`, `report/jscpd-after-3/`
  - Codigo legado: `orphaned/` (fora do fluxo ativo)

## 2. Resumo executivo

Este projeto implementa uma PI-GAN para resolver a equacao de Laplace 2D estacionaria com:

- imposicao forte de Dirichlet (`T = g + phi*N`);
- dois criticos WGAN-GP:
  - `D1` para coerencia fisica (residuo da PDE),
  - `D2` para proximidade com referencia numerica;
- referencia gerada internamente por FDM-SOR.

Resultado principal disponivel em `runs/results` (treino de `2026-02-17`):

- alta aderencia ao campo de referencia:
  - `RMSE = 4.8399e-04`
  - `relative_l2_error_vs_fdm = 3.1648e-06`
- melhoria forte vs baseline nao treinado:
  - `RMSE`: ganho de `128.78x`
  - `pde_residual_mean`: ganho de `34.13x`
- limitacao fisica residual:
  - alvo configurado `1e-3` nao foi atingido
  - final medido: `pde_residual_mean = 9.1884e-02`

## 3. Problema fisico e formulacao

### 3.1 Equacao e dominio

- Dominio: `Omega = [0, LX] x [0, LY]` com defaults `LX = LY = 1.0`
- PDE: `nabla^2 T = 0`
- Contorno: Dirichlet

### 3.2 Parametrizacao com hard constraint

O gerador impõe contorno forte via:

`T_theta(x,y,z) = g(x,y) + phi(x,y) * N_theta(x,y,z)`

onde:

- `g(x,y)` extende os valores de fronteira;
- `phi(x,y)=0` na borda;
- `N_theta` e o campo predito pela rede.

No codigo:

- `g` e construido em `src/utils.py::build_dirichlet_extension`
- `phi` em `src/utils.py::build_hard_constraint_mask`
- composicao final em `src/models.py::UNetGenerator2D.forward`

### 3.3 Condicoes de contorno aplicadas

Com defaults atuais:

- esquerda: `T_LEFT = 200`
- direita: `T_RIGHT = 100`
- topo e base: perfil linear em `x` com perturbacao senoidal de amplitude `boundary_sine_amplitude` (default `1.0`)

## 4. Referencia numerica (FDM-SOR)

A referencia `T_ref` e calculada por diferencas finitas com SOR red-black:

- arquivo: `src/fdm.py`
- solver: `solve_laplace_dirichlet`
- defaults:
  - `fdm_tol = 1e-12`
  - `fdm_max_iter = 100000`
  - `fdm_omega = 1.0`

No run registrado (`2026-02-17`), o log indica:

- `fdm_iterations = 852` para grade `32x32`.

## 5. Arquitetura do modelo

### 5.1 Gerador `G`

- Tipo: U-Net 2D (`src/models.py::UNetGenerator2D`)
- Entrada:
  - campo base `g` (1 canal),
  - coordenadas fisicas normalizadas `(x,y)` (2 canais opcionais, ativas por default),
  - vetor latente `z` (modo estocastico).
- Defaults relevantes:
  - `generator_mode = stochastic_pigan`
  - `latent_dim = 8`
  - `base_channels = 12`
  - `depth = 3`
  - ativacao `SiLU`
  - pooling `avg`
  - suavizacao de saida ativa (`steps=2`, `strength=0.35`)

### 5.2 Discriminadores

- `D1` (fisico): recebe mapa de residuo laplaciano.
- `D2` (dados): recebe par de canais `[candidato, referencia]`.
- Implementacao em `src/models.py`:
  - `PhysicsDiscriminator2D`
  - `DataDiscriminator2D`
  - `FieldDualDiscriminator`
- Configuracao default:
  - `discriminator_base_channels = 12`
  - `capacity_scale = 0.72`
  - `dropout = 0.20`
  - `spectral_norm = True`

### 5.3 Operador fisico

- `LaplacianLayer` em `src/models.py`
- stencil discreto 3x3 nao treinavel
- residuo definido apenas no interior (borda zerada)

## 6. Funcao objetivo e estrategia de treinamento

### 6.1 Perdas principais

No `src/trainer.py`:

- Gerador:
  - `g_adv1 = -E[D1(fake_residual)]`
  - `g_adv2 = -E[D2(fake_pair)]`
  - `g_pde_raw = mean(|nabla^2 T|)` ponderado para foco em cantos
  - `g_bc = erro MSE de fronteira`
  - `g_total = lambda_pde_dyn*g_pde_raw + adv_grad_scale*(lambda_adv1_eff*g_adv1 + lambda_adv2_eff*g_adv2) + lambda_bc*g_bc`
- Criticos:
  - `L_D = E[D(fake)] - E[D(real)] + lambda_gp*GP + drift + gap_penalty`
  - com WGAN-GP e penalidade de gap excessivo.

### 6.2 Mecanismos de estabilidade

Implementados em `FieldPIGANTrainer`:

- `lambda_pde` adaptativo com EMA e clipping dinamico;
- balanceamento dinamico adversarial (`target_adv_over_pde`);
- gate adversarial por residuo + warmup + histerese;
- boost por estagnacao;
- gradnorm para ajustar escala adversarial;
- reducao de LR por plateau e por divergencia;
- precision refine:
  - aumenta `n_critic` tardiamente,
  - reduz teto efetivo de `lambda_pde_max`.

## 7. Pipeline experimental

Fluxo em `src/pipeline.py`:

1. inicializa sistema e device;
2. gera malha, `g`, `phi`, referencia FDM e mascaras;
3. instancia `G`, `D1`, `D2`, `LaplacianLayer`;
4. monta trainer com hiperparametros;
5. treina, avalia e salva artefatos (`metrics`, `history`, `npy`, plots, checkpoints).

Artefatos esperados:

- `runs/results/*.json`, `*.npy`, `l2_relative_vs_fdm.txt`
- `runs/plots/*.png`
- `runs/checkpoints/checkpoint_epoch_<N>.pt`

## 8. Configuracao default consolidada (codigo ativo)

Fonte: `src/config.py::ExperimentConfig` e `SystemConfig`.

| Grupo | Parametros chave (default) |
|---|---|
| Modo | `generator_mode=stochastic_pigan`, `latent_dim=8` |
| Dominio | `grid_size=32x32`, `LX=LY=1.0` |
| Treino | `epochs=4000`, `batch_size=16`, `steps_per_epoch=1` |
| LR | `gen_lr=1.15e-4`, `disc_lr=8.625e-5` |
| Pesos | `lambda_adv1=0.5`, `lambda_adv2=0.2`, `lambda_pde=37`, `lambda_bc=20`, `lambda_gp=8` |
| Estabilidade | `max_grad_norm=1.85`, `max_critic_gap=11`, `critic_gap_penalty=0.09` |
| Gate adv | `adv_warmup_epochs=120`, `adv_residual_gate_target=0.01`, `adv_residual_gate_min=0.09` |
| PDE adaptativo | `adaptive_lambda_pde=True`, `lambda_pde_min=19`, `lambda_pde_max=98` |
| Precision refine | `enable=True`, `n_critic_target=3`, `lambda_pde_max_scale=0.72` |
| Sistema | `deterministic_run=True`, `seed=42`, `use_double=True` no `main.py` |

## 9. Resultados quantitativos do run registrado (`runs/`)

### 9.1 Contexto do run

- Data/hora do log: `2026-02-17`
- Hardware: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- Device: `cuda:0`
- Duracao total: `1168.2 s` (~19.47 min)
- Taxa media aproximada: `~3.42 epocas/s` (4000 epocas, `steps_per_epoch=1`)

### 9.2 Metricas finais (`runs/results/metrics.json`)

| Metrica | Valor |
|---|---:|
| MAE | `3.0951e-04` |
| RMSE | `4.8399e-04` |
| MAPE (%) | `2.2264e-04` |
| R2 | `0.999999999736` |
| Relative L2 | `3.1648e-06` |
| Max error | `2.3383e-03` |
| PDE residual mean | `9.1884e-02` |
| PDE residual L2 | `2.4621e-01` |
| PDE residual max | `3.8584e+00` |
| Boundary error | `1.6973e-13` |

### 9.3 Baseline vs final (`training_gain.json`)

| Indicador | Baseline | Final | Ganho |
|---|---:|---:|---:|
| RMSE | `6.2327e-02` | `4.8399e-04` | `128.78x` |
| PDE residual mean | `3.1362e+00` | `9.1884e-02` | `34.13x` |
| PDE residual max | `9.2131e+00` | `3.8584e+00` | `2.39x` |
| Relative L2 | `4.0756e-04` | `3.1648e-06` | `128.78x` |

### 9.4 Saude adversarial (`adversarial_health.json`)

| Metrica | Valor |
|---|---:|
| `critics_paused_ratio` | `0.0` |
| `critics_reduced_ratio` | `0.0` |
| `d1_nonzero_ratio` | `1.0` |
| `d2_nonzero_ratio` | `1.0` |
| `disc_updates_mean` | `1.5135` |
| `adv_gate_open_ratio` | `0.97175` |
| `adv_gate_start` | `0.00833` |
| `adv_gate_end` | `1.0` |
| `adv_over_pde_start` | `4.3178e-04` |
| `adv_over_pde_end` | `2.5050e-02` |
| `g_residual_mean_abs_start` | `3.1362` |
| `g_residual_mean_abs_end` | `0.1195` |

### 9.5 Dinamica de treino (extraida de `training_history.json`)

- Melhor `g_residual_mean_abs`: `0.1075` na epoca `3986`.
- `g_residual_mean_abs` final: `0.1195`.
- `g_adv_gate` atingiu `1.0` na epoca `120`.
- `lr_plateau_triggered` ocorreu 10 vezes nas epocas:
  - `260, 378, 582, 900, 964, 1016, 1120, 1171, 1221, 1271`
- `lr_drop_triggered` por divergencia: `0` ocorrencias.
- `n_critic_effective`:
  - inicia em `1`
  - sobe para `2` na epoca `2849`
  - sobe para `3` na epoca `3099`
- `lambda_pde_cap_eff`:
  - inicia em `98.0`
  - comeca a reduzir na epoca `2600`
  - termina em `70.56`

### 9.6 Observacao fisica critica

O proprio pipeline registra warning:

- `Meta de tolerancia fisica nao atingida`
- alvo configurado: `1e-3`
- residual final observado: `~1e-1`

Interpretacao:

- o modelo esta muito aderente ao campo de referencia;
- porem o residuo fisico absoluto ainda esta duas ordens de magnitude acima da meta declarada.

## 10. Qualidade de software e engenharia

### 10.1 Estrutura de codigo ativo

- Arquivos Python ativos (sem `orphaned/`, `.venv/`, cache): `20`
- Linhas Python ativas aproximadas: `10001`
- Distribuicao:
  - `src/`: `8` arquivos, `6859` linhas
  - `scripts/`: `5` arquivos, `2604` linhas
  - `tests/`: `6` arquivos, `389` linhas

### 10.2 Testes existentes

Cobertura funcional observada na suite:

- `test_pipeline_execution.py`: fluxo end-to-end CPU
- `test_checkpoint_resume.py`: retomar treino por checkpoint
- `test_field_training_modes.py`: modos adversarial/PDE
- `test_laplacian_field_operator.py`: operador laplaciano discreto
- `test_cpu_fallback.py`: fallback para CPU

### 10.3 Estado de execucao dos testes neste ambiente

Comandos executados nesta analise:

- `python -m pytest -q`
- `.venv\Scripts\python.exe -m pytest -q`

Resultado:

- ambos falharam na coleta, por problema de ambiente:
  - ausencia de `torch` no Python global;
  - falha de extensoes C de `torch` e `numpy` no `.venv` atual.

Conclusao:

- a suite esta definida, mas esta execucao local nao estava pronta para valida-la.

### 10.4 Duplicacao de codigo

Fonte: `report/jscpd-after-3/jscpd-report.json` (`detectionDate=2026-02-19T05:47:45.170Z`)

- arquivos analisados: `17`
- linhas analisadas: `4374`
- clones detectados: `0`
- duplicacao: `0%`

## 11. Automacao experimental (Optuna e scripts)

Scripts principais:

- `scripts/optuna_search.py`: busca focada em dinamica adversarial e estabilidade
- `scripts/tune_optuna_physics_first.py`: busca com prioridade fisica
- `scripts/tune_active_pigan.py`: varredura ativa em duas fases
- `scripts/run_final_4000_best.py`: treino final longo a partir de `best_trial.json`
- `scripts/run_optuna_*_3050.ps1`: presets para RTX 3050 6GB

Saidas padrao de estudo:

- `best_trial.json`
- `study_summary.json`
- `trials/trial_XXXX/results/optuna_trial_summary.json`

## 12. Reproducibilidade

### 12.1 Comandos de execucao

Treino principal:

```powershell
python main.py
```

Treino deterministico explicito:

```powershell
python main.py --deterministic
```

Resume por checkpoint:

```powershell
python main.py --resume-checkpoint runs/checkpoints/checkpoint_epoch_1000.pt
```

### 12.2 Dependencias declaradas

`requirements.txt`:

- `torch>=1.9.0`
- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `optuna>=3.0.0`

## 13. Limitacoes e riscos cientificos

- Escopo atual focado em Laplace 2D estacionaria, dominio retangular.
- Referencia numerica interna e FDM-SOR; nao ha benchmarking externo consolidado no repositorio ativo.
- Ha alta aderencia ao campo FDM, mas a meta de residuo fisico (`1e-3`) nao foi atingida no run principal.
- Robustez fora da condicao de contorno usada (`boundary_sine_amplitude=1.0`) nao esta validada aqui.
- O ambiente Python local desta analise esta inconsistente para testes automatizados (torch/numpy C-ext).

## 14. Material pronto para artigo

### 14.1 Tabelas prontas

1. Configuracao experimental (Secao Metodos): usar Secao 8.
2. Metricas finais e baseline (Secao Resultados): usar Secao 9.2 e 9.3.
3. Saude adversarial e estabilidade (Secao Discussao): usar Secao 9.4 e 9.5.

### 14.2 Figuras prontas no repositorio

- `runs/plots/field_comparison.png`
- `runs/plots/training_curves.png`
- `runs/plots/training_history.png`
- `runs/plots/physics_consistency.png`
- `runs/plots/gan_quality_metrics.png`
- `runs/plots/ensemble_predictions.png`
- `runs/plots/uncertainty_analysis.png`

### 14.3 Narrativa sugerida para Discussao

- Ponto forte: erro de campo extremamente baixo contra FDM.
- Ponto de atencao: residuo PDE absoluto ainda acima da meta strict.
- Hipotese tecnica: regularizacao adversarial estabiliza e melhora campo, mas o nivel de fisica strict pode exigir ajuste de balanceamento final e/ou refinamento de malha/protocolo.

## 15. Referencias teoricas usadas no projeto

- Raissi, M.; Perdikaris, P.; Karniadakis, G. E. (2019). Physics-Informed Neural Networks.
- Goodfellow, I. et al. (2014). Generative Adversarial Nets.
- Arjovsky, M.; Chintala, S.; Bottou, L. (2017). Wasserstein GAN.
- Gulrajani, I. et al. (2017). Improved Training of Wasserstein GANs (WGAN-GP).

---

Relatorio gerado a partir de leitura direta do codigo-fonte, scripts, testes e artefatos em disco no estado atual do workspace.
