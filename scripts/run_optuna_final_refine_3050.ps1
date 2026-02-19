# Preset Optuna focado em refino de alta precisão para RTX 3050 6GB.
# - Ativa busca de precision_refine_* no optuna_search.py
# - Prioriza gate adversarial aberto e críticos ativos na fase final
# - Persistente via SQLite para retomar overnight

param(
    [string]$PythonExe = "python",
    [string]$OutputRoot = "runs_optuna_final_refine_3050",
    [string]$StudyName = "pigan_final_refine_3050",
    [int]$Trials = 120,
    [int]$TimeoutSeconds = 39600
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$storage = "sqlite:///$OutputRoot/optuna.db"

Write-Host "[final-refine] python=$PythonExe"
Write-Host "[final-refine] output_root=$OutputRoot"
Write-Host "[final-refine] study_name=$StudyName"
Write-Host "[final-refine] storage=$storage"
Write-Host "[final-refine] trials=$Trials timeout_s=$TimeoutSeconds"

& $PythonExe scripts/optuna_search.py `
  --output-root $OutputRoot `
  --study-name $StudyName `
  --storage $storage `
  --trials $Trials `
  --timeout $TimeoutSeconds `
  --use-gpu `
  --show-progress `
  --focus-final-refine `
  --generator-mode stochastic_pigan `
  --latent-dim 8 `
  --epochs 220 `
  --steps-per-epoch 1 `
  --batch-size 12 `
  --grid-size 32 `
  --analysis-num-samples 6 `
  --generator-base-channels 12 `
  --generator-depth 3 `
  --discriminator-base-channels 12 `
  --boundary-sine-amplitude 1.0 `
  --max-paused-ratio 0.10 `
  --max-reduced-ratio 0.35 `
  --min-adv-gate-end 0.85 `
  --min-adv-gate-open-ratio 0.75 `
  --pruner-startup-trials 10 `
  --pruner-warmup-epochs 45 `
  --pruner-interval 5 `
  --top-k 20

Write-Host ""
Write-Host "[done] principais arquivos:"
Write-Host "  $OutputRoot/best_trial.json"
Write-Host "  $OutputRoot/study_summary.json"
Write-Host "  $OutputRoot/trials/trial_XXXX/results/optuna_trial_summary.json"
