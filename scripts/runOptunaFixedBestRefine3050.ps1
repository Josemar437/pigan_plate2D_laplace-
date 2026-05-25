# Preset Optuna para refino final com parametros base fixados.
# - Fixa os hiperparametros "otimos" vindos de best_trial.json
# - Busca apenas precision_refine_* (fase final)
# - Persistente via SQLite para retomada

param(
    [string]$PythonExe = "python",
    [string]$BestTrial = "runs_optuna_control_3050_overnight/best_trial.json",
    [string]$OutputRoot = "runs_optuna_fixed_best_refine_3050",
    [string]$StudyName = "pigan_fixed_best_refine_3050",
    [int]$Trials = 120,
    [int]$TimeoutSeconds = 39600
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$storage = "sqlite:///$OutputRoot/optuna.db"

Write-Host "[fixed-refine] python=$PythonExe"
Write-Host "[fixed-refine] best_trial=$BestTrial"
Write-Host "[fixed-refine] output_root=$OutputRoot"
Write-Host "[fixed-refine] study_name=$StudyName"
Write-Host "[fixed-refine] storage=$storage"
Write-Host "[fixed-refine] trials=$Trials timeout_s=$TimeoutSeconds"

& $PythonExe scripts/optunaSearch.py `
  --output-root $OutputRoot `
  --study-name $StudyName `
  --storage $storage `
  --trials $Trials `
  --timeout $TimeoutSeconds `
  --use-gpu `
  --show-progress `
  --focus-final-refine `
  --fixed-from-best-trial $BestTrial `
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
  --boundary-sine-amplitude 0.0 `
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
Write-Host "  $OutputRoot/fixed_params.json"
Write-Host "  $OutputRoot/best_trial.json"
Write-Host "  $OutputRoot/study_summary.json"
Write-Host "  $OutputRoot/trials/trial_XXXX/results/optuna_trial_summary.json"
