# Preset de busca Optuna para overnight em RTX 3050 6GB.
# - Resumivel via SQLite (study persistente)
# - Foco em controle adversarial (gate/pause/boost)
# - Seguro para VRAM de 6GB (batch moderado)

param(
    [string]$PythonExe = "python",
    [string]$OutputRoot = "runs_optuna_control_3050_overnight",
    [string]$StudyName = "pigan_control_3050_overnight",
    [int]$Trials = 120,
    [int]$TimeoutSeconds = 39600
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$storage = "sqlite:///$OutputRoot/optuna.db"

Write-Host "[overnight] python=$PythonExe"
Write-Host "[overnight] output_root=$OutputRoot"
Write-Host "[overnight] study_name=$StudyName"
Write-Host "[overnight] storage=$storage"
Write-Host "[overnight] trials=$Trials timeout_s=$TimeoutSeconds"

& $PythonExe scripts/optunaSearch.py `
  --output-root $OutputRoot `
  --study-name $StudyName `
  --storage $storage `
  --trials $Trials `
  --timeout $TimeoutSeconds `
  --use-gpu `
  --show-progress `
  --generator-mode stochastic_pigan `
  --latent-dim 8 `
  --epochs 160 `
  --steps-per-epoch 1 `
  --batch-size 12 `
  --grid-size 32 `
  --analysis-num-samples 6 `
  --generator-base-channels 12 `
  --generator-depth 3 `
  --discriminator-base-channels 12 `
  --boundary-sine-amplitude 0.0 `
  --max-paused-ratio 0.35 `
  --max-reduced-ratio 0.70 `
  --min-adv-gate-end 0.65 `
  --min-adv-gate-open-ratio 0.55 `
  --pruner-startup-trials 10 `
  --pruner-warmup-epochs 35 `
  --pruner-interval 5 `
  --top-k 15

Write-Host ""
Write-Host "[done] principais arquivos:"
Write-Host "  $OutputRoot/best_trial.json"
Write-Host "  $OutputRoot/study_summary.json"
Write-Host "  $OutputRoot/trials/trial_XXXX/results/optuna_trial_summary.json"
