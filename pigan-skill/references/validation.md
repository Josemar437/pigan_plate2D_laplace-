# Validação Honesta de PI-GAN

## Métricas obrigatórias

Reporte sempre métricas de ajuste e de física:

```json
{
  "mae": float,
  "rmse": float,
  "r2": float,
  "relative_l2_error_vs_fdm": float,
  "max_error": float,
  "pde_residual_mean": float,
  "pde_residual_l2": float,
  "pde_residual_max": float,
  "boundary_error": float
}
```

MAE, RMSE e R2 medem proximidade a uma referência. Resíduos PDE e erro de
fronteira medem consistência física.

## Thresholds de alarme

| Métrica | Alarme |
|---|---|
| `boundary_error` com hard constraint | `> 1e-6` |
| `boundary_error` esperado em precisão dupla | `> 1e-10` |
| `pde_residual_max` | `> 2x` do run base |
| Variância média do ensemble | `< 1e-20` em alegação de UQ |
| Warmup adversarial | `> 15%` das épocas em modo estocástico |

## Validação de incerteza epistêmica

Só reporte mapas de incerteza quando:

- `generator_mode="stochastic_pigan"`.
- `latent_dim > 0`.
- `predict(num_samples=N)` usa latentes diferentes por amostra.
- A variância do ensemble é significativamente maior que ruído numérico.
- Amostras diferentes preservam condições de contorno e resíduos aceitáveis.

Figuras que exigem modo estocástico:

- Mapa de variância ou desvio padrão.
- Coeficiente de variação espacial.
- Intervalos de confiança.
- Curva de diversidade versus número de amostras.

## Figuras recomendadas

- Campo predito médio.
- Campo de referência.
- Erro absoluto espacial.
- Resíduo PDE espacial.
- Histórico de `G_total`, `G_PDE`, `G_adv` e `D_total`.
- Histórico de `lambda_PDE_dyn` e `lambda_adv_eff`.
- Mapa ou histograma de incerteza somente quando o ensemble é estocástico.

## Regressão

Ao alterar modo, pesos ou arquitetura, compare contra o run anterior:

- MAE, R2 e boundary error não devem degradar sem justificativa.
- Resíduo máximo deve ser avaliado por mapa espacial, não apenas média.
- Ganho de diversidade não deve vir às custas de violar fronteira ou PDE.
