# Diagnóstico de Problemas Comuns em PI-GAN

## Variância de ensemble nula

Sintoma: variância média do ensemble próxima de zero, por exemplo `< 1e-20`.

Verifique:

- `generator_mode` está em `deterministic_adversarial`.
- `latent_dim` é `0` ou foi forçado para `0` na configuração.
- `predict()` está usando `deterministic=True` por `lambda_adv=0`.
- O gerador recebe `z`, mas aprendeu a ignorar o latente por warmup muito longo.

Correções típicas:

- Usar `generator_mode="stochastic_pigan"` e `latent_dim >= 8`.
- Encerrar `adv_warmup_epochs` em até 15% do total de épocas.
- Usar `lambda_adv` em torno de `1e-3` a `1e-2`.
- Usar `target_adv_over_pde` em torno de `0.03` a `0.10`.
- Adicionar `lambda_diversity` conservador, por exemplo `1e-4`.

## Boundary error não nulo com hard constraint

Sintoma: `boundary_error > 1e-6` quando `hard_constraint_bc=True`.

Verifique:

- A máscara `phi` é exatamente zero na fronteira.
- A composição usa `T = T_boundary + phi * T_network`.
- Pós-processamento, suavização ou normalização não altera a fronteira.
- A avaliação usa a mesma discretização da malha de treino.

Com hard constraint ativo, `lambda_bc` deve ser efetivamente zero. Se a
fronteira depende de penalidade, o método não está impondo Dirichlet por
construção.

## Resíduo PDE alto apesar de MAE/R2 bons

Sintoma: `R2` próximo de 1, mas `pde_residual_max` ou mapa de resíduo mostra
hotspots.

Verifique:

- O Laplaciano usa `hx` e `hy` corretos, inclusive em malhas não quadradas.
- O resíduo é medido apenas no interior quando o stencil usa `padding=0`.
- O peso PDE adaptativo não está saturado no teto durante todo o treino.
- A escala de referência do resíduo é compatível com o resíduo convergido.

Não aceite MAE/R2 como evidência suficiente de solução física.

## Instabilidade adversarial

Sintomas: gaps dos críticos crescem, gradientes explodem, termo PDE piora ao
ativar adversarial.

Verifique:

- WGAN-GP está sendo aplicado no domínio correto de cada crítico.
- O discriminador condicional recebe pares completos, como `(T_theta, T_ref)`.
- `lambda_adv_eff` não domina `lambda_PDE_dyn * L_PDE`.
- Drift e gap penalty estão definidos e reportados quando usados.

## Uso incorreto de referência numérica

Se `T_ref` entra no discriminador ou na loss, declare o modelo como
"informado por física e assistido por referência numérica". Não chame de
puramente physics-informed.
