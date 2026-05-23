---
name: pigan-rigorosa
description: >
  Guia para formular, implementar, treinar, validar e revisar Physics-Informed
  GANs (PI-GANs) aplicadas a EDPs. Use quando o usuário mencionar PI-GAN,
  PINN com componente adversarial, GAN física, discriminador de resíduo,
  WGAN-GP com restrição física, quantificação de incerteza em campos físicos,
  modo estocástico/determinístico, colapso de modo, resíduo PDE alto,
  boundary error, ou revisão/correção de texto acadêmico sobre PI-GANs.
---

# PI-GAN Rigorosa

Use este skill para revisar rigor matemático, implementação e validação de
PI-GANs. Declare sempre a variante arquitetural, o modo do gerador e o papel
de referências numéricas no treinamento.

## Referências

- Leia `references/formulation.md` quando precisar escrever ou revisar perdas,
  críticos, hard constraint, operador Laplaciano ou pesos adaptativos.
- Leia `references/pitfalls.md` quando houver colapso de modo, variância nula,
  resíduo PDE alto, boundary error não nulo ou treino instável.
- Leia `references/validation.md` quando precisar avaliar métricas, figuras,
  regressões ou alegações de incerteza epistêmica.

## Workflow

1. Identifique se a PI-GAN é generator-physics, discriminator-physics ou
   híbrida. Não use "PI-GAN" sem especificar a variante.
2. Confirme o modo do gerador:
   - `stochastic_pigan`: exige `latent_dim > 0`; ensemble pode medir incerteza.
   - `deterministic_adversarial`: força `latent_dim = 0`; ensemble é duplicado.
3. Verifique condições de contorno:
   - Com hard constraint, use `T = T_boundary + phi * T_network`.
   - Com hard constraint ativo, `lambda_bc` deve ser efetivamente zero.
   - `boundary_error` deve ficar abaixo de `1e-10` em precisão dupla.
4. Revise a perda do gerador:
   `L_G = lambda_PDE_dyn L_PDE + lambda_adv_eff L_adv + lambda_BC L_BC`
   e adicione `lambda_div L_div` apenas em modo estocástico quando usado.
5. Revise o termo adversarial com argumentos completos. Para discriminador
   de dados pareado, escreva `D_2(T_theta, T_ref)`, não apenas `D(fake)`.
6. Reporte métricas físicas e de ajuste: MAE/RMSE/R2, erro relativo L2,
   resíduo PDE mean/L2/max e boundary error.
7. Remova ou marque como inválidas figuras de incerteza se `latent_dim = 0`
   ou se a variância do ensemble for numericamente nula.

## Regras Críticas

- Não apresente incerteza epistêmica com `latent_dim = 0`.
- Não chame um modelo assistido por FDM/FVM de "puramente physics-informed".
- Não descreva os termos de WGAN como Wasserstein sem mencionar a restrição
  Lipschitz imposta por gradient penalty ou mecanismo equivalente.
- Não omita o argumento real/fake dos críticos.
- Não reporte apenas métricas de ajuste quando o problema é governado por EDP.

## Checklist

- [ ] Variante arquitetural declarada.
- [ ] Modo estocástico/determinístico declarado.
- [ ] Papel de `T_ref`/FDM/FVM declarado, se usado.
- [ ] Hard constraint e `lambda_bc` coerentes.
- [ ] `boundary_error`, `pde_residual_mean`, `pde_residual_l2` e
      `pde_residual_max` reportados.
- [ ] Termos adversariais com argumentos completos.
- [ ] WGAN-GP, drift e gap penalty definidos quando presentes.
- [ ] UQ reportada somente com diversidade real do ensemble.
