# Formulação Matemática Completa — PI-GAN

## Notação padrão

| Símbolo | Significado |
|---|---|
| $T_\theta$ | Campo predito pelo gerador (parâmetros $\theta$) |
| $T_{ref}$ | Solução de referência FDM/FVM |
| $\nabla^2$ | Operador Laplaciano espacial |
| $\Omega_{int}$ | Pontos interiores do domínio |
| $\partial\Omega$ | Fronteira do domínio |
| $N_r$ | Número de pontos de colocação interiores |
| $N_b$ | Número de pontos de fronteira |

---

## Função de perda do gerador

Forma geral:

$$L_G = \lambda_{PDE}^{dyn}\, L_{PDE} + \sum_k \lambda_{adv,k}^{eff}\, L_{adv,k} + \lambda_{BC}\, L_{BC} + \lambda_{div}\,L_{div}$$

Use a soma adversarial quando a arquitetura tiver mais de um crítico. Para
um único crítico, a soma se reduz ao termo correspondente.

### Termo PDE

$$L_{PDE} = \frac{1}{N_r}\sum_{i=1}^{N_r}\left|\nabla^2_h\, T_\theta(\mathbf{x}_i)\right|, \quad \mathbf{x}_i \in \Omega_{int}$$

### Termos adversariais por variante

**Crítico de dados pareado $D_2$:**

$$L_{adv,2} = -\mathbb{E}_{\mathbf{x}\sim\Omega}\bigl[D_2\bigl(T_\theta(\mathbf{x}),\, T_{ref}(\mathbf{x})\bigr)\bigr]$$

**Crítico físico/residual $D_r$:**

$$L_{adv,r} = -\mathbb{E}_{\mathbf{x}\sim\Omega_{int}}\bigl[D_r\bigl(\nabla_h^2 T_\theta(\mathbf{x})\bigr)\bigr]$$

Declare explicitamente qual crítico está presente. Não escreva apenas
`D(fake)` quando o discriminador é condicional ou opera em resíduos.

### Termo de diversidade estocástica

Para `stochastic_pigan` com `latent_dim > 0`:

$$L_{div} = -\frac{1}{N(N-1)}\sum_{i \neq j}\frac{\|T_i - T_j\|_2}{HW}$$

Para `deterministic_adversarial` ou `latent_dim = 0`, defina
$\lambda_{div}=0$.

### Termo de contorno (soft constraint)

$$L_{BC} = \frac{1}{N_b}\sum_{j=1}^{N_b}\bigl(T_\theta(\mathbf{x}_j) - T_{\partial\Omega}(\mathbf{x}_j)\bigr)^2$$

Com hard constraint arquitetural: $\lambda_{BC} \equiv 0$.

---

## Função de perda do crítico (WGAN-GP)

Para cada crítico $D_k$:

$$L_{D_k} = \mathbb{E}[D_k(fake)] - \mathbb{E}[D_k(real)] + \lambda_{gp,k}GP_k + \lambda_{drift,k}\mathbb{E}[D_k(real)^2] + \lambda_{gap,k}GapPenalty_k$$

Os dois primeiros termos aproximam $W_1$ apenas quando a condição de
Lipschitz é satisfeita pelo gradient penalty ou por outro mecanismo válido.

### Gradient Penalty

$$GP_k = \mathbb{E}\left[\left(\|\nabla_{\hat{x}} D_k(\hat{x})\|_2 - 1\right)^2\right]$$

onde $\hat{x} = \epsilon\, x_{real} + (1-\epsilon)\, x_{fake}$, $\epsilon \sim U[0,1]$.

### Gap Penalty

$$GapPenalty_k = \max\!\left(0,\; \bigl|\mathbb{E}[D_k(real)] - \mathbb{E}[D_k(fake)]\bigr| - G_{max,k}\right)^2$$

---

## Operador Laplaciano discreto (5 pontos)

Kernel 3×3 para stencil padrão:

$$K = \frac{1}{h^2}\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

Implementado como Conv2D com `padding=0` (apenas pontos interiores).
A divisão por $h^2$ deve estar explícita.

---

## Hard Constraint de Dirichlet

$$T_\theta(\mathbf{x}) = T_{\partial\Omega}(\mathbf{x}) + \phi(\mathbf{x}) \cdot \hat{T}_{network}(\mathbf{x})$$

onde $\phi(\mathbf{x})$ é uma função de distância à fronteira com $\phi = 0$ em $\partial\Omega$.
Isso garante satisfação exata das condições de Dirichlet por construção.

---

## Esquema adaptativo $\lambda_{PDE}^{dyn}$

$$\rho = \max\!\left(\frac{L_{PDE}}{\bar{r}_0},\; 1\right)$$
$$\lambda_{des} = \mathrm{clip}\!\left(\lambda_{base}\cdot(1 + \alpha\log_{10}\rho),\;\lambda_{min},\;\lambda_{max}^{eff}\right)$$
$$\lambda_{PDE}^{dyn} \leftarrow \beta\,\lambda_{PDE}^{dyn} + (1-\beta)\,\lambda_{des}$$

Parâmetros: $\bar{r}_0$ (escala de referência), $\alpha$ (expoente de crescimento),
$\beta$ (fator EMA), $\lambda_{max}^{eff}$ (teto — pode ser reduzido na fase final).
