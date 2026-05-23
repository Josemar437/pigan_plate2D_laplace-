---
name: camelcase-refactor
description: >
  Refatora código reescrevendo todas as funções, métodos, variáveis e identificadores de camelCase para snake_case
  seguindo a PEP 8 de forma rigorosa e completa, sem quebrar o código. Use este skill sempre que o usuário mencionar
  "refatorar para snake_case", "converter camelCase", "padronizar para PEP 8", "PEP8 refactor", "seguir PEP 8",
  "reescrever funções em snake_case", ou qualquer pedido de conversão de estilo de nomenclatura em código Python.
  Também use quando o usuário pedir uma "refatoração perfeita" de código que contém camelCase em funções ou variáveis Python.
---

# PEP 8 snake_case Refactor Skill

Refatora código Python convertendo todos os identificadores de `camelCase` (e variações) para `snake_case`,
seguindo rigorosamente a [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/).

---

## Regras PEP 8 aplicadas

| Tipo de identificador        | Convenção PEP 8         | Exemplo antes → depois              |
|------------------------------|--------------------------|--------------------------------------|
| Funções e métodos            | `snake_case`             | `calcularMedia` → `calcular_media`  |
| Variáveis locais e globais   | `snake_case`             | `nomeCompleto` → `nome_completo`    |
| Parâmetros de funções        | `snake_case`             | `userId` → `user_id`                |
| Módulos e pacotes            | `snake_case` curto       | `myModule` → `my_module`            |
| Classes                      | `PascalCase` (preservar) | `UserModel` → manter                |
| Constantes de módulo         | `SCREAMING_SNAKE_CASE`   | `maxRetries` → `MAX_RETRIES`        |
| `__dunder__`                 | Preservar                | `__init__` → manter                 |

### O que NÃO converter
- Strings literais: `"minhaChave"`, `"camelKey"` — são dados, não identificadores
- Chaves de dicionário/JSON provenientes de APIs externas
- Nomes de métodos herdados de bibliotecas externas: `.appendChild()`, `.getElementById()`
- Variáveis de ambiente em strings: `os.environ["myVar"]`
- Nomes de classes (`PascalCase` é o padrão PEP 8 correto)
- `__dunder__` methods

---

## Processo de refatoração

### Passo 1 — Inventariar
Liste todos os identificadores camelCase encontrados, classificados por tipo:
- Funções/métodos definidos no código
- Variáveis locais e globais
- Parâmetros de funções
- Constantes (verificar se devem virar `SCREAMING_SNAKE_CASE`)

### Passo 2 — Mapa de renomeação
Produza um mapa explícito e completo antes de alterar qualquer linha:
```
getUserById      → get_user_by_id
userName         → user_name
calculateTotal   → calculate_total
maxRetries       → MAX_RETRIES   (constante de módulo)
```

### Passo 3 — Aplicar em ordem segura
1. Renomear definições (função/variável/classe)
2. Renomear todas as chamadas e referências ao identificador
3. Renomear parâmetros nas assinaturas **e** em todos os usos no corpo
4. Converter constantes para `SCREAMING_SNAKE_CASE` onde aplicável

### Passo 4 — Verificação final
- Buscar por identificadores camelCase restantes que deveriam ter sido convertidos
- Confirmar que strings literais e APIs externas **não** foram alteradas
- Confirmar que classes permanecem em `PascalCase`
- Indicar linhas/funções onde o usuário deve testar após a refatoração

---

## Formato de saída

Sempre apresente:
1. **Mapa de renomeação** — tabela antes → depois, com o tipo de cada identificador
2. **Código refatorado completo** em bloco Python
3. **Resumo** — quantidade de identificadores alterados
4. **Avisos** — ambiguidades, constantes, nomes externos preservados intencionalmente

---

## Exemplos de uso

### Exemplo 1 — Funções e variáveis simples

**Entrada:**
```python
def calcularMedia(listaNotas):
    somaTotal = sum(listaNotas)
    numElementos = len(listaNotas)
    return somaTotal / numElementos

notaFinal = calcularMedia([8, 9, 7])
```

**Mapa de renomeação:**
| Antes           | Depois            | Tipo       |
|-----------------|-------------------|------------|
| `calcularMedia` | `calcular_media`  | função     |
| `listaNotas`    | `lista_notas`     | parâmetro  |
| `somaTotal`     | `soma_total`      | variável   |
| `numElementos`  | `num_elementos`   | variável   |
| `notaFinal`     | `nota_final`      | variável   |

**Saída:**
```python
def calcular_media(lista_notas):
    soma_total = sum(lista_notas)
    num_elementos = len(lista_notas)
    return soma_total / num_elementos

nota_final = calcular_media([8, 9, 7])
```

---

### Exemplo 2 — Classe com métodos, constantes e API externa

**Entrada:**
```python
maxRetries = 3  # constante de módulo

class UserService:
    def __init__(self, dbConnection):
        self.dbConnection = dbConnection

    def getUserById(self, userId):
        rawData = self.dbConnection.query(userId)
        fullName = rawData["fullName"]  # chave de API externa
        return fullName
```

**Mapa de renomeação:**
| Antes          | Depois           | Tipo                          |
|----------------|------------------|-------------------------------|
| `maxRetries`   | `MAX_RETRIES`    | constante de módulo           |
| `dbConnection` | `db_connection`  | parâmetro / atributo          |
| `getUserById`  | `get_user_by_id` | método                        |
| `userId`       | `user_id`        | parâmetro                     |
| `rawData`      | `raw_data`       | variável                      |
| `fullName`     | `full_name`      | variável local                |
| `"fullName"`   | ⚠️ preservado    | string — chave de API externa |

**Saída:**
```python
MAX_RETRIES = 3  # constante de módulo

class UserService:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def get_user_by_id(self, user_id):
        raw_data = self.db_connection.query(user_id)
        full_name = raw_data["fullName"]  # preservado: chave de API externa
        return full_name
```

---

## Dúvidas comuns

**O que fazer com `mixedCase` legado (PEP 8 §3.3)?**
→ PEP 8 permite manter `mixedCase` apenas para compatibilidade com código já existente. Se for código novo ou em refatoração total, converter para `snake_case`. Avisar o usuário.

**E se o mesmo nome aparecer como variável local e como chave de dict?**
→ Renomear a variável, preservar a string. Ex: `fullName = data["fullName"]` → `full_name = data["fullName"]`.

**Múltiplos arquivos?**
→ Processar arquivo por arquivo. Apresentar o mapa de renomeação global antes de começar, pois nomes públicos alterados (funções exportadas, atributos de classe) podem quebrar importações em outros módulos.

**O código vai quebrar?**
→ Sim, se houver chamadas externas ao módulo que dependam dos nomes antigos (serialização, `getattr`, reflection, testes com strings hardcoded). Identificar e avisar explicitamente nesses casos.