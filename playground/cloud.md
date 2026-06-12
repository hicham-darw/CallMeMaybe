# **CallMeMaybe Project Cloud**

## **Project Goal**

**CallMeMaybe** is an experimental local LLM function-calling project.

The project tries to convert natural-language prompts into strict JSON function-call objects. It uses:

- **Function definitions** from `data/input/functions_definition.json`
- **Prompt tests** from `data/input/function_calling_tests.json`
- **Pydantic validation** for input structure
- **A local Hugging Face causal language model** through `llm_sdk`
- **A planned constrained-decoding state machine** for valid JSON generation

Expected output shape:

```json
{
  "prompt": "<user prompt>",
  "name": "<function_name_or_null>",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

## **High-Level Structure**

```text
CallMeMaybe/
├── src/
│   ├── __main__.py
│   ├── arg_parser.py
│   ├── processing_stage.py
│   ├── json_manager.py
│   ├── json_reader.py
│   ├── json_parser.py
│   ├── json_generator.py
│   ├── json_writer.py
│   ├── validator.py
│   ├── state.py
│   ├── enums.py
│   ├── loader.py
│   └── StructuredGenerator.py
├── llm_sdk/
│   └── llm_sdk/
│       └── __init__.py
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── playground/
│   ├── __main__.py
│   ├── parse.py
│   └── cloud.md
├── assets/
├── pyproject.toml
├── uv.lock
├── Makefile
└── READEME.md
```

## **Main Execution Flow**

The main entry point is:

```text
src/__main__.py
```

Current pipeline:

```text
ArgParser
   ↓
JSONManager
   ↓
JSONReader
   ↓
JSONParser
   ↓
JSONGenerator
```

The `JSONWriter` stage exists, but it is currently commented out inside `JSONManager`.

## **Pipeline Responsibilities**

### **1. Argument Parsing**

File:

```text
src/arg_parser.py
```

Responsibilities:

- Reads CLI arguments.
- Provides default paths for input, output, and model.

Default values:

```text
--functions_definition data/input/functions_definition.json
--input                data/input/function_calling_tests.json
--output               data/output/function_calling_results.json
--model                Qwen/Qwen3-0.6B
```

### **2. Pipeline Management**

File:

```text
src/json_manager.py
```

Responsibilities:

- Creates the processing pipeline.
- Stores the configured stages.
- Runs each stage in order.

Current stages:

```python
[
    JSONReader(),
    JSONParser(),
    JSONGenerator(),
]
```

### **3. Stage Interface**

File:

```text
src/processing_stage.py
```

Responsibilities:

- Defines the abstract `ProcessingStage` interface.
- Requires all pipeline stages to implement:

```python
execute(self, data: Any) -> Any
```

### **4. JSON Reading**

File:

```text
src/json_reader.py
```

Responsibilities:

- Loads the prompts JSON file.
- Loads the function definitions JSON file.
- Passes raw JSON data to the next stage.

Output shape:

```python
{
    "functions_definition": [...],
    "prompts": [...]
}
```

### **5. JSON Parsing and Validation**

Files:

```text
src/json_parser.py
src/validator.py
```

Responsibilities:

- Validates prompt objects.
- Validates function definition objects.
- Converts raw dictionaries into Pydantic models.

Main schemas:

```python
PromptSchema
FunctionDefinitionSchema
```

Prompt format:

```json
{
  "prompt": "What is the sum of 2 and 3?"
}
```

Function definition format:

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": {
      "type": "number"
    },
    "b": {
      "type": "number"
    }
  },
  "returns": {
    "type": "number"
  }
}
```

### **6. JSON Generation**

File:

```text
src/json_generator.py
```

Responsibilities:

- Extends `Small_LLM_Model`.
- Builds a strict function-calling prompt.
- Sends the prompt to a local causal language model.
- Starts implementing constrained decoding with allowed tokens.

Current generation prompt tells the model to:

- Choose one available function.
- Output only valid JSON.
- Never invent unavailable functions.
- Extract parameter values accurately.

Current status:

- Prompt construction is mostly implemented.
- Model loading is handled by `Small_LLM_Model`.
- Constrained decoding is started but incomplete.
- Only the first prompt is currently processed:

```python
prompt = data["prompts"][0]
```

### **7. LLM SDK**

File:

```text
llm_sdk/llm_sdk/__init__.py
```

Responsibilities:

- Loads a Hugging Face tokenizer.
- Loads a causal language model.
- Selects device automatically:
  - `mps`
  - `cuda`
  - `cpu`
- Provides:
  - `encode`
  - `decode`
  - `generate`
  - `get_logits_from_input_ids`
  - tokenizer file helpers

Default model:

```text
Qwen/Qwen3-0.6B
```

## **Constrained Decoding Design**

Files:

```text
src/state.py
src/json_generator.py
```

The project is moving toward a finite-state-machine decoder that only allows valid next tokens for each JSON position.

Planned states:

```python
IN_START
IN_OPEN_BRACE
IN_KEY
IN_DOUBLE_POINTS
IN_VALUE
IN_COMMA
IN_CLOSE_BRACE
IN_END
```

Example intended flow:

```text
START
  ↓
{
  ↓
"prompt" | "name" | "parameters"
  ↓
:
  ↓
value
  ↓
, or }
  ↓
END
```

Current issue:

```python
self.state = JSONState.IDLE
```

But `JSONState.IDLE` does not exist. The enum currently starts with `IN_START`.

## **Data Files**

### **Function Definitions**

File:

```text
data/input/functions_definition.json
```

Available functions:

- `fn_add_numbers`
- `fn_greet`
- `fn_reverse_string`
- `fn_get_square_root`
- `fn_substitute_string_with_regex`

### **Prompt Tests**

File:

```text
data/input/function_calling_tests.json
```

Contains natural-language prompts such as:

- Sum two numbers.
- Greet a person.
- Reverse a string.
- Calculate a square root.
- Replace text using regex-like intent.

## **Current Implementation Gaps**

### **Critical**

- `JSONState.IDLE` is referenced but not defined.
- `JSONGenerator.execute()` currently handles only one prompt.
- `JSONWriter` is not active in the pipeline.
- `JSONWriter.execute()` opens the output file without write mode.
- Constrained decoding has incomplete token handling.
- `JSONGenerator.execute()` uses `tok` instead of `token` inside the allowed-token loop.
- Tensor concatenation currently uses `input_ids = input_ids + next_token`, but generation normally needs `torch.cat`.

### **Structural**

- `src/loader.py` duplicates older reader behavior.
- `src/StructuredGenerator.py` is empty.
- `playground/parse.py` is only an argparse experiment.
- `src/__main__.py` contains a large commented prototype block.
- `READEME.md` should probably be renamed to `README.md`.

### **Security**

- `json_generator.py` contains a hardcoded Hugging Face token in `set_token_hugging_face`.
- That token should be removed from code and loaded from an environment variable instead.

## **Recommended Cleanup Plan**

### **Step 1: Stabilize the Pipeline**

- Fix `JSONState.IDLE`.
- Make `JSONGenerator` process every prompt.
- Return generated results from `JSONGenerator.execute()`.
- Enable `JSONWriter` as the final stage.
- Create `data/output/` if it does not exist.

### **Step 2: Fix JSON Output**

- Open output files with `"w"` mode.
- Write a valid JSON array.
- Add clear formatting with `indent=2`.
- Validate generated output before writing.

### **Step 3: Complete Constrained Decoding**

- Define legal transitions in `FiniteStateMachine.change_state`.
- Track current JSON field.
- Convert allowed strings into tokenizer IDs correctly.
- Mask logits using token IDs.
- Append tokens using `torch.cat`.
- Stop only when a complete object reaches `IN_END`.

### **Step 4: Remove Dead Code**

- Delete or archive old prototype code from `src/__main__.py`.
- Remove `src/loader.py` if it is no longer used.
- Either implement or remove `src/StructuredGenerator.py`.
- Keep playground files only for active experiments.

### **Step 5: Improve Documentation**

- Rename `READEME.md` to `README.md`.
- Add install instructions.
- Add run command examples.
- Add input/output JSON examples.
- Document current model requirements.

## **Suggested Run Command**

From the project root:

```bash
python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --model Qwen/Qwen3-0.6B
```

## **Clean Command**

The `Makefile` provides:

```bash
make clean
```

It removes:

- `__pycache__`
- `.mypy_cache`
- `.pytest_cache`
- `.pyc` files

## **Project Summary**

**CallMeMaybe** is best understood as a function-calling JSON generation pipeline.

The project already has the right broad architecture:

```text
read → validate → generate → write
```

The strongest next improvement is to make that pipeline fully executable end to end, then finish constrained decoding after the basic JSON generation flow is stable.
