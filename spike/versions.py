import inspect_ai, numpy, openai, scipy

print("inspect_ai", inspect_ai.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| openai", openai.__version__)
from inspect_ai.model import GenerateConfig

print("GenerateConfig fields with 'reason' or 'extra':",
      [f for f in GenerateConfig.model_fields if "reason" in f or "extra" in f])
print("reasoning_effort annotation:", GenerateConfig.model_fields["reasoning_effort"].annotation)
