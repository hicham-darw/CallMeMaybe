from src.llm_sdk import llm_sdk


model = llm_sdk.Small_LLM_Model()
input_ids = model.encode("what is my name ?")
print("model:", model)
print("input ids encode():", input_ids);

