from llm_sdk.llm_sdk import Small_LLM_Model


if __name__ == '__main__':
    small_model = Small_LLM_Model()
    print("small_model:", small_model)
    print("OK!, start CallMeMaybe!\n")
    small_model.encode("hello!, what's your name?")
    tensor_ids = small_model.encode("Hello, What's your name?")
    print("tensor_ids from encode:", tensor_ids)