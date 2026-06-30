from pydantic import ValidationError


try:
	x = 10
	raise ValueError("ValueError")
	assert x == 5
except ValidationError as e:
	print(e)

