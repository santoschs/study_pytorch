import torch

print(torch.__version__)

vector = torch.tensor((1,3))
print(vector)
random_tensor = torch.rand(2,3,4)
#criando vetores/matrizes tensores aleatorias 
print(random_tensor)
print(random_tensor[0])
print(random_tensor[0][1][0])

# criando tensors com valores zero ou um com as dimensões de interesse
zeros = torch.zeros((3,4))
ones  = torch.zeros((3,4))
print(zeros)

# criando vetores com ranges
range = torch.range(0,10)
print(range)
range = torch.arange(10, -1, -2)
print(range)
zeros = torch.zeros_like(range)
print(zeros)

# criando tensors com tipos de dados
float_32_tensor = torch.tensor([3.1, 6.2, 9.3], dtype=None, device=None, requires_grad=False)
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_32_tensor)
print(f"Mostrando tensor com 32 bits: {float_32_tensor} e 16 bits: {float_16_tensor} on {float_32_tensor.device}")

# matrix multiplication - tensor
# two main ways of performing nn in deep learning
# 1. element wise multiplication
# 2. matrix multiplication

matrix = torch.tensor([[1,2,3],[4,5,6], [7,8,9]])
matrix_random = torch.rand((3,3))
vector_random = torch.rand(3)
print(matrix," * ", matrix_random)
print(f"Result: {matrix * matrix_random}")              # Multiplicação de elemento-elemento
resultado = torch.matmul(vector_random, matrix_random)  #  produto vetorial interno (inner dimensions)
print(f"Result Matrix Mat: {resultado}")
resultado = vector_random @ matrix_random
print(resultado.dtype)
print(resultado.shape)

print("---------------------------")
a = torch.rand((3,2))
print(a)

# transpose, min, max, mean and sum values
print(a.T)
print(a.max())
print(a.min())
print(a.mean())
print(a.sum())

# retornando o indice do min, max parametros
print(a.argmin())
print(a.argmax())