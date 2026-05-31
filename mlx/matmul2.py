import mlx.core as mx

a = mx.array([[1,2],[3,4]], mx.float32)
b = mx.array([[5,6],[7,8]], mx.float32)

c = mx.matmul(a, b)

print(a)
print(b)
print(c)