import mlx.core as mx

a = mx.array([[1,2],[3,4]], mx.float32)
b = mx.flatten(a)

print(a)
print(b)
