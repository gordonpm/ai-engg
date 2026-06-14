import mlx.core as mx

a = mx.array([1,2,3], mx.float32)
b = mx.array([4,5,6], mx.float32)

c = mx.matmul(a, b)

print(a)
print(b)
print(c)