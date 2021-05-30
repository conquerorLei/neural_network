import numpy as np

if __name__ == "__main__":
    angle = np.random.randint(-180, 180, 5)
    print(angle)
    print(angle.dtype)
    print(angle.data)
    print(angle.any())
    print(angle.all())
