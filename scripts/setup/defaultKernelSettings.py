from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, RationalQuadratic, DotProduct, ExpSineSquared, Polynomial

DEFAULT_KERNEL_SETTINGS = {
    "class": "Matern",
    "C": 1.0,
    "C_bounds": (1e-3, 1e3),
    "length_scale": 1.0,
    "length_scale_bounds": (1e-2, 1e2),
    "nu": 2.5,
    "add_white": True,
    "white_noise": 1e-6,
    "white_bounds": (1e-9, 1e-1)
}

KERNEL_CLASSES = {
    "Matern": Matern,
    "RBF": RBF,
    "RationalQuadratic": RationalQuadratic,
    "DotProduct": DotProduct,
    "ExpSineSquared": ExpSineSquared,
    "Polynomial": Polynomial
}
