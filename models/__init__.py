# MNIST
from .mnist_ss import mnist_ss_28, mnist_ss_56
from .mnist_sevf import mnist_sevf_scalar_28, mnist_sevf_scalar_56, mnist_sevf_vector_28, mnist_sevf_vector_56
from .mnist_cnn import mnist_cnn_28, mnist_cnn_56
from .mnist_kanazawa import mnist_kanazawa_28, mnist_kanazawa_56
from .mnist_xu import mnist_xu_28, mnist_xu_56
from .mnist_dss import mnist_dss_vector_28, mnist_dss_vector_56, mnist_dss_scalar_28, mnist_dss_scalar_56
from .mnist_ses import mnist_ses_scalar_28, mnist_ses_scalar_56, mnist_ses_vector_28, mnist_ses_vector_56
from .mnist_ses import mnist_ses_scalar_28p, mnist_ses_scalar_56p, mnist_ses_vector_28p, mnist_ses_vector_56p


# # STL 10
from .stl_wrn import wrn_16_8
from .stl_kanazawa import wrn_16_8_kanazawa
from .stl_xu import wrn_16_8_xu
from .stl_ss import wrn_16_8_ss
from .stl_dss import wrn_16_8_dss
from .stl_ses import wrn_16_8_ses_a, wrn_16_8_ses_b, wrn_16_8_ses_c
