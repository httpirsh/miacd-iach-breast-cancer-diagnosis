"""Preparar a GPU NVIDIA local para o TensorFlow (sem efeito se não houver GPU)."""

import ctypes
import glob
import os
import sys


def _preload_cuda_libs():
    """Carrega as bibliotecas CUDA instaladas pelo pip (tensorflow[and-cuda]).

    O TensorFlow encontra a maioria sozinho, mas não o libcusolver, e sem ele não usa a GPU.
    Carregá-las aqui pelo caminho completo resolve isso sem mexer no LD_LIBRARY_PATH.
    """
    for site in sys.path:
        for lib in sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib", "lib*.so*"))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def setup_gpu():
    """Prepara a GPU (se existir) e diz se o treino vai correr na GPU ou no CPU.

    Chamar antes de criar qualquer modelo. Ativa o "memory growth", para o TensorFlow só reservar
    a memória da GPU de que precisa (importante em GPUs pequenas, ex.: 4 GB).
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # esconder as mensagens internas do TensorFlow
    if sys.platform.startswith("linux"):
        _preload_cuda_libs()
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")   # esconder avisos de desempenho que não afetam os resultados

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", gpus[0].name)
        print(f"A treinar na GPU: {name}")
    else:
        print("GPU não encontrada: a treinar no CPU (mais lento)")
    return bool(gpus)
