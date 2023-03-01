import os

PLUGINS = []
if 'PLUGINS' in os.environ:
    for path in os.environ['PLUGINS'].split(','):
        PLUGINS.append(path)
ONNX_ONLY = False if 'ONNX_ONLY' not in os.environ or not eval(os.environ['ONNX_ONLY']) else True
DISABLE_FASTLDM = False if 'DISABLE_FASTLDM' not in os.environ or not eval(os.environ['DISABLE_FASTLDM']) else True
print('DISABLE_FASTLDM', DISABLE_FASTLDM)
print('ONNX_ONLY', ONNX_ONLY)
print('PLUGINS', PLUGINS)
TRT_PATH = None if 'TRT_PATH' not in os.environ else os.environ['ONNX_ONLY']
TRT_NUM_WORKER = 1 if 'TRT_NUM_WORKER' not in os.environ else int(os.environ['TRT_NUM_WORKER'])