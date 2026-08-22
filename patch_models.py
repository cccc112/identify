import h5py
import os
import re

def patch_model(path):
    if not os.path.exists(path): 
        print(f"File not found: {path}")
        return
    try:
        with h5py.File(path, 'r+') as f:
            if 'model_config' in f.attrs:
                config = f.attrs['model_config']
                config_str = config.decode('utf-8') if isinstance(config, bytes) else config
                
                # 1. 修正 batch_shape
                if '"batch_shape"' in config_str:
                    config_str = config_str.replace('"batch_shape"', '"batch_input_shape"')
                    
                # 2. 修正 DTypePolicy (將複雜的 dict 取代為單純的 "float32" 讓 Keras 2 看得懂)
                # Keras 3 儲存格式: "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}
                # Keras 2 讀取格式: "dtype": "float32"
                pattern = r'\{"module": "keras", "class_name": "DTypePolicy", "config": \{"name": "([^"]+)"\}, "registered_name": null\}'
                if "DTypePolicy" in config_str:
                    config_str = re.sub(pattern, r'"\1"', config_str)

                f.attrs['model_config'] = config_str.encode('utf-8') if isinstance(config, bytes) else config_str
                print(f"Successfully patched {path}")
            else:
                print(f"No need to patch {path}")
    except Exception as e:
        print(f"Error patching {path}: {e}")

if __name__ == "__main__":
    patch_model('C:/hand/best_model.h5')
    patch_model('C:/hand/augmented_model.h5')
    patch_model('C:/hand/symbol.h5')
