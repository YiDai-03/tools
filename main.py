from pyner.component import Component
from pyner.config.new_config import configs as config

if __name__ == '__main__':
    tokenizer = Component(config)
    tokenizer.predict()