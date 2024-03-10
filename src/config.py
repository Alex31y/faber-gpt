import torch


class Config:
    _instance = None  # This will hold the single instance of the class

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # Put any initialization here that should only happen once
            cls.vocab_size = None
            cls.batch_size = 32
            cls.block_size = 8
            cls.max_iters = 3000
            cls.eval_interval = 300
            cls.learning_rate = 1e-2
            cls.eval_iters = 200
            cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return cls._instance

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size

    def is_set_vocab_size(self):
        return self.vocab_size is not None



"""
The __new__ method is a special static method in Python, responsible for creating and returning a new instance of a class.
By overriding this method, the Config class controls the instantiation process to ensure that only one instance is 
ever created. When __new__ is called, it checks if an instance of Config already exists. If not, it creates a new 
instance and initializes it with specified attributes. If an instance already exists, it simply returns the existing 
instance.
"""
"""This implementation does not include an __init__ method. Typically, __init__ is used for initializing new 
instances of a class. However, in a singleton pattern where instance creation is controlled strictly by __new__, 
the absence of __init__ is strategic. If included, __init__ would be called every time an instance is attempted to be 
created, which could lead to re-initializing the instance multiple times, potentially overwriting changes made after 
the first instantiation. However, if needed for setting or updating instance attributes after the initial creation, 
it must be carefully managed to avoid unintended side effects.
"""