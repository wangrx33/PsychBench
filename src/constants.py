import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ))

# from workers.huatuo import HuatuoWorker

from workers.mymodel import MyModelWorker
from workers.PsychAiD import PsychAiDWorker


id2worker_class = {
    # 'huatuo': HuatuoWorker,
    'psychAiD': PsychAiDWorker,
    'my_model': MyModelWorker, # modify here
} 


