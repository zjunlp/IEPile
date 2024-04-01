import re
import json
from eval.extracter.extracter import Extracter


class KGExtracter(Extracter):
    def __init__(self, language="zh", NAN="NAN", prefix="输入中包含的关系三元组是：\n", Reject="No relation triples found."):
        super().__init__(language, NAN, prefix, Reject)
    

    def post_process(self, result):   
        raise NotImplementedError("This method should be implemented by subclass.")

