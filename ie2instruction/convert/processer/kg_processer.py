import random
random.seed(42)
import numpy as np
np.random.seed(42)
from convert.processer.processer import Processer


class KGProcesser(Processer):
    def __init__(self, type_list, role_list, type_role_dict, negative=-1):
        super().__init__(type_list, role_list, type_role_dict, negative)
    
    def filter_lable(self, record):
        return record

    def set_negative(self, neg_schema):
        self.label_length = len(self.role_list)
        self.negative = neg_schema
        self.negative_role = neg_schema

    def get_schemas(self, task_record=None):
        type_role_list = []
        for key, value in self.type_role_dict.items():
            type_role_list.append({'entity_type': key, 'attribute':list(value)})
        return type_role_list
    

    def get_task_record(self, record):
        return record['relation']
    

    def get_positive_type_role(self, records):
        positive_role = set()
        for record in records:
            positive_role.add(record['relation'])
        return list(self.type_role_dict.keys())[0], positive_role, self.role_list


    def get_final_schemas(self, total_schemas):
        final_total_schemas = []
        for it in total_schemas:
            tmp_schema = []
            for iit in it:
                tmp_schema.append({'entity_type':iit[0], 'attribute': iit[1]})
            final_total_schemas.append(tmp_schema)
        return final_total_schemas
    
    
    def negative_sample(self, record, split_num=4, random_sort=True):
        task_record = self.get_task_record(record)
        # 负采样
        positive_type, positive_role, type_role = self.get_positive_type_role(task_record)
        if self.negative == 0:   
            tmp_total_schemas = [[positive_type, list(positive_role)],]
        else:
            negative_role = type_role - positive_role
            negative_role = list(negative_role)
            if self.negative_role > 0: 
                neg_length = sum(np.random.binomial(1, self.negative_role, self.label_length))
                if len(positive_role) == 0:
                    neg_length = max(neg_length, 1)
                neg_length = min(neg_length, len(negative_role))
                negative_role = random.sample(negative_role, neg_length) 
            attribute = list(positive_role) + negative_role
            if not random_sort:
                attribute = sorted(attribute)
            else:
                random.shuffle(attribute)
            tmp_total_schemas = [[positive_type, attribute],]

        # 排序
        if not random_sort:
            tmp_total_schemas = sorted(tmp_total_schemas)
        else:
            random.shuffle(tmp_total_schemas)

        # 按照split_num划分
        negative_length = max(len(tmp_total_schemas) // split_num, 1) * split_num
        total_schemas = []
        for i in range(0, negative_length, split_num):
            total_schemas.append(tmp_total_schemas[i:i+split_num])
        # 剩余的不足split_num的部分
        remain_len = max(1, split_num // 2)
        if len(tmp_total_schemas) - negative_length >= remain_len:
            total_schemas.append(tmp_total_schemas[negative_length:])
        else:
            total_schemas[-1].extend(tmp_total_schemas[negative_length:])
        return self.get_final_schemas(total_schemas)

    
