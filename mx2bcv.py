#coding=utf-8

class CrossValidation(object):


    def __init__(self, config):

        self.config = config
        self.partitions = None
        self._data_set = None
        if "data_set" in self.config:
            self.data_set = self.config["data_set"]
        pass

    def perform_partitioning(self):
        pass

    @property
    def data_set(self):
        return self._data_set

    @data_set.setter
    def data_set(self, value):
        self._data_set = value

    def get_partitions(self):
        if self.partitions is None:
            raise Exception("Partitioning is not performing, please run perform_partitioning firstly.")
        return self.partitions

    def get_data_size(self):
        if self.data_set is not None:
            data_set = self.data_set
            return data_set.get_size()
        return self.config["n_size"]
        pass

    @staticmethod
    def validate_config(cross_validation_config):
        assert "name" in cross_validation_config
        return True
        pass


class BlockRegularizedMByTwoCrossValidation(CrossValidation):

    def __init__(self, config):
        CrossValidation.__init__(self, config)
        pass

    @staticmethod
    def validate_config(config):
        is_valid = True
        is_valid = is_valid & ("m" in config)
        m = config["m"]
        return is_valid
        pass

    def perform_partitioning(self):


        import numpy
        import math
        self.partitions = []
        # n_size = self.config["n_size"]
        n_size = self.get_data_size()
        m = self.config["m"]
        if "m_prev" not in self.config:
            self.config["m_prev"] = 0
        m_prev = self.config["m_prev"]
        orth_array = None
        if "orth_array" not in self.config:
            orth_array = []
        else:
            orth_array = self.config[orth_array]
        self.config["orth_array"] = orth_array
        for temp_m in range(m_prev+1,m+1):
            two_pow_num = int(math.pow(2, math.floor(math.log(temp_m) / math.log(2))))
            column_list = []

            if temp_m == two_pow_num:
                for i in range(1,temp_m+1):
                    column_list.append(0)
                    column_list.append(1)
                orth_array.append(column_list)
            else:
                diff = temp_m - two_pow_num
                column1_list = orth_array[diff-1]
                column2_list = orth_array[two_pow_num-1]
                multiplier = len(column2_list) // len(column1_list)
                temp_column_list = []
                for i in range(len(column1_list)):
                    for j in range(multiplier):
                        temp_column_list.append(column1_list[i])
                for i in range(len(column2_list)):
                     element1 = temp_column_list[i]
                     element2 = column2_list[i]
                     if element1 == element2:
                         column_list.append(0)
                     else:
                         column_list.append(1)
                orth_array.append(column_list)
            if "blocks" not in self.config:
                block_count = len(column_list)
                block_list = []
                for i in range(n_size):
                    block_list.append(i)
                blocks = self.standardKFCV(block_list,block_count,self.config)
                self.config["blocks"] = blocks
            else:
                current_blocks = self.config["blocks"]
                blocks = []
                multiplier = len(column_list) // len(current_blocks)
                if multiplier >1:
                    for i in range(len(current_blocks)):
                        a_block = current_blocks[i]
                        output_blocks = self.standardKFCV(a_block,multiplier,self.config)
                        for j in range(len(output_blocks)):
                            blocks.append(output_blocks[j])
                else:
                    blocks = current_blocks
                self.config["blocks"] = blocks
            test_index_list = []
            train_index_list = []
            for i in range(len(column_list)):
                element = column_list[i]
                a_block = blocks[i]
                if element == 0:
                    for j in range(len(a_block)):
                        test_index_list.append(a_block[j])
                else:
                    for j in range(len(a_block)):
                        train_index_list.append(a_block[j])
            train_index = numpy.array(train_index_list)
            test_index = numpy.array(test_index_list)
            self.partitions.append([train_index,test_index])
            self.partitions.append([test_index,train_index])
        pass

    def standardKFCV(self, population, v, config):
        import numpy
        population_array = numpy.array(population)
        numpy.random.shuffle(population_array)
        population = list(population_array)
        reg_frame = False
        if "reg_frame" in self.config:
            reg_frame = bool(self.config["reg_frame"])
        blocks = [[] for i in range(0, v)]
        if not reg_frame:
            for i in range(len(population)):
                index = i % v
                blocks[index].append(population[i])
        return blocks

_supported_cross_validations = {
    "MX2BCV": BlockRegularizedMByTwoCrossValidation,
}


def create_partitions_with_config(cross_validation_config):
    assert isinstance(cross_validation_config, dict)
    assert "name" in cross_validation_config
    cv_name = cross_validation_config["name"]
    assert cv_name in _supported_cross_validations
    cv_class_name = _supported_cross_validations[cv_name]
    cross_validation = cv_class_name(cross_validation_config)
    if 'data_set' in cross_validation_config:
        cross_validation.data_set = cross_validation_config['data_set']
    is_valid = cv_class_name.validate_config(cross_validation_config)
    if not is_valid:
        return None
    cross_validation.perform_partitioning()
    partitions = cross_validation.get_partitions()
    return partitions
    pass


