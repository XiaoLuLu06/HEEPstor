import heepstorch as hp
import numpy as np
import numpy.typing as npt
import os
from string import Template

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


class CodeGenerator:
    def __init__(self, project_name: str, sequential_network: 'hp.module.SequentialNetwork'):
        self.project_name = project_name
        self.sequential_network = sequential_network

    def generate_code(self, append_final_softmax) -> str:
        def gen_mod_constexpr_definitions_and_wrapper(name: str, mod: 'hp.module.Module') -> (str, str):
            layer_name_and_type = f'{name}: {type(mod).__name__}'

            raw_constexpr_defs, raw_wrapper = mod.generate_model_parameters_c_code_constexpr_definitions()

            # 1. Generate constexpr definitions

            constexpr_def_header = '//' + '/' * 20 + '\n'
            constexpr_def_header += f'//   {layer_name_and_type}\n'
            constexpr_def_header += '//' + '/' * 20 + '\n' * 2
            constexpr_definitions = constexpr_def_header + raw_constexpr_defs

            # 2. Generate wrappers

            wrapper_header = f'// {layer_name_and_type}\n'
            wrapper = wrapper_header + raw_wrapper

            return constexpr_definitions, wrapper

        model_parameter_constexpr_definitions, wrapper = ['\n\n'.join(x) for x in
                                                          zip(*[gen_mod_constexpr_definitions_and_wrapper(n, m) for n, m
                                                                in
                                                                self.sequential_network.modules.items()])]

        return model_parameter_constexpr_definitions + '\n\n' + wrapper

    def generate_inference_function(self, append_final_softmax) -> (str, str):
        """
        Returns two strings. The first is the code for instantiating temporal matrices to be used during the
        """

        # 1. Intermediate storage

        # TODO: In the future, do a smarter allocation strategy. For example, use two arenas from which we instantiate
        #  matrices and ping-pong them, and clean the unused arena after each stage. For now, we will keep it simple.

        # 2. Inference steps.

        assert append_final_softmax == False

    @staticmethod
    def quantized_weights_to_packed_c_array(x: npt.NDArray[np.int8], identifier_name: str) -> (str, int):
        """
        Packs the quantized matrix x into an uint32_t array and returns the array definition code and the size of the
        flattened array. Note: the array is packed in a systolic-array friendly way: that is, any uint32_t will contain
        values from a single column. If the number of columns is not a multiple of 4, the last element of each row will
        be padded with 0s. The array is flattened in row-major order, and will contain uint32_t elements with the hex
        representation of 4 twos complement int8. The first element of the array will be in the highest bits of the
        uint32_t.
        """
        num_rows, num_cols = x.shape

        # Pad the cols to the minimum multiple of 4 >= num_cols
        num_padded_cols = ((num_cols + 3) // 4) * 4

        assert num_padded_cols % 4 == 0

        # Create and fill padded array with zeros
        padded_array = np.zeros((num_rows, num_padded_cols), dtype=np.int8)
        padded_array[:, :num_cols] = x
        # Array that contains all uint32_t
        packed_uint32_array = []

        def to_twos_complement_uint(val: int) -> int:
            assert -127 <= val <= 127

            if val >= 0:
                return val
            return val + (1 << 8)

        for r in range(num_rows):
            for c in range(0, num_cols, 4):
                four_weights = padded_array[r, c: c + 4]
                packed_val = 0
                for i, val in enumerate(four_weights):
                    packed_val |= (to_twos_complement_uint(int(val)) & 0xFF) << ((3 - i) * 8)
                packed_uint32_array.append(packed_val)

        array_size = len(packed_uint32_array)

        c_code = f"static constexpr uint32_t {identifier_name}[{array_size}] = {{\n    "
        hex_values = [f"0x{val:08X}" for val in packed_uint32_array]
        rows_of_values = [hex_values[i:i + num_padded_cols // 4] for i in
                          range(0, len(hex_values), num_padded_cols // 4)]
        c_code += ",\n    ".join([", ".join(row) for row in rows_of_values])
        c_code += "\n};"

        return c_code, len(packed_uint32_array)

    @staticmethod
    def bias_to_c_array(x: npt.NDArray[np.float32], identifier_name: str) -> (str, int):
        """
        Returns the C code for defining an array of float with the input bias and the size of that array.
        """

        # Bias should be a 1-dimensional array
        assert len(x.shape) == 1

        array_size = len(x)

        c_code = f"static constexpr float {identifier_name}[{array_size}] = {{\n    "

        # Use %.9g to maintain precision while avoiding unnecessary decimal places
        float_values = [f"{val:.9g}f" for val in x]
        rows_of_values = [float_values[i:i + 4] for i in range(0, len(float_values), 4)]
        c_code += ",\n    ".join([", ".join(row) for row in rows_of_values])
        c_code += "\n};"

        return c_code, array_size
