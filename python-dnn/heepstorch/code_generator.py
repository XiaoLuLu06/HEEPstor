from dataclasses import dataclass

import heepstorch as hp
import numpy as np
import numpy.typing as npt
import os
from pathlib import Path
from string import Template

SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
CODEGEN_TEMPLATE_DIR = SCRIPT_DIR / 'templates' / 'codegen'
HEEPSTOR_C_APPS_DIR = SCRIPT_DIR.parent.parent / 'sw' / 'applications'


@dataclass
class Buffer:
    name: str
    num_features: int


class CodeGenerator:

    def __init__(self, project_name: str, sequential_network: 'hp.module.SequentialNetwork'):
        self.project_name = project_name
        self.sequential_network = sequential_network

    @staticmethod
    def indent_lines(text: str, spaces: int = 4) -> str:
        """Indents all lines of input text by specified number of spaces"""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line else line for line in text.splitlines())

    def generate_code(self, append_final_softmax, overwrite_existing_generated_files=False):
        """
        Generates all necessary C code for inference in Heepstor, and writes it to the provided project.
        """

        def gen_mod_constexpr_definitions_and_wrapper(name: str, mod: 'hp.module.Module') -> (str, str):
            layer_name_and_type = f'{name}: {type(mod).__name__}'

            raw_constexpr_defs, raw_wrapper, raw_declarations = mod.generate_model_parameters_c_code_constexpr_definitions()

            # 1. Generate constexpr definitions

            if raw_constexpr_defs is None:
                constexpr_definitions = None
            else:
                constexpr_def_header = '//' + '/' * 20 + '\n'
                constexpr_def_header += f'//   {layer_name_and_type}\n'
                constexpr_def_header += '//' + '/' * 20 + '\n' * 2
                constexpr_definitions = constexpr_def_header + raw_constexpr_defs

            # 2. Generate wrappers

            if raw_wrapper is None:
                wrapper = None
            else:
                wrapper_header = f'// {layer_name_and_type}\n'
                wrapper = wrapper_header + raw_wrapper

            # 3. Generate declarations
            if raw_declarations is None:
                declarations = None
            else:
                declarations_wrapper = f'// {layer_name_and_type}\n'
                declarations = declarations_wrapper + raw_declarations

            return constexpr_definitions, wrapper, declarations

        model_parameter_constexpr_definitions, wrapper, declarations = \
            ['\n\n'.join([e for e in x if e is not None]) for x in zip(*[gen_mod_constexpr_definitions_and_wrapper(n, m)
                                                                         for n, m in
                                                                         self.sequential_network.modules.items()])]

        buffer_declarations, inference_steps, num_input_features, num_output_features = self.generate_inference_function(
            append_final_softmax)

        num_layers = len(self.sequential_network.modules) + 1 if append_final_softmax else 0
        perf_timer_layer_list = [f'{m.get_name()} ({type(m).__name__})' for m in
                                 self.sequential_network.modules.values()]

        if append_final_softmax:
            perf_timer_layer_list.append('Final Softmax')

        # Read template files
        model_hpp = (CODEGEN_TEMPLATE_DIR / 'model.hpp.tpl').read_text()
        model_parameters_hpp = (CODEGEN_TEMPLATE_DIR / 'model_parameters.hpp.tpl').read_text()
        model_parameters_cpp = (CODEGEN_TEMPLATE_DIR / 'model_parameters.cpp.tpl').read_text()

        # Create template substitutions
        model_hpp_template = Template(model_hpp)
        model_parameters_hpp_template = Template(model_parameters_hpp)
        model_parameters_cpp_template = Template(model_parameters_cpp)

        # Substitute values in templates
        model_hpp_indent_space_num = 8

        model_hpp_content = model_hpp_template.substitute(
            NUM_INPUT_FEATURES=num_input_features,
            NUM_OUTPUT_FEATURES=num_output_features,
            MODEL_PARAMETER_WRAPPERS=self.indent_lines(wrapper, model_hpp_indent_space_num),
            INTERMEDIATE_BUFFER_DECLARATIONS=self.indent_lines(buffer_declarations, model_hpp_indent_space_num),
            INFERENCE_STEPS=self.indent_lines(inference_steps, model_hpp_indent_space_num),
            NUM_LAYERS=num_layers,
            PERF_TIMER_LAYER_LIST=', '.join([f'"{x}"' for x in perf_timer_layer_list])
        )

        model_parameters_hpp_indent_space_num = 4

        model_parameters_hpp_content = model_parameters_hpp_template.substitute(
            MODEL_PARAMETER_CONSTEXPR_DEFINITIONS=self.indent_lines(model_parameter_constexpr_definitions,
                                                                    model_parameters_hpp_indent_space_num)
        )

        model_parameters_cpp_content = model_parameters_cpp_template.substitute(
            MODEL_PARAMETERS_DECLARATION=declarations
        )

        # Write templated files.

        project_dir = HEEPSTOR_C_APPS_DIR / self.project_name
        gen_dir = project_dir / 'gen'

        if project_dir.exists():
            if gen_dir.exists():
                if overwrite_existing_generated_files:
                    print(f'Overwriting existing generated files in {project_dir}...')
                else:
                    raise ValueError(
                        f'Project {project_dir} already exists and overwrite_existing_generated_files=False, aborting...')
            else:
                raise ValueError(
                    f"Project {project_dir} exists but has no gen directory. Aborting, as something seems wrong.")

        # Create directories if needed
        if not project_dir.exists():
            project_dir.mkdir()
            gen_dir.mkdir()

        # Write files
        (gen_dir / 'model.hpp').write_text(model_hpp_content)
        (gen_dir / 'model_parameters.hpp').write_text(model_parameters_hpp_content)
        (gen_dir / 'model_parameters.cpp').write_text(model_parameters_cpp_content)

    def generate_example_main(
            self,
            filename: str,
            input_matrix: npt.NDArray[np.float32],
            expected_output_matrix: npt.NDArray[np.float32],
            expected_predictions: list[int],
            true_label_values: list[int],
            overwrite_existing_generated_files: bool
    ):
        """
        Generates example main.cpp with test matrices and predictions
        """

        project_dir = HEEPSTOR_C_APPS_DIR / self.project_name

        if not project_dir.exists():
            raise ValueError(
                f"Project {project_dir} does not exist, please generate code before generating the example main.")

        dest_file = project_dir / filename

        if dest_file.exists():
            if overwrite_existing_generated_files:
                print(f'Overwriting existing example main in {dest_file}')
            else:
                raise ValueError(f"File {dest_file} exists and overwrite_existing_generated_files=False")

        def matrix_to_c_initializer(matrix: npt.NDArray[np.float32]) -> str:
            rows, cols = matrix.shape
            row_strs = []
            indent = ' ' * 8
            for i in range(rows):
                elements = [f"{x:.9g}f" for x in matrix[i]]
                row_strs.append(indent + "{" + ", ".join(elements) + "}")
            return "{\n" + ",\n".join(row_strs) + "\n    }"

        template = Template((CODEGEN_TEMPLATE_DIR / 'example_main.cpp.tpl').read_text())

        content = template.substitute(
            INPUT_MATRIX=matrix_to_c_initializer(input_matrix),
            EXPECTED_OUTPUT_MATRIX=matrix_to_c_initializer(expected_output_matrix),
            EXPECTED_PREDICTIONS=str(expected_predictions),
            TRUE_LABEL_VALUES=str(true_label_values),
            PROJECT_NAME=self.project_name,
        )

        dest_file.write_text(content)

    def generate_buffers(self) -> [Buffer]:
        """
        Generates name and sizes for all buffers, including input, output and intermediate buffers. Only generates
        intermediate buffers for operations not done in place.
        """

        # TODO: In the future, when we support convolutional neural networks, the buffer management system will have to
        #  be reworked if we still want to keep supporting only matrices. In that case, maybe we can force the inference
        #  to have a batch_size of 1, and use the number of rows instead to represent the different channels of the
        #  convolutional operation.

        # TODO: In the future, do a smarter allocation strategy. For example, use two arenas from which we instantiate
        #  matrices and ping-pong them, and clean the unused arena after each stage. For now, we will keep it simple.

        modules = list(self.sequential_network.modules.values())
        num_buffers = 1 + sum([0 if m.performs_inference_in_place() else 1 for m in modules])

        # For now, assume that there will always be a non-in-place operation.
        assert num_buffers >= 2

        buffers = [Buffer('inputs' if i == 0 else 'outputs' if i == num_buffers - 1 else f'intermediate_buf_{i}', None)
                   for i in range(num_buffers)]

        # Perform two different passes in order to propagate the number of feature of sizes. First a forward one, and
        #  then a backward one. Note that some layers such as ReLU can operate on a buffer of any size, so they depend
        #  on having another layer such as Linear that defines the actual sizing. If we cannot figure out the buffer
        #  sizes, return an error.

        # 1. Forward pass
        current_features = None
        buffer_idx = 0

        for module in modules:
            # Validate input features match if module specifies them
            if module.num_input_features() is not None:
                if current_features is not None and current_features != module.num_input_features():
                    raise ValueError(
                        f"Module {module.get_name()} ({type(module).__name__}) expects {module.num_input_features()} input features but receives {current_features}")
                current_features = module.num_input_features()

            if current_features is not None:
                buffers[buffer_idx].num_features = current_features

            if module.num_output_features() is not None:
                current_features = module.num_output_features()

            if not module.performs_inference_in_place():
                buffer_idx += 1

        # 2. Backward pass
        current_features = None
        buffer_idx = len(buffers) - 1

        for module in reversed(modules):
            if module.num_output_features() is not None:
                if current_features is not None and current_features != module.num_output_features():
                    raise ValueError(
                        f"Module {module.get_name()} ({type(module).__name__}) outputs {module.num_output_features()} features but next layer expects {current_features}")
                current_features = module.num_output_features()

            if current_features is not None:
                buffers[buffer_idx].num_features = current_features

            if module.num_input_features() is not None:
                current_features = module.num_input_features()

            if not module.performs_inference_in_place():
                assert buffer_idx > 0
                buffer_idx -= 1

        # Validate all buffers have sizes
        for buf in buffers:
            if buf.num_features is None:
                raise ValueError(f"Could not determine size for buffer {buf.name}")

        return buffers

    def generate_inference_function(self, append_final_softmax) -> (str, str, int, int):
        """
        Returns two strings and two integers. The first string is the code for instantiating intermediate buffers to be used
        during the inference, and the second string is the code for performing the inference. The first integer is the
        number of input features and the second integer is the number of output features.
        """

        # 1. Intermediate storage
        buffers = self.generate_buffers()
        buffer_declarations = []

        # Skip input and output declarations, which are arguments to the function.
        for buffer in buffers[1:-1]:
            # TODO: When implementing Conv2d layers, this will have to be changed.
            buffer_declarations.append(f'Matrix<float> {buffer.name}(BATCH_SIZE, {buffer.num_features});')

        buffer_declaration_c_code = '\n'.join(buffer_declarations)

        # 2. Inference steps.

        inference_steps = []

        buffer_idx = 0

        last_idx = None

        for i, (name, module) in enumerate(self.sequential_network.modules.items()):
            inference_steps.append(f'// {i + 1}. {name}: {type(module).__name__}')

            input_buffer = buffers[buffer_idx]
            output_buffer = buffers[buffer_idx] if module.performs_inference_in_place() else buffers[buffer_idx + 1]

            inference_steps.append(module.generate_inference_c_code(input_buffer.name, output_buffer.name))
            inference_steps.append('performance_timer.checkpoint();\n')

            if not module.performs_inference_in_place():
                buffer_idx += 1

            last_idx = i

        if append_final_softmax:
            inference_steps.append(f'// {last_idx + 2}. Final Softmax')
            inference_steps.append(f'Softmax::forward({buffers[-1].name});')
            inference_steps.append('performance_timer.checkpoint();')

        inference_steps_c_code = '\n'.join(inference_steps)

        return buffer_declaration_c_code, inference_steps_c_code, buffers[0].num_features, buffers[-1].num_features

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
