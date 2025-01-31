from dataclasses import dataclass
from typing import Optional, Tuple, List

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
    channels: int
    mode: 'hp.module.NetworkMode'
    image_dims: Optional['hp.module.ImageDimensions'] = None

    def get_declaration_code(self) -> str:
        """Returns C++ code to declare this buffer based on mode"""
        if self.mode == hp.module.NetworkMode.CONV:
            if self.image_dims is None:
                raise ValueError(f"Buffer {self.name} in CONV mode but has no image dimensions")
            # For conv mode: fixed dimensions based on image size
            return f'Matrix<float> {self.name}({self.image_dims.height}*{self.image_dims.width}, {self.channels});'
        else:
            # For linear mode: dynamic batch size from input
            return f'Matrix<float> {self.name}(batch_size, {self.channels});'


class CodeGenerator:

    def __init__(self, project_name: str, sequential_network: 'hp.module.SequentialNetwork'):
        self.project_name = project_name
        self.sequential_network = sequential_network

    @staticmethod
    def indent_lines(text: str, spaces: int = 4) -> str:
        """Indents all lines of input text by specified number of spaces"""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line else line for line in text.splitlines())

    def generate_code(self, append_final_softmax: bool, overwrite_existing_generated_files: bool = False):
        """
        Generates all necessary C code for inference in Heepstor, and writes it to the provided project.
        """

        def gen_mod_constexpr_definitions_and_wrapper(name: str, mod: 'hp.module.Module') -> Tuple[
            Optional[str], Optional[str], Optional[str]]:
            layer_name_and_type = f'{name}: {type(mod).__name__}'
            raw_constexpr_defs, raw_wrapper, raw_declarations = mod.generate_model_parameters_c_code_constexpr_definitions()

            if raw_constexpr_defs is None:
                return None, None, None

            constexpr_def_header = '//' + '/' * 20 + '\n'
            constexpr_def_header += f'//   {layer_name_and_type}\n'
            constexpr_def_header += '//' + '/' * 20 + '\n' * 2

            wrapper_header = f'// {layer_name_and_type}\n'
            declarations_header = f'// {layer_name_and_type}\n'

            return (
                constexpr_def_header + raw_constexpr_defs,
                wrapper_header + raw_wrapper,
                declarations_header + raw_declarations
            )

        # Generate code for all modules
        modules_code = [
            gen_mod_constexpr_definitions_and_wrapper(n, m)
            for n, m in self.sequential_network.modules.items()
        ]

        model_parameter_constexpr_definitions = '\n\n'.join(
            [c for c, _, _ in modules_code if c is not None])
        wrapper = '\n\n'.join([w for _, w, _ in modules_code if w is not None])
        declarations = '\n\n'.join([d for _, _, d in modules_code if d is not None])

        # Generate inference code
        buffer_declarations, inference_steps, input_info, output_info = self.generate_inference_function(
            append_final_softmax)

        # Handle performance timer configuration
        num_layers = len(self.sequential_network.modules) + (1 if append_final_softmax else 0)
        perf_timer_layer_list = [
            f'{m.get_name()} ({type(m).__name__})'
            for m in self.sequential_network.modules.values()
        ]
        if append_final_softmax:
            perf_timer_layer_list.append('Final Softmax')

        # Read templates
        model_hpp = (CODEGEN_TEMPLATE_DIR / 'model.hpp.tpl').read_text()
        model_parameters_hpp = (CODEGEN_TEMPLATE_DIR / 'model_parameters.hpp.tpl').read_text()
        model_parameters_cpp = (CODEGEN_TEMPLATE_DIR / 'model_parameters.cpp.tpl').read_text()

        # Create substitutions
        model_hpp_template = Template(model_hpp)
        model_parameters_hpp_template = Template(model_parameters_hpp)
        model_parameters_cpp_template = Template(model_parameters_cpp)

        # Check if network supports batching
        supports_batching = all(
            m.network_mode() == hp.module.NetworkMode.LINEAR
            for m in self.sequential_network.modules.values()
        )

        validation_code = """HEEPSTOR_ASSERT(inputs.num_cols() == NUM_INPUT_FEATURES);
HEEPSTOR_ASSERT(outputs.num_cols() == NUM_OUTPUT_FEATURES);
HEEPSTOR_ASSERT(inputs.num_rows() == outputs.num_rows());

const size_t batch_size = inputs.num_rows();""" if supports_batching else """HEEPSTOR_ASSERT(inputs.num_rows() == INPUT_HEIGHT * INPUT_WIDTH);
HEEPSTOR_ASSERT(inputs.num_cols() == NUM_INPUT_CHANNELS);
HEEPSTOR_ASSERT(outputs.num_cols() == NUM_OUTPUT_FEATURES);
HEEPSTOR_ASSERT(outputs.num_rows() == 1);

const size_t batch_size = 1;"""

        # Generate content
        model_hpp_content = model_hpp_template.substitute(
            INPUT_INFO=self.indent_lines(input_info, 4),
            OUTPUT_INFO=self.indent_lines(output_info, 4),
            VALIDATION_CODE=self.indent_lines(validation_code, 8),
            MODEL_PARAMETER_WRAPPERS=self.indent_lines(wrapper, 8),
            INTERMEDIATE_BUFFER_DECLARATIONS=self.indent_lines(buffer_declarations, 8),
            INFERENCE_STEPS=self.indent_lines(inference_steps, 8),
            NUM_LAYERS=num_layers,
            PERF_TIMER_LAYER_LIST=', '.join([f'"{x}"' for x in perf_timer_layer_list])
        )

        model_parameters_hpp_content = model_parameters_hpp_template.substitute(
            MODEL_PARAMETER_CONSTEXPR_DEFINITIONS=self.indent_lines(
                model_parameter_constexpr_definitions, 4)
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

    def generate_buffers(self) -> List[Buffer]:
        """
        Generates name and sizes for all buffers, including input, output and intermediate buffers. Only generates
        intermediate buffers for operations not done in place.
        """

        modules = list(self.sequential_network.modules.values())
        current_mode = (hp.module.NetworkMode.CONV
                        if self.sequential_network.input_dimensions
                        else hp.module.NetworkMode.LINEAR)
        current_channels = None
        current_dims = self.sequential_network.input_dimensions

        buffers = []

        # Input buffer
        buffers.append(Buffer("inputs",
                              modules[0].num_input_channels() or 1,
                              current_mode,
                              current_dims))

        # Generate intermediate buffers
        for i, module in enumerate(modules):
            if module.num_input_channels() is not None:
                current_channels = module.num_input_channels()

            # Update dimensions for CONV mode
            if current_mode == hp.module.NetworkMode.CONV:
                current_dims = module.output_image_dimensions(current_dims, None)

            # Handle CONV -> LINEAR transitions
            if isinstance(module, hp.module.Flatten):
                current_mode = hp.module.NetworkMode.LINEAR
                current_dims = None

            # Create new buffer if needed
            if not module.performs_inference_in_place():
                buffer_name = f"intermediate_buf_{len(buffers)}" if i < len(modules) - 1 else "outputs"
                buffers.append(Buffer(
                    buffer_name,
                    current_channels if module.num_output_channels() is None else module.num_output_channels(),
                    current_mode,
                    current_dims
                ))

            if module.num_output_channels() is not None:
                current_channels = module.num_output_channels()

        return buffers

    def generate_inference_function(self, append_final_softmax: bool) -> Tuple[str, str, str, str]:
        """Generate C++ inference code and configuration"""
        buffers = self.generate_buffers()

        # Network is batch-capable only if all layers after first are LINEAR mode
        supports_batching = all(b.mode == hp.module.NetworkMode.LINEAR for b in buffers[1:])

        # Generate input/output configuration
        if supports_batching:
            input_info = f"static constexpr size_t NUM_INPUT_FEATURES = {buffers[0].channels};"
        else:
            input_info = f"""static constexpr size_t NUM_INPUT_CHANNELS = {buffers[0].channels};
static constexpr size_t INPUT_HEIGHT = {buffers[0].image_dims.height};
static constexpr size_t INPUT_WIDTH = {buffers[0].image_dims.width};"""

        output_info = f"static constexpr size_t NUM_OUTPUT_FEATURES = {buffers[-1].channels};"

        # Generate buffer declarations, skipping input/output buffers
        buffer_declarations = []
        for buffer in buffers[1:-1]:
            buffer_declarations.append(buffer.get_declaration_code())

        # Generate inference steps
        inference_steps = []
        buffer_idx = 0
        for i, (name, module) in enumerate(self.sequential_network.modules.items()):
            inference_steps.append(f'// {i + 1}. {name}: {type(module).__name__}')

            input_buffer = buffers[buffer_idx]
            output_buffer = buffers[buffer_idx] if module.performs_inference_in_place() else buffers[buffer_idx + 1]

            inference_steps.append(module.generate_inference_c_code(
                input_buffer.name, output_buffer.name))
            inference_steps.append('performance_timer.checkpoint();\n')

            if not module.performs_inference_in_place():
                buffer_idx += 1

        if append_final_softmax:
            inference_steps.append(f'// {len(self.sequential_network.modules) + 1}. Final Softmax')
            inference_steps.append(f'Softmax::forward({buffers[-1].name});')
            inference_steps.append('performance_timer.checkpoint();')

        return (
            '\n'.join(buffer_declarations),
            '\n'.join(inference_steps),
            input_info,
            output_info
        )

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


def flatten_input_to_matrix(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
        Flattens the input tensor (with possible multiple channels) to a 2-d matrix to be
        passed as input to the generated C code for executing convolutions with im2col transformations.
    Args:
        x: input tensor of shape [BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]. Note that BATCH_SIZE must be 1.
            or tensor of shape [NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]
    Returns:
        2-d matrix of shape [IMAGE_HEIGHT * IMAGE_WIDTH, NUM_CHANNELS]
    """

    if len(x.shape) == 3:
        NUM_CHANNELS, H, W = x.shape
    else:
        BATCH_SIZE, NUM_CHANNELS, H, W = x.shape
        assert BATCH_SIZE == 1

    return x.reshape(NUM_CHANNELS, H * W).T
