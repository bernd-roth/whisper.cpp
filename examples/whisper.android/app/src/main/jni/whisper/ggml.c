#include <time.h>
//#include <type_traits>
#include <assert.h>
#include <android/NeuralNetworks.h>
#include "ggml.h"

void    ggml_time_init(void) {
}
int64_t ggml_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
int64_t ggml_cycles(void) {
    return clock();
}
int64_t ggml_cycles_per_ms(void) {
    return CLOCKS_PER_SEC/1000;
}
static const size_t GGML_TYPE_SIZE[GGML_TYPE_COUNT] = {
        sizeof(int8_t),
        sizeof(int16_t),
        sizeof(int32_t),
        sizeof(ggml_fp16_t),
        sizeof(float  ),
};
size_t ggml_type_size   (enum ggml_type type) {
    return GGML_TYPE_SIZE[type];
}
size_t ggml_nbytes   (const struct ggml_tensor * tensor) {
/*
     static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
     return ggml_nelements(tensor)*GGML_TYPE_SIZE[tensor->type];
*/
    // FIXME
}
size_t ggml_element_size(const struct ggml_tensor * tensor) {
    // FIXME
}

int    ggml_nelements(const struct ggml_tensor * tensor) {
    assert(GGML_MAX_DIMS == 4); // , "GGML_MAX_DIMS is not 4 - update this function");
    //return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}
struct ggml_context {
    ANeuralNetworksModel* model;
};
static struct ggml_context context1; // FIXME thread safe, multiple

/*

static const char * GGML_OP_LABEL[GGML_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "SUM",
    "MEAN",
    "REPEAT",
    "ABS",
    "SGN",
    "NEG",
    "STEP",
    "RELU",
    "GELU",
    "NORM",

    "MUL_MAT",

    "SCALE",
    "CPY",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "DIAG_MASK_INF",
    "SOFT_MAX",
    "ROPE",
    "CONV_1D_1S",
    "CONV_1D_2S",

    "FLASH_ATTN",
    "FLASH_FF",
};

static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {
    "none",

    "x",
    "x+y",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "Σx",
    "Σx/n",
    "repeat(x)",
    "abs(x)",
    "sgn(x)",
    "-x",
    "step(x)",
    "relu(x)",
    "gelu(x)",
    "norm(x)",

    "X*Y",

    "x*v",
    "x-\\>y",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "diag_mask_inf(x)",
    "soft_max(x)",
    "rope(x)",
    "conv_1d_1s(x)",
    "conv_1d_2s(x)",

    "flash_attn(x)",
    "flash_ff(x)",
};

 */


static std::pair<int, ANeuralNetworksMemory*> CreateASharedMemory(
        const char* name, uint32_t size, int prot) {
    int fd = ASharedMemory_create(name, size * sizeof(float));

    // Create an ANeuralNetworksMemory object from the corresponding ASharedMemory
    // objects.
    ANeuralNetworksMemory* memory = nullptr;
    int32_t status = ANeuralNetworksMemory_createFromFd(size * sizeof(float),
                                                        prot, fd, 0, &memory);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksMemory_createFromFd failed for %s",
                            name);
        close(fd);
        return {-1, nullptr};
    }

    return {fd, memory};
}

std::tie(initialStateFd_, memoryInitialState_) =
CreateASharedMemory("initialState", tensorSize_, PROT_READ);
std::tie(ratioFd_, memoryRatio_) =
CreateASharedMemory("ratio", tensorSize_, PROT_READ);
std::tie(sumInFd_, memorySumIn_) =
CreateASharedMemory("sumIn", tensorSize_, PROT_READ | PROT_WRITE);
std::tie(sumOutFd_, memorySumOut_) =
CreateASharedMemory("sumOut", tensorSize_, PROT_READ | PROT_WRITE);

#define CHECK_OPERATION_STATUS(description, action) \
  if (status != ANEURALNETWORKS_NO_ERROR) { \
    __android_log_print( \
        ANDROID_LOG_ERROR, LOG_TAG, \
        "ANeuralNetworksModel_" action " failed for operation " description, \
        operandId); \
    return NULL; \
  }

#define CHECK_OPERAND_STATUS(operandId, action) \
  if (status != ANEURALNETWORKS_NO_ERROR) { \
    __android_log_print( \
        ANDROID_LOG_ERROR, LOG_TAG, \
        "ANeuralNetworksModel_" action " failed for operand (%d)", \
        operandId); \
    return NULL; \
  }

#define ADD_OPERAND(type_description, operandId) \
  status = ANeuralNetworksModel_addOperand(model_, type_description); \
  uint32_t operandId = opIdx++; \
  CHECK_OPERAND_STATUS(operandId, "addOperand")

#define SET_OPERAND_VALUE(operandId, value) \
  status = ANeuralNetworksModel_setOperandValue( \
      model_, operandId, &value, \
      sizeof(value)); \
  CHECK_OPERAND_STATUS(operandId, "setOperandValue")

#define ADD_OPERATION(operator, input_operands, output_operands) \
  status = ANeuralNetworksModel_addOperation( \
    model_, ANEURALNETWORKS_ ## operator, \
    sizeof((uint32_t[]) input_operands)/sizeof(uint32_t), (uint32_t[]) input_operands, \
    sizeof((uint32_t[]) output_operands)/sizeof(uint32_t), sizeof((uint32_t[]) output_operands)/sizeof(uint32_t)); \
  CHECK_OPERATION_STATUS(""#operator, "addOperation")

/**
 * Compile the model.
 *
 * @return true for success, false otherwise
 */
bool CreateCompilation() {
    int32_t status;

    // Create the ANeuralNetworksCompilation object for the constructed model.
    status = ANeuralNetworksCompilation_create(model_, &compilation_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksCompilation_create failed");
        return false;
    }

    // Set the preference for the compilation_, so that the runtime and drivers
    // can make better decisions.
    // Here we prefer to get the answer quickly, so we choose
    // ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER.
    status = ANeuralNetworksCompilation_setPreference(
            compilation_, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksCompilation_setPreference failed");
        return false;
    }

    // Finish the compilation.
    status = ANeuralNetworksCompilation_finish(compilation_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksCompilation_finish failed");
        return false;
    }
    return true;
}

/**
 * Create a graph that consists of two operations: one addition and one
 * multiplication. This graph is used for computing a single step of
 * accumulating a geometric progression.
 *
 *   sumIn ---+
 *            +--- ADD ---> sumOut
 * stateIn ---+
 *            +--- MUL ---> stateOut
 *   ratio ---+
 *
 * The ratio is a constant tensor, defined in the model. It represents the
 * weights that would have been learned during a training process.
 *
 * The sumIn and stateIn are input tensors. Their values will be provided when
 * we execute the model. These values can change from execution to execution.
 *
 * To compute the sum of a geometric progression, the graph will be executed
 * multiple times with inputs and outputs chained together.
 *
 *                 +----------+   +----------+         +----------+
 *   initialSum -->| Simple   |-->| Simple   |-->   -->| Simple   |--> sumOut
 *                 | Sequence |   | Sequence |   ...   | Sequence |
 * initialState -->| Model    |-->| Model    |-->   -->| Model    |--> stateOut
 *                 +----------+   +----------+         +----------+
 *
 * @return true for success, false otherwise
 */

struct ggml_context * ggml_init(struct ggml_init_params params) {
    // FIXME make this function thread safe
    //ggml_critical_section_start();
    struct ggml_context* ctx = &context1; // FIXME thread safe, multiple
    ANeuralNetworksModel_create(&ctx->model);
    ANeuralNetworksModel* model_ = ctx->model;
    /* TODO:
     * int fd = open("training_data", O_RDONLY);
     * ANeuralNetworksMemory_createFromFd(file_size, PROT_READ, fd, 0, &mem1);
     */
    /*
     * // Configure and create AHardwareBuffer object
AHardwareBuffer_Desc desc = ...
AHardwareBuffer* ahwb = nullptr;
AHardwareBuffer_allocate(&desc, &ahwb);

// Create ANeuralNetworksMemory from AHardwareBuffer
ANeuralNetworksMemory* mem2 = NULL;
ANeuralNetworksMemory_createFromAHardwareBuffer(ahwb, &mem2);
     */
    // FIXME
    ANeuralNetworksOperandType float32TensorType = {
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = sizeof(dimensions) / sizeof(dimensions[0]),
            .dimensions = dimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    // FIXME
    ANeuralNetworksOperandType scalarInt32Type = {
            .type = ANEURALNETWORKS_INT32,
            .dimensionCount = 0,
            .dimensions = nullptr,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

// We first add the operand for the NONE activation function, and set its
// value to ANEURALNETWORKS_FUSED_NONE.
// This constant scalar operand will be used for both ADD and MUL.
/*
 *   sumIn ---+
 *            +--- ADD ---> sumOut
 * stateIn ---+
 *            +--- MUL ---> stateOut
 *   ratio ---+
 *
 */
    ADD_OPERAND(&scalarInt32Type, fusedActivationFuncNone)
    FuseCode fusedActivationCodeValue = ANEURALNETWORKS_FUSED_NONE;
    SET_OPERAND_VALUE(fusedActivationFuncNone, fusedActivationCodeValue)

    ADD_OPERAND(&float32TensorType, sumIn) // determined pre-execution
    ADD_OPERAND(&float32TensorType, stateIn) // determined pre-execution

// ratio is a constant tensor that was established during training.
// We read these values from the corresponding ANeuralNetworksMemory object.
    ADD_OPERAND(&float32TensorType, ratio)

    status = ANeuralNetworksModel_setOperandValueFromMemory(model_, ratio, memoryRatio_ /* ANeuralNetworksMemory */, 0, tensorSize_ * sizeof(float));
    CHECK_OPERAND_STATUS(ratio, "setOperandValueFromMemory")

// sumOut is the output of the ADD operation.
// Its value will be computed during execution.
    ADD_OPERAND(&float32TensorType, sumOut)

// stateOut is the output of the MUL operation.
// Its value will be computed during execution.
    ADD_OPERAND(&float32TensorType, stateOut)

    ADD_OPERATION(ADD, {sumIn, stateIn, fusedActivationFuncNone}, {sumOut})
    ADD_OPERATION(MUL, {stateIn, ratio, fusedActivationFuncNone}, {stateOut})


// Identify the input and output tensors to the model.
// Inputs: {sumIn, stateIn}
// Outputs: {sumOut, stateOut}
    std::vector<uint32_t> modelInputs = {
            sumIn,
            stateIn,
    };
    std::vector<uint32_t> modelOutputs = {
            sumOut,
            stateOut,
    };
    status = ANeuralNetworksModel_identifyInputsAndOutputs(
            model_, modelInputs.size(), modelInputs.data(), modelOutputs.size(),
            modelOutputs.data());
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_identifyInputsAndOutputs failed");
        return false;
    }

// Finish constructing the model.
// The values of constant operands cannot be altered after
// the finish function is called.
    status = ANeuralNetworksModel_finish(model_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_finish failed");
        return false;
    }
    return ctx;
}
struct ggml_tensor { // dupe

};


static OperandCode nnapi_tensor_type_from_ggml_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_I8:
            abort();
            return ANEURALNETWORKS_TENSOR_BOOL8; // FIXME
        case GGML_TYPE_I16:
            abort();
            return ANEURALNETWORKS_TENSOR_INT32; // FIXME
        case GGML_TYPE_I32:
            return ANEURALNETWORKS_TENSOR_INT32;
        case GGML_TYPE_F16:
            return ANEURALNETWORKS_TENSOR_FLOAT16;
        case GGML_TYPE_F32:
            return ANEURALNETWORKS_TENSOR_FLOAT32;
    }
};

struct ggml_tensor * ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0) {
    uint32_t dimensions[] = {ne0};
    ANeuralNetworksOperandType optype = {
            .type = nnapi_tensor_type_from_ggml_type(type),
            .dimensionCount = sizeof(dimensions) / sizeof(dimensions[0]),
            .dimensions = dimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    ADD_OPERAND(&optype, operandId)
    return operandId; // FIXME
}

struct ggml_tensor * ggml_new_tensor_2d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1) {
    uint32_t dimensions[] = {ne0, ne1};
    ANeuralNetworksOperandType optype = {
            .type = nnapi_tensor_type_from_ggml_type(type),
            .dimensionCount = sizeof(dimensions) / sizeof(dimensions[0]),
            .dimensions = dimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    ADD_OPERAND(&optype, operandId)
    return operandId; // FIXME
}

struct ggml_tensor * ggml_new_tensor_3d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1,
        int    ne2) {
    uint32_t dimensions[] = {ne0, ne1, ne2};
    ANeuralNetworksOperandType optype = {
            .type = nnapi_tensor_type_from_ggml_type(type),
            .dimensionCount = sizeof(dimensions) / sizeof(dimensions[0]),
            .dimensions = dimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    ADD_OPERAND(&optype, operandId)
    return operandId; // FIXME
}

void ggml_free(struct ggml_context * ctx) {
    // TODO
}
// padding = 1
// TODO: we don't support extra parameters for now
//       that's why we are hard-coding the stride, padding, and dilation
//       not great ..
struct ggml_tensor * ggml_conv_1d_1s(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {

    /*
     * ANEURALNETWORKS_CONV_2D
ANEURALNETWORKS_DEPTHWISE_CONV_2D
ANEURALNETWORKS_GROUPED_CONV_2D
ANEURALNETWORKS_TRANSPOSE_CONV_2D
     */
}
struct ggml_tensor * ggml_conv_1d_2s(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    /*
     * ANEURALNETWORKS_CONV_2D
ANEURALNETWORKS_DEPTHWISE_CONV_2D
ANEURALNETWORKS_GROUPED_CONV_2D
ANEURALNETWORKS_TRANSPOSE_CONV_2D
     */
}
// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
struct ggml_tensor * ggml_repeat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {

}
struct ggml_tensor * ggml_add(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {

}
// TODO: double-check this computation is correct
struct ggml_tensor * ggml_gelu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    //     return 0.5*x*(1.0 + tanh(SQRT_2_OVER_PI*x*(1.0 + GELU_COEF_A*x*x)));
    /*
     * inline static void ggml_vec_gelu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = table_gelu_f16[i16[i]];
    }
}
     */
    ANEURALNETWORKS_TANH

}
struct ggml_tensor * ggml_view_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        size_t                nb1, // row stride in bytes
        size_t                offset) {

}
/*struct ggml_tensor * ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3);
*/
// alias for ggml_permute(ctx, a, 1, 0, 2, 3)
struct ggml_tensor * ggml_transpose(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    ANEURALNETWORKS_TRANSPOSE
}

/*
 * struct ggml_tensor * ggml_norm_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        assert(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_NORM;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store epsilon here?

    return result;
}

 */
// normalize along rows
// TODO: eps is hardcoded to 1e-5 for now
struct ggml_tensor * ggml_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {

}

struct ggml_tensor * ggml_mul(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {

    /* not:    const int ne[4] = { a->ne[1], b->ne[1], a->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, MIN(a->n_dims, b->n_dims), ne); */

}
