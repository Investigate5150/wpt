// META: title=validation tests for WebNN API gru operation
// META: global=window,dedicatedworker
// META: script=../resources/utils_validation.js
// META: timeout=long

'use strict';

const batchSize = 3, inputSize = 4, hiddenSize = 5;

const tests = [
    {
        name: '[gruCell] Test with default options',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize,
        output: { dataType: 'float32', dimensions: [batchSize, hiddenSize] }
    },
    {
        name: '[gruCell] Test with given options',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize,
        options: {
            bias: { dataType: 'float32', dimensions: [3 * hiddenSize] },
            recurrentBias: { dataType: 'float32', dimensions: [3 * hiddenSize] },
            restAfter: true,
            layout: 'rzn',
            activations: ['sigmoid', 'relu']
        },
        output:
            { dataType: 'float32', dimensions: [batchSize, hiddenSize] }
    },
    {
        name: '[gruCell] TypeError is expected if hiddenSize equals to zero',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: 0
    },
    {
        name: '[gruCell] TypeError is expected if hiddenSize is too large',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: 4294967295,
    },
    {
        name:
            '[gruCell] TypeError is expected if the data type of the inputs is not one of the floating point types',
        input: { dataType: 'uint32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'uint32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'uint32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'uint32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name:
            '[gruCell] TypeError is expected if the rank of input is not 2',
        input: { dataType: 'float32', dimensions: [batchSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name:
            '[gruCell] TypeError is expected if the input.dimensions[1] is incorrect',
        input: { dataType: 'float32', dimensions: [inputSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name: '[gruCell] TypeError is expected if data type of weight is not one of the floating point types',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'int8',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name: '[gruCell] TypeError is expected if rank of weight is not 2',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name: '[gruCell] TypeError is expected if weight.dimensions[0] is not 3 * hiddenSize',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [4 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name: '[gruCell] TypeError is expected if data type of recurrentWeight is not one of the floating point types',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'int32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name:
            '[gruCell] TypeError is expected if the rank of recurrentWeight is not 2',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight:
            { dataType: 'float32', dimensions: [3 * hiddenSize] },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name:
            '[gruCell] TypeError is expected if the recurrentWeight.dimensions is invalid',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight:
            { dataType: 'float32', dimensions: [4 * hiddenSize, inputSize] },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name:
            '[gruCell] TypeError is expected if data type of hiddenState is not one of the floating point types',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight:
            { dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize] },
        hiddenState: {
            dataType: 'uint32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name:
            '[gruCell] TypeError is expected if the rank of hiddenState is not 2',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight:
            { dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize] },
        hiddenState: {
            dataType: 'float32',
            dimensions: [hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name:
            '[gruCell] TypeError is expected if the hiddenState.dimensions is invalid',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight:
            { dataType: 'float32', dimensions: [3 * hiddenSize, inputSize] },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, 3 * hiddenSize]
        },
        hiddenSize: hiddenSize
    },
    {
        name:
            '[gruCell] TypeError is expected if the size of options.activations is not 2',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize,
        options: { activations: ['sigmoid', 'tanh', 'relu'] }
    },
    {
        name:
            '[gruCell] TypeError is expected if data type of options.bias is not one of the floating point types',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize,
        options: { bias: { dataType: 'uint8', dimensions: [3 * hiddenSize] } }
    },
    {
        name:
            '[gruCell] TypeError is expected if the rank of options.bias is not 1',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize,
        options: { bias: { dataType: 'float32', dimensions: [batchSize, 3 * hiddenSize] } }
    },
    {
        name:
            '[gruCell] TypeError is expected if options.bias.dimensions[0] is not 3 * hiddenSize',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize,
        options: { bias: { dataType: 'float32', dimensions: [3 * hiddenSize] } }
    },
    {
        name:
            '[gruCell] TypeError is expected if data type of options.recurrentBias is not one of the floating point types',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize,
        options: { recurrentBias: { dataType: 'int8', dimensions: [3 * hiddenSize] } }
    },
    {
        name:
            '[gruCell] TypeError is expected if the rank of options.recurrentBias is not 1',
        input: { dataType: 'float32', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float32',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize,
        options: { recurrentBias: { dataType: 'float32', dimensions: [batchSize, 3 * hiddenSize] } }
    },
    {
        name:
            '[gruCell] TypeError is expected if options.recurrentBias.dimensions[0] is not 3 * hiddenSize',
        input: { dataType: 'float16', dimensions: [batchSize, inputSize] },
        weight: {
            dataType: 'float16',
            dimensions: [3 * hiddenSize, inputSize]
        },
        recurrentWeight: {
            dataType: 'float16',
            dimensions: [3 * hiddenSize, hiddenSize]
        },
        hiddenState: {
            dataType: 'float32',
            dimensions: [batchSize, hiddenSize]
        },
        hiddenSize: hiddenSize,
        options: {
            recurrentBias: { dataType: 'float16', dimensions: [4 * hiddenSize] }
        }
    }
];

tests.forEach(
    test => promise_test(async t => {
        const input = builder.input(
            'input',
            { dataType: test.input.dataType, dimensions: test.input.dimensions });
        const weight = builder.input(
            'weight',
            { dataType: test.weight.dataType, dimensions: test.weight.dimensions });
        const recurrentWeight = builder.input('recurrentWeight', {
            dataType: test.recurrentWeight.dataType,
            dimensions: test.recurrentWeight.dimensions
        });
        const hiddenState = builder.input('hiddenState', {
            dataType: test.hiddenState.dataType,
            dimensions: test.hiddenState.dimensions
        });

        const options = {};
        if (test.options) {
            if (test.options.bias) {
                options.bias = builder.input('bias', {
                    dataType: test.options.bias.dataType,
                    dimensions: test.options.bias.dimensions
                });
            }
            if (test.options.recurrentBias) {
                options.bias = builder.input('recurrentBias', {
                    dataType: test.options.recurrentBias.dataType,
                    dimensions: test.options.recurrentBias.dimensions
                });
            }
            if (test.options.resetAfter) {
                options.resetAfter = test.options.resetAfter;
            }
            if (test.options.layout) {
                options.layout = test.options.layout;
            }
            if (test.options.activations) {
                options.activations = [];
                test.options.activations.forEach(
                    activation => options.activations.push(builder[activation]()));
            }
        }

        if (test.output) {
            const output = builder.gruCell(
                input, weight, recurrentWeight, hiddenState, test.hiddenSize,
                options);
            assert_equals(output.dataType(), test.output.dataType);
            assert_array_equals(output.shape(), test.output.dimensions);
        } else {
            assert_throws_js(
                TypeError,
                () => builder.gruCell(
                    input, weight, recurrentWeight, hiddenState, test.hiddenSize,
                    options));
        }
    }, test.name));