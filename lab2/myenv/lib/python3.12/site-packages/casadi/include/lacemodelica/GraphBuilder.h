// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "BaseModelicaParser.h"

namespace onnx {
    class GraphProto;
    class NodeProto;
    class TensorShapeProto;
}

namespace lacemodelica {

/**
 * A fluent builder for constructing ONNX graphs.
 *
 * GraphBuilder provides a clean abstraction over the low-level protobuf
 * manipulation required for ONNX graph construction. It owns the node
 * counter state and provides methods for:
 *
 * - Creating tensor names with automatic unique numbering
 * - Adding constant nodes (int64, double, bool, arrays)
 * - Adding unary and binary operation nodes
 * - Adding control flow nodes (If, Loop)
 * - Adding graph inputs and outputs
 * - Creating subgraph builders for branches
 *
 * Example usage:
 *   GraphBuilder builder(graph, counter);
 *   auto left = builder.addDoubleConstant(1.0);
 *   auto right = builder.addDoubleConstant(2.0);
 *   auto result = builder.addBinaryOp("Add", left, right);
 */
class GraphBuilder {
public:
    GraphBuilder(onnx::GraphProto* graph, int& nodeCounter);
    GraphBuilder(onnx::GraphProto* graph, int& nodeCounter, const std::string& prefix);

    // Accessors
    onnx::GraphProto* graph() const { return graph_; }
    int& counter() { return nodeCounter_; }
    const std::string& prefix() const { return prefix_; }

    // Create a subgraph builder (shares counter, different graph and prefix)
    GraphBuilder forSubgraph(onnx::GraphProto* subgraph, const std::string& subPrefix = "") const;

    // Create a builder with a different prefix (same graph and counter)
    GraphBuilder withPrefix(const std::string& newPrefix) const;

    // --- Tensor naming ---

    std::string makeTensorName();
    std::string makeTensorName(const std::string& hint);

    // --- Constant creation ---

    std::string addInt64Constant(int64_t value);
    std::string addInt64Constant(int64_t value, const std::string& nameHint);
    std::string addDoubleConstant(double value);
    std::string addBoolConstant(bool value);
    std::string addInt64ArrayConstant(const std::vector<int64_t>& values);
    std::string addDoubleArrayConstant(const std::vector<double>& values);
    std::string addDoubleZerosConstant(const std::vector<int64_t>& shape);

    // --- Operation nodes ---

    std::string addUnaryOp(const std::string& opType, const std::string& input);
    std::string addBinaryOp(const std::string& opType,
                            const std::string& left,
                            const std::string& right);

    // --- Indexing operations ---

    std::string addGather(const std::string& data, const std::string& indices, int axis = 0);
    std::string addGatherND(const std::string& data, const std::vector<int64_t>& indices);
    std::string addScatterND(const std::string& data,
                             const std::vector<int64_t>& indices,
                             const std::string& updates);

    // Scatter multiple elements at given 1D indices (for range assignments)
    // indices are 0-based positions, updates should have shape [n] matching indices count
    std::string addScatterND1D(const std::string& data,
                                const std::vector<int64_t>& indices1D,
                                const std::string& updates);

    // Slice operation for range subscripts (0-based, exclusive end)
    std::string addSlice(const std::string& data,
                         const std::vector<int64_t>& starts,
                         const std::vector<int64_t>& ends,
                         const std::vector<int64_t>& axes);

    // Squeeze operation to remove dimensions of size 1
    std::string addSqueeze(const std::string& input, const std::vector<int64_t>& axes);

    // Concatenate tensors along an axis
    std::string addConcat(const std::vector<std::string>& inputs, int64_t axis);

    // Transpose with permutation
    std::string addTranspose(const std::string& input, const std::vector<int64_t>& perm);

    // Convert Modelica 1-based index to ONNX 0-based
    std::string convertToZeroBasedIndex(const std::string& oneBasedTensor);

    // Reshape a scalar to 1D for scatter operations
    std::string addUnsqueeze(const std::string& input, const std::vector<int64_t>& axes);

    // --- Control flow ---

    std::string addIfNode(const std::string& condition,
                          onnx::GraphProto& thenBranch,
                          onnx::GraphProto& elseBranch,
                          const std::string& nameHint = "If");

    // --- Identity/renaming ---

    std::string addIdentity(const std::string& input, const std::string& outputName,
                            const std::string& nodeName);

    // Rename a tensor by finding its producer node and changing the output name,
    // or creating an Identity if the tensor is a direct input.
    // Returns true if producer was found and renamed.
    bool renameTensor(const std::string& tensorName, const std::string& newName,
                      const std::string& identityPrefix);

    // --- Graph structure ---

    void addScalarDoubleOutput(const std::string& tensorName);
    void addShapeDimensions(onnx::TensorShapeProto* shape,
                            const std::vector<std::string>& dimensions);

    // --- Array subscript handling ---

    struct SubscriptAnalysis {
        bool hasLoopVariable = false;
        std::vector<int64_t> staticIndices;
    };

    SubscriptAnalysis analyzeSubscripts(
        const std::vector<basemodelica::BaseModelicaParser::SubscriptContext*>& subscripts,
        const std::map<std::string, std::string>* variableMap);

    std::string applySubscripts(
        const std::string& baseTensor,
        const std::vector<basemodelica::BaseModelicaParser::SubscriptContext*>& subscripts,
        const std::map<std::string, std::string>* variableMap);

    // --- Loop helpers ---

    void addLoopPassthrough(
        onnx::NodeProto* loopNode,
        onnx::GraphProto* bodyGraph,
        const std::string& loopNodeName,
        const std::string& inputName,
        const std::string& bodyInputName,
        int elemType,
        const std::vector<std::string>& dimensions,
        const std::string& outputSuffix);

private:
    onnx::GraphProto* graph_;
    int& nodeCounter_;
    std::string prefix_;
};

} // namespace lacemodelica
