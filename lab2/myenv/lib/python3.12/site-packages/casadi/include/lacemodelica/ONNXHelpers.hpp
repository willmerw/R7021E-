// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include <string>
#include <vector>
#include <map>
#include "BaseModelicaParser.h"
#include "GraphBuilder.h"

// Forward declarations
namespace onnx {
    class GraphProto;
    class NodeProto;
    class TypeProto_Tensor;
    class ValueInfoProto;
    class TensorShapeProto;
}

namespace lacemodelica {

class ModelInfo;  // Forward declaration

/**
 * Context object bundling all parameters needed for expression conversion.
 *
 * ConversionContext provides a unified interface for code that converts
 * Modelica expressions to ONNX. It combines:
 *
 * - Model information (variable types, dimensions, functions)
 * - A GraphBuilder for constructing ONNX nodes
 * - Variable mappings (for loop variables, function parameters)
 * - Derivative tracking (for der() calls that need inputs)
 *
 * Child contexts can be created for subgraphs (If/Loop branches) using
 * withGraph() and withPrefix() methods, sharing the node counter.
 */
struct ConversionContext {
    const ModelInfo& info;
    onnx::GraphProto* graph;
    int& nodeCounter;
    const std::map<std::string, std::string>* variableMap;
    std::map<std::string, std::vector<std::string>>* derivativeInputs;
    std::string tensorPrefix;

    ConversionContext(
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = "")
        : info(info)
        , graph(graph)
        , nodeCounter(nodeCounter)
        , variableMap(variableMap)
        , derivativeInputs(derivativeInputs)
        , tensorPrefix(tensorPrefix)
    {}

    // Create a child context with a different graph (for subgraphs)
    ConversionContext withGraph(onnx::GraphProto* newGraph) const {
        return ConversionContext(info, newGraph, nodeCounter, variableMap, derivativeInputs, tensorPrefix);
    }

    // Create a child context with a different prefix
    ConversionContext withPrefix(const std::string& newPrefix) const {
        return ConversionContext(info, graph, nodeCounter, variableMap, derivativeInputs, newPrefix);
    }

    // Get a GraphBuilder for this context
    GraphBuilder builder() const {
        return GraphBuilder(graph, const_cast<int&>(nodeCounter), tensorPrefix);
    }
};

// Generate a unique tensor name with optional prefix
std::string makeTensorName(const std::string& prefix, int& counter);

// Create ONNX Constant nodes for scalar values
std::string createInt64Constant(onnx::GraphProto* graph, int64_t value, int& counter,
                                 const std::string& nameHint = "");
std::string createDoubleConstant(onnx::GraphProto* graph, double value, int& counter);
std::string createBoolConstant(onnx::GraphProto* graph, bool value, int& counter);

// Create an Identity node to rename/copy a tensor
std::string createIdentityNode(onnx::GraphProto* graph, const std::string& inputTensor,
                                const std::string& outputName, const std::string& nodeName);

// Convert Modelica 1-based index to ONNX 0-based index
std::string convertTo0BasedIndex(onnx::GraphProto* graph, const std::string& oneBasedTensor,
                                  int& counter, const std::string& prefix = "");

// Create a Gather node for dynamic array indexing
std::string createGatherNode(onnx::GraphProto* graph, const std::string& dataTensor,
                              const std::string& indexTensor, int axis, int& counter,
                              const std::string& prefix = "");

// Create binary operation nodes (Add, Sub, Mul, Div, Pow, etc.)
std::string createBinaryOp(onnx::GraphProto* graph, const std::string& opType,
                            const std::string& left, const std::string& right,
                            int& counter, const std::string& prefix = "");

// Create unary operation nodes (Neg, Not, Sin, Cos, etc.)
std::string createUnaryOp(onnx::GraphProto* graph, const std::string& opType,
                           const std::string& input, int& counter,
                           const std::string& prefix = "");

// Add a scalar double output to a subgraph (used for If branches)
void addScalarDoubleOutput(onnx::GraphProto* graph, const std::string& tensorName);

// Create an If node with then/else branch subgraphs
std::string createIfNode(onnx::GraphProto* graph, const std::string& condTensor,
                          onnx::GraphProto& thenBranch, onnx::GraphProto& elseBranch,
                          int& counter, const std::string& prefix = "",
                          const std::string& nameHint = "If");

// Create a Constant node with an int64 array (for indices)
std::string createInt64ArrayConstant(onnx::GraphProto* graph, const std::vector<int64_t>& values, int& counter);

// Create a GatherND node for multi-dimensional static indexing
std::string createGatherNDNode(onnx::GraphProto* graph, const std::string& dataTensor,
                                const std::vector<int64_t>& indices, int& counter,
                                const std::string& prefix = "");

// Add dimensions to a tensor shape, parsing numeric dimensions as values
// and treating non-numeric ones as symbolic parameters
void addShapeDimensions(onnx::TensorShapeProto* shape, const std::vector<std::string>& dimensions);

// Result of checking subscripts for loop variables
struct SubscriptAnalysis {
    bool hasLoopVariable = false;
    std::vector<int64_t> staticIndices;  // Only valid when !hasLoopVariable
};

// Analyze array subscripts to determine if they contain loop variables
// Returns analysis with hasLoopVariable flag and staticIndices if all are static
SubscriptAnalysis analyzeSubscripts(
    const std::vector<basemodelica::BaseModelicaParser::SubscriptContext*>& subscriptList,
    const std::map<std::string, std::string>* variableMap);

// Apply array subscripts to a tensor, handling both static and dynamic indexing
// Returns the resulting tensor name after all subscripts are applied
std::string applyArraySubscripts(
    onnx::GraphProto* graph,
    const std::string& baseTensor,
    const std::vector<basemodelica::BaseModelicaParser::SubscriptContext*>& subscriptList,
    const std::map<std::string, std::string>* variableMap,
    int& nodeCounter,
    const std::string& tensorPrefix);

// Rename a tensor to a desired output name by either:
// 1. Finding and renaming the producer node's output, or
// 2. Creating an Identity node if no producer found (tensor is a direct input)
// Returns true if producer was found and renamed, false if Identity was created
bool renameTensorToOutput(
    onnx::GraphProto* graph,
    const std::string& tensorName,
    const std::string& outputName,
    const std::string& identityNamePrefix);

// Add a loop-carried passthrough variable to an ONNX Loop body.
// This creates the input/output/identity pattern required for loop variables
// that need to be accessible inside the loop body but aren't modified.
// Parameters:
//   loopNode: The Loop node to add input/output to
//   bodyGraph: The loop body subgraph
//   loopNodeName: Name prefix for generated nodes
//   inputName: Name of the input tensor (added to loop inputs)
//   bodyInputName: Name for the variable inside loop body
//   elemType: ONNX element type (e.g., TensorProto::DOUBLE)
//   dimensions: Tensor dimensions (empty for scalar)
//   outputSuffix: Suffix for loop output name
void addLoopPassthrough(
    onnx::NodeProto* loopNode,
    onnx::GraphProto* bodyGraph,
    const std::string& loopNodeName,
    const std::string& inputName,
    const std::string& bodyInputName,
    int elemType,
    const std::vector<std::string>& dimensions,
    const std::string& outputSuffix);

} // namespace lacemodelica
