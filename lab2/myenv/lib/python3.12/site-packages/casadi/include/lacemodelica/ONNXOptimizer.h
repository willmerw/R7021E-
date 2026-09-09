// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

namespace onnx {
    class ModelProto;
    class GraphProto;
    class NodeProto;
}

namespace lacemodelica {

/**
 * ONNX model optimizer.
 *
 * Applies transformations to improve model efficiency.
 */
class ONNXOptimizer {
public:
    /**
     * Optimize the model by applying all available transformations.
     * Iterates depth-first over all nodes and applies transforms.
     * @param model The ONNX model to optimize (modified in place)
     * @return Number of transformations applied
     */
    static int optimize(onnx::ModelProto& model);

private:
    /**
     * Recursively optimize a graph (handles subgraphs in Loop/If nodes).
     */
    static int optimizeGraph(onnx::GraphProto& graph);

    /**
     * Attempt to convert a Loop node to a Scan node.
     * @param loopNode The Loop node to convert
     * @param graph The containing graph (modified if successful)
     * @return true if conversion was successful
     */
    static bool loopToScan(const onnx::NodeProto& loopNode, onnx::GraphProto& graph);
};

} // namespace lacemodelica
