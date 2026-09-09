// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "ModelInfo.h"
#include "ONNXHelpers.hpp"
#include "BaseModelicaParser.h"
#include <string>

namespace onnx {
    class GraphProto;
    class ModelProto;
}

namespace lacemodelica {

/**
 * Generates ONNX models from Modelica model information.
 *
 * This class coordinates the conversion of parsed Modelica models into
 * ONNX format. It delegates expression conversion to ExpressionConverter
 * and equation generation to EquationGenerator.
 *
 * Responsibilities:
 * - Model assembly (inputs, outputs, initializers)
 * - FunctionProto generation for Modelica functions
 * - Manifest generation for FMI layered standard
 * - Orchestrating the conversion pipeline
 */
class ONNXGenerator {
public:
    // Generate ONNX model file and layered standard manifest
    // Returns the directory path where files were generated
    static std::string generate(const ModelInfo& info, const std::string& outputDir);

    // Convert expression (delegates to ExpressionConverter for backward compatibility)
    static std::string convertExpression(antlr4::ParserRuleContext* expr, const ConversionContext& ctx);

private:
    static void generateONNXModel(const ModelInfo& info, const std::string& filepath);
    static void generateManifest(const std::string& filepath);

    // Create ONNX FunctionProto for a Modelica function with algorithm
    static void createFunctionProto(
        const Function& func,
        const ModelInfo& info,
        onnx::ModelProto* model);
};

} // namespace lacemodelica
