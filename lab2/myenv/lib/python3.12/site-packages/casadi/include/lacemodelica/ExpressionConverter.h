// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "ModelInfo.h"
#include "ONNXHelpers.hpp"
#include "BaseModelicaParser.h"
#include <string>
#include <vector>
#include <map>

namespace onnx {
    class GraphProto;
}

namespace lacemodelica {

// Modelica relational operators -> ONNX comparison operators
extern const std::map<std::string, std::string> kRelationalOpMap;

// Modelica math functions -> ONNX unary operators
extern const std::map<std::string, std::string> kMathFunctionMap;

/**
 * Converts Modelica expressions to ONNX graph nodes.
 *
 * This class implements a recursive descent converter that transforms
 * the Modelica expression AST into ONNX operations. Each convert method
 * returns the name of the output tensor containing the result.
 *
 * The converter handles:
 * - Arithmetic expressions (+, -, *, /, ^)
 * - Relational expressions (<, >, <=, >=, ==, <>)
 * - Logical expressions (and, or, not)
 * - If expressions (if-then-elseif-else)
 * - Function calls (built-in math functions, der(), user functions)
 * - Array subscripting with loop variable substitution
 */
class ExpressionConverter {
public:
    // Convert any expression context to ONNX, returning the output tensor name
    static std::string convert(antlr4::ParserRuleContext* expr, const ConversionContext& ctx);

    // Convert multi-output function calls, returning multiple tensor names
    static std::vector<std::string> convertMultiOutput(
        antlr4::ParserRuleContext* expr,
        const ConversionContext& ctx,
        size_t expectedOutputCount);

private:
    // Expression hierarchy converters (recursive descent)
    static std::string convertIfExpression(
        basemodelica::BaseModelicaParser::IfExpressionContext* expr,
        const ConversionContext& ctx);

    static std::string convertNestedIfElse(
        const std::vector<basemodelica::BaseModelicaParser::ExpressionNoDecorationContext*>& expressions,
        size_t startIdx,
        const ConversionContext& ctx);

    static std::string convertSimpleExpression(
        basemodelica::BaseModelicaParser::SimpleExpressionContext* expr,
        const ConversionContext& ctx);

    static std::string convertRelation(
        basemodelica::BaseModelicaParser::RelationContext* relation,
        const ConversionContext& ctx);

    static std::string convertArithmeticExpression(
        basemodelica::BaseModelicaParser::ArithmeticExpressionContext* expr,
        const ConversionContext& ctx);

    static std::string convertTerm(
        basemodelica::BaseModelicaParser::TermContext* expr,
        const ConversionContext& ctx);

    static std::string convertFactor(
        basemodelica::BaseModelicaParser::FactorContext* expr,
        const ConversionContext& ctx);

    static std::string convertPrimary(
        basemodelica::BaseModelicaParser::PrimaryContext* expr,
        const ConversionContext& ctx);

    // Specialized function call handlers
    static std::string convertDerCall(
        basemodelica::BaseModelicaParser::PrimaryContext* expr,
        const ConversionContext& ctx);

    static std::string convertUserFunctionCall(
        const std::string& funcName,
        basemodelica::BaseModelicaParser::PrimaryContext* expr,
        const Function* func,
        const ConversionContext& ctx);

    // Array constructor handlers
    static std::string convertArrayConstructor(
        basemodelica::BaseModelicaParser::ArrayArgumentsContext* arrayArgs,
        const ConversionContext& ctx);

    static std::string convertArrayLiteral(
        basemodelica::BaseModelicaParser::ArrayArgumentsContext* arrayArgs,
        const ConversionContext& ctx);

    static std::string convertArrayComprehension(
        basemodelica::BaseModelicaParser::ExpressionContext* bodyExpr,
        basemodelica::BaseModelicaParser::ForIndexContext* forIndex,
        const ConversionContext& ctx);

    // Reduction operation handlers (sum, product with for)
    static std::string convertReductionExpression(
        const std::string& reductionOp,
        basemodelica::BaseModelicaParser::ExpressionContext* bodyExpr,
        basemodelica::BaseModelicaParser::ForIndexContext* forIndex,
        const ConversionContext& ctx);
};

// Collect function arguments from recursive grammar structure
std::vector<basemodelica::BaseModelicaParser::ExpressionContext*>
collectFunctionArguments(basemodelica::BaseModelicaParser::FunctionArgumentsContext* funcArgs);

// Check if a variable is a 2D matrix
bool isMatrixVariable(const std::string& tensorName, const ModelInfo& info);

} // namespace lacemodelica
