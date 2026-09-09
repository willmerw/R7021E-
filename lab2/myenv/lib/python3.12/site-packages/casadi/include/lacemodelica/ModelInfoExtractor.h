// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "BaseModelicaParser.h"
#include "ModelInfo.h"
#include <set>

namespace lacemodelica {

class ModelInfoExtractor {
public:
    ModelInfo extract(basemodelica::BaseModelicaParser::BaseModelicaContext* tree, const std::string& sourceFile = "");

private:
    ModelInfo info;
    std::string sourceFile;
    std::set<std::string> derivativeCalls;  // Track der() calls
    std::map<std::string, std::vector<Variable>> recordDefinitions;  // Record type -> field variables

    void extractPackageAndModelName(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void extractRecordDefinitions(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void extractEnumDefinitions(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void extractGlobalConstants(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void extractVariables(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void extractEquations(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void extractFunctions(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void identifyDerivatives();

    void processEquation(basemodelica::BaseModelicaParser::EquationContext* equation, std::vector<Equation>& target);

    // Scan equation text for der() calls and add to derivativeCalls set
    void scanForDerivativeCalls(const std::string& equationText);

    // Helper to find named attribute in modification's class modification argument list
    antlr4::tree::ParseTree* findModificationAttribute(basemodelica::BaseModelicaParser::ModificationContext* ctx, const std::string& attrName);

    std::string extractStartValue(basemodelica::BaseModelicaParser::ModificationContext* ctx);
    std::string extractMinValue(basemodelica::BaseModelicaParser::ModificationContext* ctx);
    std::string extractMaxValue(basemodelica::BaseModelicaParser::ModificationContext* ctx);
    antlr4::ParserRuleContext* extractBindingContext(basemodelica::BaseModelicaParser::ModificationContext* ctx);
    antlr4::ParserRuleContext* extractMinContext(basemodelica::BaseModelicaParser::ModificationContext* ctx);
    antlr4::ParserRuleContext* extractMaxContext(basemodelica::BaseModelicaParser::ModificationContext* ctx);
    bool isConstExpression(const std::string& expr);
    std::string extractDescription(basemodelica::BaseModelicaParser::CommentContext* ctx);
    std::vector<std::string> extractDimensions(basemodelica::BaseModelicaParser::DeclarationContext* ctx);
};

} // namespace lacemodelica
