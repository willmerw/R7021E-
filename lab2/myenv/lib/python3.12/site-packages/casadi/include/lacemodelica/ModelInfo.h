// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include <string>
#include <vector>
#include <map>

// Forward declare ANTLR context types
namespace basemodelica {
    class BaseModelicaParser;
}

namespace antlr4 {
    class ParserRuleContext;
}

namespace lacemodelica {

struct Equation {
    std::string comment;  // String comment / documentation
    std::string sourceFile;  // Source filename for debugging
    size_t sourceLine = 0;  // Source line number for debugging
    antlr4::ParserRuleContext* lhsContext = nullptr;  // AST node for LHS
    antlr4::ParserRuleContext* rhsContext = nullptr;  // AST node for RHS
    antlr4::ParserRuleContext* forEquationContext = nullptr;  // AST node for for-equation (if this is a for-equation)
    antlr4::ParserRuleContext* ifEquationContext = nullptr;  // AST node for if-equation (if this is an if-equation)

    bool isForEquation() const { return forEquationContext != nullptr; }
    bool isIfEquation() const { return ifEquationContext != nullptr; }
};

struct Variable {
    std::string name;
    std::string type;  // "Real", "Integer", "Boolean"
    std::string causality;  // "parameter", "structuralParameter", "local", "output"
    std::string variability;  // "fixed", "tunable", "continuous"
    std::string initial;  // "exact", "approx", "calculated"
    std::string startValue;
    std::string minValue;
    std::string maxValue;
    std::string description;  // Variable description from comment
    int valueReference;
    int derivativeOf = -1;  // valueReference of state if this is a derivative
    std::vector<std::string> dimensions;  // For arrays
    bool isStructuralParameter = false;
    bool isDerivative = false;
    std::string sourceFile;  // Source filename for debugging
    size_t sourceLine = 0;  // Source line number for debugging
    antlr4::ParserRuleContext* bindingContext = nullptr;  // AST node for binding expression
    antlr4::ParserRuleContext* minContext = nullptr;  // AST node for min expression
    antlr4::ParserRuleContext* maxContext = nullptr;  // AST node for max expression
};

struct Statement {
    antlr4::ParserRuleContext* lhsContext = nullptr;  // componentReference on left of :=
    antlr4::ParserRuleContext* rhsContext = nullptr;  // expression on right of :=
    antlr4::ParserRuleContext* forStatementContext = nullptr;  // AST node for for-statement
    antlr4::ParserRuleContext* whileStatementContext = nullptr;  // AST node for while-statement
    antlr4::ParserRuleContext* ifStatementContext = nullptr;  // AST node for if-statement
    antlr4::ParserRuleContext* functionCallContext = nullptr;  // AST node for standalone function call (no assignment)
    std::string sourceFile;
    size_t sourceLine = 0;

    bool isForStatement() const { return forStatementContext != nullptr; }
    bool isWhileStatement() const { return whileStatementContext != nullptr; }
    bool isIfStatement() const { return ifStatementContext != nullptr; }
    bool isFunctionCallStatement() const { return functionCallContext != nullptr; }
};

struct Function {
    std::string name;
    std::string description;
    std::vector<Variable> inputs;
    std::vector<Variable> outputs;
    std::vector<Statement> algorithmStatements;
    std::string sourceFile;
    size_t sourceLine = 0;
};

struct Assertion {
    antlr4::ParserRuleContext* conditionContext = nullptr;  // The condition expression
    std::string message;  // The error/warning message
    std::string level = "error";  // "error" or "warning" (from AssertionLevel)
    std::string sourceFile;
    size_t sourceLine = 0;
};

struct EnumLiteral {
    std::string name;
    std::string description;
    int value;  // 0-based integer value
};

struct EnumType {
    std::string name;
    std::string description;
    std::vector<EnumLiteral> literals;

    // Get the integer value for a literal name, returns -1 if not found
    int getValue(const std::string& literalName) const {
        for (const auto& lit : literals) {
            if (lit.name == literalName) {
                return lit.value;
            }
        }
        return -1;
    }
};

class ModelInfo {
public:
    std::string packageName;
    std::string modelName;
    std::string description;

    std::vector<Variable> variables;
    std::map<std::string, int> variableIndex;  // name -> index in variables
    std::vector<Equation> equations;
    std::vector<Equation> initialEquations;
    std::vector<Assertion> assertions;
    std::vector<Function> functions;
    std::map<std::string, int> functionIndex;  // name -> index in functions
    std::vector<EnumType> enumTypes;
    std::map<std::string, int> enumTypeIndex;  // name -> index in enumTypes

    int nextValueReference = 1;

    void addVariable(const Variable& var) {
        variableIndex[var.name] = variables.size();
        variables.push_back(var);
    }

    Variable* findVariable(const std::string& name) {
        auto it = variableIndex.find(name);
        if (it != variableIndex.end()) {
            return &variables[it->second];
        }
        return nullptr;
    }

    const Variable* findVariable(const std::string& name) const {
        auto it = variableIndex.find(name);
        if (it != variableIndex.end()) {
            return &variables[it->second];
        }
        return nullptr;
    }

    void addFunction(const Function& func) {
        functionIndex[func.name] = functions.size();
        functions.push_back(func);
    }

    Function* findFunction(const std::string& name) {
        auto it = functionIndex.find(name);
        if (it != functionIndex.end()) {
            return &functions[it->second];
        }
        return nullptr;
    }

    const Function* findFunction(const std::string& name) const {
        auto it = functionIndex.find(name);
        if (it != functionIndex.end()) {
            return &functions[it->second];
        }
        return nullptr;
    }

    void addEnumType(const EnumType& enumType) {
        enumTypeIndex[enumType.name] = enumTypes.size();
        enumTypes.push_back(enumType);
    }

    EnumType* findEnumType(const std::string& name) {
        auto it = enumTypeIndex.find(name);
        if (it != enumTypeIndex.end()) {
            return &enumTypes[it->second];
        }
        return nullptr;
    }

    const EnumType* findEnumType(const std::string& name) const {
        auto it = enumTypeIndex.find(name);
        if (it != enumTypeIndex.end()) {
            return &enumTypes[it->second];
        }
        return nullptr;
    }

    std::vector<Variable> getStates() const {
        std::vector<Variable> states;
        for (const auto& var : variables) {
            if (var.variability == "continuous" && !var.isDerivative && var.causality == "local") {
                states.push_back(var);
            }
        }
        return states;
    }

    std::vector<Variable> getDerivatives() const {
        std::vector<Variable> derivatives;
        for (const auto& var : variables) {
            if (var.isDerivative) {
                derivatives.push_back(var);
            }
        }
        return derivatives;
    }

    std::vector<Variable> getOutputs() const {
        std::vector<Variable> outputs;
        for (const auto& var : variables) {
            if (var.causality == "output") {
                outputs.push_back(var);
            }
        }
        return outputs;
    }
};

} // namespace lacemodelica
