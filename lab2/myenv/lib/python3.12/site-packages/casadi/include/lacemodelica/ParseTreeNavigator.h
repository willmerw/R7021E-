// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "BaseModelicaParser.h"
#include <antlr4-runtime.h>
#include <typeindex>
#include <functional>
#include <map>
#include <utility>

namespace lacemodelica {

/**
 * Parse tree navigator that provides generic navigation through ANTLR parse trees.
 * Uses a registry pattern to map node types to navigation strategies.
 */
class ParseTreeNavigator {
public:
    /**
     * Generic template to find any node type in the parse tree by navigating
     * through the grammar hierarchy.
     */
    template<typename TargetType>
    static TargetType* findNode(antlr4::tree::ParseTree* start) {
        if (!start) return nullptr;

        antlr4::tree::ParseTree* current = start;

        // First check if start is already the target type
        if (auto target = dynamic_cast<TargetType*>(current)) {
            return target;
        }

        // Navigate down the tree using registered navigation rules
        while (current) {
            auto* ruleCtx = dynamic_cast<antlr4::ParserRuleContext*>(current);
            if (!ruleCtx) break;

            // Check if current node is the target
            if (auto target = dynamic_cast<TargetType*>(ruleCtx)) {
                return target;
            }

            // Get navigation function for this node type
            std::type_index typeIdx(typeid(*ruleCtx));
            auto it = navigationMap.find(typeIdx);
            if (it == navigationMap.end()) {
                break; // No navigation rule for this type
            }

            // Navigate to next node
            current = it->second(ruleCtx);
        }

        return nullptr;
    }

    /**
     * Convenience wrapper to find Primary context.
     */
    static basemodelica::BaseModelicaParser::PrimaryContext*
        findPrimary(antlr4::tree::ParseTree* start) {
        return findNode<basemodelica::BaseModelicaParser::PrimaryContext>(start);
    }

    /**
     * Convenience wrapper to find OutputExpressionList context.
     * Navigates to Primary first, then checks for outputExpressionList.
     */
    static basemodelica::BaseModelicaParser::OutputExpressionListContext*
        findOutputExpressionList(antlr4::tree::ParseTree* start) {
        auto primary = findPrimary(start);
        return primary ? primary->outputExpressionList() : nullptr;
    }

    /**
     * Check if an expression is a range expression (has colons like "2:4" or "1:2:10").
     */
    static bool isRangeExpression(basemodelica::BaseModelicaParser::ExpressionContext* expr);

    /**
     * Parse range bounds from a range expression like "2:4" or "1:2:10".
     * Returns {start, end} (1-based, inclusive as in Modelica).
     * Step is ignored for now (assumes step=1).
     */
    static std::pair<int64_t, int64_t> parseRangeBounds(
        basemodelica::BaseModelicaParser::ExpressionContext* expr);

private:
    // Navigation map: type -> function that returns next child node
    using NavigationFunc = std::function<antlr4::tree::ParseTree*(antlr4::ParserRuleContext*)>;
    static std::map<std::type_index, NavigationFunc> navigationMap;

    // Initialize navigation rules at startup
    static bool initialized;
    static bool initialize();
    static const bool dummy; // Trigger initialization
};

} // namespace lacemodelica
