// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include <string>

namespace lacemodelica {

// Strip surrounding single quotes from BaseModelica quoted identifiers
// e.g., "'foo'" -> "foo", "bar" -> "bar"
inline std::string stripQuotes(const std::string& str) {
    if (str.size() >= 2 && str.front() == '\'' && str.back() == '\'') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

// Check if a string represents a constant value (number or boolean literal).
// Returns true for numeric literals and "true"/"false", false for expressions.
inline bool isConstValue(const std::string& value) {
    if (value.empty()) return false;

    // Boolean literals
    if (value == "true" || value == "false") return true;

    // Try to parse as number - must consume entire string
    try {
        size_t pos = 0;
        std::stod(value, &pos);
        // Skip trailing whitespace
        while (pos < value.size() && std::isspace(value[pos])) {
            pos++;
        }
        return pos == value.size();
    } catch (...) {
        return false;
    }
}

// Add source file and line metadata to any ONNX object with add_metadata_props()
// Works with ValueInfoProto, NodeProto, etc.
template<typename T>
void addSourceLocationMetadata(T* obj, const std::string& sourceFile, size_t sourceLine) {
    if (sourceFile.empty()) return;

    auto* metaFile = obj->add_metadata_props();
    metaFile->set_key("source_file");
    metaFile->set_value(sourceFile);

    auto* metaLine = obj->add_metadata_props();
    metaLine->set_key("source_line");
    metaLine->set_value(std::to_string(sourceLine));
}

} // namespace lacemodelica
