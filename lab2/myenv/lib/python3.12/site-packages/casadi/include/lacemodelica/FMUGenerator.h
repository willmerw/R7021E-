// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "ModelInfo.h"
#include <string>

namespace lacemodelica {

class FMUGenerator {
public:
    bool generateFMU(const ModelInfo& info, const std::string& outputPath);

private:
    bool generateModelDescription(const ModelInfo& info, const std::string& xmlPath);
    std::string generateGUID();
};

} // namespace lacemodelica
