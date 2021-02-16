/*
 * Copyright 2021 InterDigital
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * License is avaiable on the root under license.txt and you may obtain a copy
 * of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
// Copyright 2017 Jens Behley (jens.behley@igg.uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <map>
#include "fastcd/mesh.h"

/**
 * @brief      Class for loading a mesh from a Wavefront .obj file.
 */
class ObjReader {
 public:
  /**
   * @brief      Read a mesh from a file with the given filename.
   *
   * @param[in]  filename  The filename
   *
   * @return     The mesh.
   */
  static Mesh FromFile(const std::string& filename);
#ifndef NOLIVIER
  static Mesh FromFileNoMaterial(const std::string& filename);
#endif //NOLIVIER

 protected:
  /**
   * @brief      Parse the given material file
   *
   * @param[in]  filename   The filename
   * @param[out] materials  The materials
   */
  static void ParseMaterials(const std::string& filename,
                             std::map<std::string, Mesh::Material>& materials);
};
