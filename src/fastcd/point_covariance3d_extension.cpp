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
 */#ifndef NOLIVIER
#include "fastcd/point_covariance3d.h"
#include <Eigen/Core>
#include "fastcd/mesh.h"
#include <fstream>

namespace fastcd {

void PointCovariance3d::ToObj(const std::string& filename, int stacks,
                              int slices) const {
  std::ofstream out(filename.c_str());
  if (!out.is_open()) throw std::runtime_error("Failed to open obj-file.");

  int vertices(0);
  float t_step = M_PI / static_cast<float>(slices);
  float s_step = M_PI / static_cast<float>(stacks);
  for (float t = -M_PI / 2; t <= (M_PI / 2) + .0001; t += t_step) {
    for (float s = -M_PI; s <= M_PI + .0001; s += s_step) {
      Eigen::Vector3d vertex;
      vertex << cos(t) * cos(s), cos(t) * sin(s), sin(t);
      vertex = eigenvectors_ * scaling_ * vertex + point_;
      vertices++;
      out << "v " << vertex(0) << ' ' << vertex(1) << ' ' << vertex(2) << '\n';

      vertex << cos(t + t_step) * cos(s), cos(t + t_step) * sin(s),
          sin(t + t_step);
      vertex = eigenvectors_ * scaling_ * vertex + point_;
      vertices++;
      out << "v " << vertex(0) << ' ' << vertex(1) << ' ' << vertex(2) << '\n';
    }
  }
  out << "# " << vertices << " vertices, 0 vertices normals\n\n";

  int triangles(0);
  for (size_t i = 2; i < vertices; i++) {
    Mesh::Triangle t;
    t.vertices[0] = i-2;
    t.vertices[1] = i % 2 == 0 ? i - 1 : i;
    t.vertices[2] = i % 2 == 0 ? i : i - 1;

    if (t.vertices[0] != t.vertices[1] && t.vertices[1] != t.vertices[2] &&
        t.vertices[2] != t.vertices[0]) {
      triangles++;
      out << "\nf";
      for (auto vertex : t.vertices)
        out << ' ' << vertex << '/' /*<< vertex*/ << '/' << vertex;
    }
  }
  out << "\n# " << triangles << " faces, 0 coords texture\n\n";
  out << "# End of File" << std::flush;
}

}  // namespace fastcd
#endif //NOLIVIER
