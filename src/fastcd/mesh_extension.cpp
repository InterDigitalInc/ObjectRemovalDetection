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
#ifndef NOLIVIER
#include "fastcd/mesh.h"

#include <glow/ScopedBinder.h>

void Mesh::initialize(const std::vector<Vertex>& verts,
                      const std::vector<Triangle>& faces,
                      const std::vector<Material>& materials) {
  if (verts.size() == 0) return;

  std::vector<Vertex> vertices = verts;

  if (std::abs(length(vertices[0].normal) - 1.0f) > 0.01) {
    // assume, no normals given:
    for (auto face : faces) {
      Eigen::Vector3f v0 = toEigen(vertices[face.vertices[0]].position);
      Eigen::Vector3f v1 = toEigen(vertices[face.vertices[1]].position);
      Eigen::Vector3f v2 = toEigen(vertices[face.vertices[2]].position);

      Eigen::Vector3f n = ((v1 - v0).cross(v2 - v0)).normalized();
      for (uint32_t i = 0; i < 3; ++i) {
        vertices[face.vertices[i]].normal.x = n.x();
        vertices[face.vertices[i]].normal.y = n.y();
        vertices[face.vertices[i]].normal.z = n.z();
      }
    }
  }
  std::vector<std::vector<uint32_t> > tris(std::max((int)materials.size(), 1));
  for (uint32_t i = 0; i < faces.size(); ++i) {
    assert(faces[i].material < tris.size());
    tris[faces[i].material].push_back(faces[i].vertices[0]);
    tris[faces[i].material].push_back(faces[i].vertices[1]);
    tris[faces[i].material].push_back(faces[i].vertices[2]);
  }


  // copying the data.
  vertices_.assign(vertices);
  for (uint32_t i = 0; i < reordered_tris.size(); ++i) {
    glow::GlBuffer<uint32_t> buf(glow::BufferTarget::ELEMENT_ARRAY_BUFFER,
                                 glow::BufferUsage::STATIC_DRAW);
    buf.assign(reordered_tris[i]);
    triangles_.push_back(buf);
  }


  // setting the vaos.
  for (uint32_t i = 0; i < reordered_tris.size(); ++i) {
    vaos_.push_back(glow::GlVertexArray());

    vaos_[i].bind();
    vaos_[i].setVertexAttribute(0, vertices_, 4, glow::AttributeType::FLOAT,
                                false, sizeof(Vertex), (GLvoid*)0);
    vaos_[i].setVertexAttribute(1, vertices_, 4, glow::AttributeType::FLOAT,
                                false, sizeof(Vertex),
                                (GLvoid*)offsetof(Vertex, normal));
    vaos_[i].setVertexAttribute(2, vertices_, 2, glow::AttributeType::FLOAT,
                                false, sizeof(Vertex),
                                (GLvoid*)offsetof(Vertex, texture));
    vaos_[i].enableVertexAttribute(0);
    vaos_[i].enableVertexAttribute(1);
    vaos_[i].enableVertexAttribute(2);
    triangles_[i].bind();
    vaos_[i].release();

    triangles_[i].release();  // release only afterwards

    CheckGlError();
  }
}

void Mesh::saveObj(const std::string& filename) const {
  std::ofstream out(filename.c_str());
  if (!out.is_open()) throw std::runtime_error("Failed to open obj-file.");

  for (auto vertex : vertices_) {
    out << 'v' << vertex.position;
    /*for (auto vertex : triangle.vertices)
      out << ' ' << vertex << '/' << vertex << '/' << vertex;*/
    out << std::endl;
  }

  /*for (auto triangle : triangles_) {
    out << 'f';
    for (auto vertex : triangle.vertices)
      out << ' ' << vertex << '/' << vertex << '/' << vertex;
    out << std::endl;
  }
  out << "# End of File" << std::flush;
  out.close();
}*/

#endif //NOLIVIER
