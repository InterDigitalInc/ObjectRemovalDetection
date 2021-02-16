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
#include "utils/obj_reader.h"
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

extern std::string trim(const std::string& str,
                        const std::string& whitespaces = " \0\t\n\r\x0B");

extern std::vector<std::string> split(const std::string& line,
                                      const std::string& delim,
                                      bool skipEmpty = false);

extern std::string dirName(const std::string& path);

Mesh ObjReader::FromFileNoMaterial(const std::string& filename) {
  //  std::cout << "ObjReader: " << filename << std::flush;

  struct ObjFace {
    int32_t vertices[3];
    int32_t normals[3];
    int32_t texcoords[3];
    int32_t material;
  };

  std::vector<Mesh::Vertex> vertices;
  std::vector<Mesh::Triangle> triangles;

  std::ifstream in(filename.c_str());
  if (!in.is_open()) throw std::runtime_error("Failed to open obj-file.");
  std::vector<std::string> tokens;
  std::string line;

  std::vector<glow::vec4> obj_verts;
  std::vector<glow::vec4> obj_normals;
  std::vector<glow::vec2> obj_texcoords;
  std::vector<ObjFace> obj_faces;

  // 1. read first all information.
  // 2. assembly then the vertices with help of vert_normal.
  // 3. construct mesh primitives.

  in.peek();
  while (!in.eof()) {
    std::getline(in, line);
    tokens = split(line, " ");
    in.peek();
    if (tokens.size() < 2) continue;
    if (tokens[0] == "#") continue;

    if (tokens[0] == "mtllib") {
      std::map<std::string, Mesh::Material> mats;
      std::string mtl_filename = dirName(filename);
      mtl_filename += "/" + tokens[1];
      ParseMaterials(mtl_filename, mats);

    } else if (tokens[0] == "v") {  // vertices.
      if (tokens.size() < 4)
        throw std::runtime_error("Failure parsing vertex: not enough tokens.");
      float x = boost::lexical_cast<float>(tokens[1]);
      float y = boost::lexical_cast<float>(tokens[2]);
      float z = boost::lexical_cast<float>(tokens[3]);

      // FIXME: interpret also w coordinate.
      obj_verts.push_back(glow::vec4(x, y, z, 1.0));
    } else if (tokens[0] == "vn") {  // normals
      if (tokens.size() < 4)
        throw std::runtime_error("Failure parsing normal: not enough tokens.");
      float x = boost::lexical_cast<float>(tokens[1]);
      float y = boost::lexical_cast<float>(tokens[2]);
      float z = boost::lexical_cast<float>(tokens[3]);

      obj_normals.push_back(glow::vec4(x, y, z, 0.0));

      // FIXME: interpret also w coordinate.
    } else if (tokens[0] == "f") {
      if (tokens.size() < 4)
        throw std::runtime_error("Failure parsing face: not enough tokens.");
      std::vector<std::string> face_tokens;
      ObjFace face;
      face.normals[0] = -1;
      face.normals[1] = -1;
      face.normals[2] = -1;
      face.material = -1;

      for (uint32_t j = 1; j < 4; ++j) {
        face_tokens = split(tokens[j], "/");
        if (face_tokens.size() == 0) {
          //          std::cout << stringify(tokens) << std::endl;
          std::cout << tokens[j] << std::endl;
          throw std::runtime_error(
              "Failure parsing face: insufficient vertex params");
        }
        if (face_tokens.size() > 0) {  // vertex
          face.vertices[j - 1] =
              boost::lexical_cast<uint32_t>(face_tokens[0]) - 1;
        }

        if (face_tokens.size() > 1) {       // texture
          if (face_tokens[1].size() > 0) {  // non-empty
            face.texcoords[j - 1] =
                boost::lexical_cast<uint32_t>(face_tokens[1]) - 1;
          }
        }

        if (face_tokens.size() > 2) {       // normal
          if (face_tokens[2].size() > 0) {  // non-empty
            face.normals[j - 1] =
                boost::lexical_cast<uint32_t>(face_tokens[2]) - 1;
          }
        }
      }

      obj_faces.push_back(face);
    }

    in.peek();
  }
  in.close();

  // getting normal; vertices together (map v * (N+1) + n -> index, where N is
  // normals size)
  std::map<uint32_t, uint32_t> vert_normal;
  // generate for all faces the corresponding vertices:
  for (uint32_t f = 0; f < obj_faces.size(); ++f) {
    const ObjFace& face = obj_faces[f];
    Mesh::Triangle triangle;
    for (uint32_t i = 0; i < 3; ++i) {
      uint32_t key =
          face.vertices[i] * (obj_normals.size() + 1) + (face.normals[i] + 1);
      if (vert_normal.find(key) == vert_normal.end()) {
        Mesh::Vertex vertex;
        vertex.position = obj_verts[face.vertices[i]];
        if (face.normals[i] >= 0) vertex.normal = obj_normals[face.normals[i]];
        // FIXME: texture coordinates.

        triangle.vertices[i] = vertices.size();
        vert_normal[key] = vertices.size();
        vertices.push_back(vertex);
      } else {
        triangle.vertices[i] = vert_normal[key];
      }
    }
    triangles.push_back(triangle);
  }

  return Mesh(vertices, triangles);
}
#endif //NOLIVIER
