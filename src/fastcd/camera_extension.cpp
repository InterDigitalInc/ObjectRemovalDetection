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

#include "fastcd/camera.h"
#include <glow/glutil.h>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace fastcd {

void Camera::ReadCalibrationIntrinsic(const std::string& filename) {
  std::ifstream file;
  file.open(filename);
  double cx, cy, fx, fy, bu;
  if (file) {
    file >> fx >> bu >> cx >> bu >>
            bu >> fy >> cy;
    file.close();
  }

  width_ = 1296;
  height_ = 968;
  calibration_.setIdentity();
  calibration_(0, 0) = fx;
  calibration_(1, 1) = fy;
  calibration_(0, 2) = cx;
  calibration_(1, 2) = cy;
  inverse_calibration_ = calibration_.inverse();
}

void Camera::ReadCalibrationExtrinsic(const std::string& filename) {
  std::ifstream file;
  file.open(filename);
  if (file) {
    for (size_t i = 0; i < 4; i++)
      for (size_t j = 0; j < 4; j++)
        file >> pose_(i, j);
    file.close();
  }
}

Eigen::Vector3d Camera::Transform(double x, double y, double z) const {
  Eigen::Vector4d world_point;
  Eigen::Vector3d camera_point;
  Eigen::Matrix<double, 3, 4> P;
  Eigen::Matrix3d R = pose_.block<3, 3>(0, 0).transpose();
  world_point << x, y, z, 1;
  P.block<3, 3>(0, 0) = R;
  P.block<3, 1>(0, 3) = -R * pose_.block<3, 1>(0, 3);
  camera_point = P * world_point;
  return camera_point;
}

Eigen::Vector3d Camera::Transform(const Eigen::Vector3d& point) const {
  return Transform(point(0), point(1), point(2));
}

Eigen::Matrix3d Camera::ComputeF(const Camera& target) const {
  Eigen::Matrix<double, 3, 4> P1(GetP());
  Eigen::Matrix<double, 3, 4> P2(target.GetP());
  Eigen::Matrix<double, 2, 4> X1, X2, X3, Y1, Y2, Y3;

  X1.block<1, 4>(0, 0) = P1.block<1, 4>(1, 0);
  X1.block<1, 4>(1, 0) = P1.block<1, 4>(2, 0);

  X2.block<1, 4>(0, 0) = P1.block<1, 4>(2, 0);
  X2.block<1, 4>(1, 0) = P1.block<1, 4>(0, 0);

  X3.block<1, 4>(0, 0) = P1.block<1, 4>(0, 0);
  X3.block<1, 4>(1, 0) = P1.block<1, 4>(1, 0);

  Y1.block<1, 4>(0, 0) = P2.block<1, 4>(1, 0);
  Y1.block<1, 4>(1, 0) = P2.block<1, 4>(2, 0);

  Y2.block<1, 4>(0, 0) = P2.block<1, 4>(2, 0);
  Y2.block<1, 4>(1, 0) = P2.block<1, 4>(0, 0);

  Y3.block<1, 4>(0, 0) = P2.block<1, 4>(0, 0);
  Y3.block<1, 4>(1, 0) = P2.block<1, 4>(1, 0);

  Eigen::Matrix4d XY11, XY21, XY31, XY12, XY22, XY32, XY13, XY23, XY33;

  XY11.block<2, 4>(0, 0) = X1;
  XY11.block<2, 4>(2, 0) = Y1;

  XY21.block<2, 4>(0, 0) = X2;
  XY21.block<2, 4>(2, 0) = Y1;

  XY31.block<2, 4>(0, 0) = X3;
  XY31.block<2, 4>(2, 0) = Y1;

  XY12.block<2, 4>(0, 0) = X1;
  XY12.block<2, 4>(2, 0) = Y2;

  XY22.block<2, 4>(0, 0) = X2;
  XY22.block<2, 4>(2, 0) = Y2;

  XY32.block<2, 4>(0, 0) = X3;
  XY32.block<2, 4>(2, 0) = Y2;

  XY13.block<2, 4>(0, 0) = X1;
  XY13.block<2, 4>(2, 0) = Y3;

  XY23.block<2, 4>(0, 0) = X2;
  XY23.block<2, 4>(2, 0) = Y3;

  XY33.block<2, 4>(0, 0) = X3;
  XY33.block<2, 4>(2, 0) = Y3;

  Eigen::Matrix3d F;
  F << XY11.determinant(), XY21.determinant(), XY31.determinant(),
       XY12.determinant(), XY22.determinant(), XY32.determinant(),
       XY13.determinant(), XY23.determinant(), XY33.determinant();
  return F;
}

}  // namespace fastcd

#endif //NOLIVIER
