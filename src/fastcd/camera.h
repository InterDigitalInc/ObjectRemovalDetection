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
// Copyright 2017 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

#ifndef NOLIVIER
#include <opencv2/opencv.hpp>
#endif //NOLIVIER

namespace fastcd {

/**
 * @brief      Class that represents a camera. It stores its calibration and
 *             allows the projection and back-projection of points.
 */
class Camera {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
  * @brief      Reads the camera calibration from an XML file with Agisoft
  *             format.
  *
  * @param[in]  filename  The file path.
  * @param[in]  id        The camera id in the XML file.
  */
  void ReadCalibration(const std::string& filename, int id);
#ifndef NOLIVIER
  void ReadCalibrationIntrinsic(const std::string& filename);
  void ReadCalibrationExtrinsic(const std::string& filename);
#endif //NOLIVIER

  /**
   * @brief      Scales the camera calibration by the specified scaling factor.
   *
   * @param[in]  scaling  The scaling factor
   */
  void ScaleCalibration(double scaling);

  /**
   * @brief      Gets the calibration matrix.
   *
   * @return     The calibration matrix.
   */
  Eigen::Matrix3d GetK() const;

  /**
   * @brief      Gets the inverse calibration matrix.
   *
   * @return     The inverse calibration matrix.
   */
  Eigen::Matrix3d GetInvK() const;

  /**
   * @brief      Gets the pose of the camera.
   *
   * @return     The pose of the camera.
   */
  Eigen::Matrix4d GetPose() const;

  /**
   * @brief      Gets the position of the camera.
   *
   * @return     The position of the camera.
   */
  Eigen::Vector3d GetPosition() const;

  /**
   * @brief      Compute the OpenGL projection matrix relative to the camera.
   *
   * @return     The OpenGL projection matrix.
   */
  Eigen::Matrix4f GetGlProjection(float near, float far);

  /**
   * @brief      Compute the OpenGL view matrix relative to the camera.
   *
   * @return     The OpenGL view matrix.
   */
  Eigen::Matrix4f GetGlView();

  /**
   * @brief      Return a vector of vertices for an OpenGL visualization of the
   *             camera in the form of a line strip.
   *
   * @param[in]  color   The color of the line strip
   * @param[in]  length  The length of the four edges of the camera
   *
   * @return     The vector of vertices.
   */
  std::vector<float> GetGlFovVertices(float color, float length);

  /**
   * @brief      Gets the width of the image plane in pixels.
   *
   * @return     The width of the image plane in pixels.
   */
  int GetWidth();

  /**
   * @brief      Gets the height of the image plane in pixels.
   *
   * @return     The height of the image plane in pixels.
   */
  int GetHeight();

  /**
   * @brief      Gets the camera projection matrix.
   *
   * @return     The camera projection matrix.
   */
  Eigen::Matrix<double, 3, 4> GetP() const;

  /**
   * @brief      Projects a point on the image plane.
   *
   * @param[in]  x     The x coordinate
   * @param[in]  y     The y coordinate
   * @param[in]  z     The z coordinate
   *
   * @return     The (u,v) coordinates of the point.
   */
  Eigen::Vector2i Project(double x, double y, double z) const;

  /**
   * @brief      Projects a point on the image plane.
   *
   * @param[in]  point  The 3D point
   *
   * @return     The (u,v) coordinates of the point.
   */
  Eigen::Vector2i Project(const Eigen::Vector3d& point) const;

#ifndef NOLIVIER
  Eigen::Vector3d Transform(double x, double y, double z) const;
  Eigen::Vector3d Transform(const Eigen::Vector3d& point) const;
#endif //NOLIVIER

  /**
   * @brief      Back-projects a pixel.
   *
   * @param[in]  u     The u coordinate
   * @param[in]  v     The v coordinate
   *
   * @return     The 3D coordinate of a point in the direction of the ray.
   */
  Eigen::Vector3d BackProject(int u, int v);

  /**
   * @brief      Back-projects a pixel.
   *
   * @param[in]  point  The 2D point
   *
   * @return     The 3D coordinate of a point in the direction of the ray
   */
  Eigen::Vector3d BackProject(const Eigen::Vector2i &point);

  /**
   * @brief      Prints the camera parameters on the output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const Camera& c);
#ifndef NOLIVIER
  Eigen::Matrix3d ComputeF(const Camera& target) const;
#endif //NOLIVIER

 protected:
  /** Width of the image plane (in pixels) */
  int width_ = 0;

  /** Height of the image plane (in pixels) */
  int height_ = 0;

  /** Calibration matrix */
  Eigen::Matrix3d calibration_ = Eigen::Matrix3d::Identity();

  /** Inverse of calibration matrix */
  Eigen::Matrix3d inverse_calibration_ = Eigen::Matrix3d::Identity();

  /** Pose of the camera */
  Eigen::Matrix4d pose_ = Eigen::Matrix4d::Identity();
};

}  // namespace fastcd
