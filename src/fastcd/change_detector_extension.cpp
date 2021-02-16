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

#include <utility>
#include <glow/glutil.h>
#include "fastcd/change_detector.h"
#include "fastcd/regions3d_projector.h"
#include "fastcd/processed_image.h"

#include <opencv2/core/eigen.hpp>

namespace fastcd {

void ChangeDetector::DepthFilter(Image &image) {
  int scaling = image.GetCamera().GetWidth() / options_.rescale_width;

  uint32_t width = static_cast<uint32_t>(image.GetCamera().GetWidth());
  uint32_t height = static_cast<uint32_t>(image.GetCamera().GetHeight());
  Eigen::Matrix4f projection = image.GetCamera().GetGlProjection(0.1f, 50.0f);
  std::unique_ptr<DepthProjector> projector = std::unique_ptr<DepthProjector>(
      new DepthProjector(width, height, projection));
  Eigen::Matrix4f view = image.GetCamera().GetGlView();
  projector->render(mesh_, view);
  std::vector<glow::vec4> data(width*height);
  projector->texture().download(data);

  ProcessedImage processed_img(image, data);
  processed_img.ComputeDepthImage();
  cv::Mat depth(processed_img.GetDepthImage());
  cv::Mat result(processed_img.GetRawImage());

  int right(scaling/2); // bottom
  int left(scaling-right); // top
  for (int i = 0; i < result.rows; ++i) {
    int km = std::max(0, i - left);
    int kM = std::min(result.rows, i + right);
    for (int j = 0; j < result.cols; j++) {
      int lm = std::max(0, j - left);
      int lM = std::min(result.cols, j + right);
      double ref(depth.at<double>(result.rows - i - 1, j));
      //cv::Vec3b color(0,0,0);
      unsigned int cB(0),cG(0),cR(0);
      double tw(0);
      for (int k = km; k < kM; k++) {
        for (int l = lm; l < lM; l++) {
          double w(1.0/(1.0+std::abs(ref-depth.at<double>(result.rows - k - 1, l))));
          //double w(1.0);
          tw += w;
          //color += w*result.at<cv::Vec3b>(result.rows - k - 1, l);
          cB += w*result.at<cv::Vec3b>(result.rows - k - 1, l)[0];
          cG += w*result.at<cv::Vec3b>(result.rows - k - 1, l)[1];
          cR += w*result.at<cv::Vec3b>(result.rows - k - 1, l)[2];
        }
      }
      //result.at<cv::Vec3b>(result.rows - i - 1, j) = color/tw;
      result.at<cv::Vec3b>(result.rows - i - 1, j) = cv::Vec3b(cB/tw,cG/tw,cR/tw);
    }
  }
  std::cout << "done with filter" << std::endl;
  if (!projector_init_)
    cv::imwrite("/home/olivier/filter.png", result);
}

void ChangeDetector::AddImage(Image &image2, int kernel_size, bool normal) {
  Image image(image2.GetRawImage().clone(), image2.GetCamera());
  // Scale the image to a fixed resolution
  double scaling = static_cast<double>(options_.rescale_width) /
                   static_cast<double>(image.GetCamera().GetWidth());
  image.Scale(scaling);
  if (!projector_init_) {
    uint32_t width = static_cast<uint32_t>(image.GetCamera().GetWidth());
    uint32_t height = static_cast<uint32_t>(image.GetCamera().GetHeight());
    Eigen::Matrix4f projection = image.GetCamera().GetGlProjection(0.1f, 1000.0f);
    depth_projector_ = std::unique_ptr<DepthProjector>(
        new DepthProjector(width, height, projection));
    if (normal)
      normal_projector_ = std::unique_ptr<DepthProjector>(
          new DepthProjector(width, height, projection, 1));
    projector_init_ = true;
  }
  Eigen::Matrix4f view = image.GetCamera().GetGlView();
  depth_projector_->render(mesh_, view);
  std::vector<glow::vec4> data(static_cast<uint32_t>(
      image.GetCamera().GetWidth() * image.GetCamera().GetHeight()));
  depth_projector_->texture().download(data);

  std::vector<glow::vec4> data_normal(static_cast<uint32_t>(
      image.GetCamera().GetWidth() * image.GetCamera().GetHeight()));
  if (normal) {
    normal_projector_->renderNormal(mesh_, view);
    normal_projector_->texture().download(data_normal);
  } else
    data_normal = data;

  std::shared_ptr<ProcessedImage> processed_img(
      new ProcessedImage(image, data, data_normal));

  image_sequence_->AddImage2(processed_img, kernel_size);
}

void ChangeDetector::GetChanges2(std::vector<PointCovariance3d> regions3d_in_model[2]) {
  Regions3dProjector projector(image_sequence_, options_.threshold_change_area,
                               options_.threshold_change_value, options_.chi_square2d, options_.chi_square3d);
  // Get the bounding box from the mesh and the cameras and discard the regions
  // outside it
  std::pair<Eigen::Vector3d, Eigen::Vector3d> bounding_box =
      mesh_.GetBoundingBox();
  for (int i = 0; i < image_sequence_->size(); i++) {
    if ((*image_sequence_)[i].GetCamera().GetPosition()(0) <
        bounding_box.first(0))
      bounding_box.first(0) =
          (*image_sequence_)[i].GetCamera().GetPosition()(0);
    if ((*image_sequence_)[i].GetCamera().GetPosition()(1) <
        bounding_box.first(1))
      bounding_box.first(1) =
          (*image_sequence_)[i].GetCamera().GetPosition()(1);
    if ((*image_sequence_)[i].GetCamera().GetPosition()(2) <
        bounding_box.first(2))
      bounding_box.first(2) =
          (*image_sequence_)[i].GetCamera().GetPosition()(2);
    if ((*image_sequence_)[i].GetCamera().GetPosition()(0) >
        bounding_box.second(0))
      bounding_box.second(0) =
          (*image_sequence_)[i].GetCamera().GetPosition()(0);
    if ((*image_sequence_)[i].GetCamera().GetPosition()(1) >
        bounding_box.second(1))
      bounding_box.second(1) =
          (*image_sequence_)[i].GetCamera().GetPosition()(1);
    if ((*image_sequence_)[i].GetCamera().GetPosition()(2) >
        bounding_box.second(2))
      bounding_box.second(2) =
          (*image_sequence_)[i].GetCamera().GetPosition()(2);
  }

  std::vector<PointCovariance3d> regions3d = projector.GetProjectedRegions();
  for (auto &region : regions3d) {
    if (region.Point()(0) > bounding_box.first(0) &&
        region.Point()(1) > bounding_box.first(1) &&
        region.Point()(2) > bounding_box.first(2) &&
        region.Point()(0) < bounding_box.second(0) &&
        region.Point()(1) < bounding_box.second(1) &&
        region.Point()(2) < bounding_box.second(2)) {
      std::vector<Eigen::Vector3d> vertices = region.Vertices();
      bool out = false;
      for (auto &v : vertices){
        if(v(0) < bounding_box.first(0) ||
        v(1) < bounding_box.first(1) ||
        v(2) < bounding_box.first(2) ||
        v(0) > bounding_box.second(0) ||
        v(1) > bounding_box.second(1) ||
        v(2) > bounding_box.second(2)){
          break;
        }
      }
      if (!out) {
        regions3d_in_model[0].push_back(region);
      }
    }
  }
  std::vector<PointCovariance3d> regions3d2 = projector.GetProjectedRegions2();
  for (auto &region : regions3d2) {
    if (region.Point()(0) > bounding_box.first(0) &&
        region.Point()(1) > bounding_box.first(1) &&
        region.Point()(2) > bounding_box.first(2) &&
        region.Point()(0) < bounding_box.second(0) &&
        region.Point()(1) < bounding_box.second(1) &&
        region.Point()(2) < bounding_box.second(2)) {
      std::vector<Eigen::Vector3d> vertices = region.Vertices();
      bool out = false;
      for (auto &v : vertices){
        if(v(0) < bounding_box.first(0) ||
        v(1) < bounding_box.first(1) ||
        v(2) < bounding_box.first(2) ||
        v(0) > bounding_box.second(0) ||
        v(1) > bounding_box.second(1) ||
        v(2) > bounding_box.second(2)){
          break;
        }
      }
      if (!out) {
        regions3d_in_model[1].push_back(region);
      }
    }
  }
  return;
}

cv::Mat ChangeDetector::ImageChange(Image &image, const std::vector<PointCovariance3d> &regions3d, int width) {
  // Scale the image to a fixed resolution
  if (width) {
    double scaling = static_cast<double>(width) /
                     static_cast<double>(image.GetCamera().GetWidth());
    image.Scale(scaling);
  }
  if (projector_init_) {
    uint32_t width = static_cast<uint32_t>(image.GetCamera().GetWidth());
    uint32_t height = static_cast<uint32_t>(image.GetCamera().GetHeight());
    Eigen::Matrix4f projection = image.GetCamera().GetGlProjection(0.1f, 50.0f);
    depth_projector_ = std::unique_ptr<DepthProjector>(
        new DepthProjector(width, height, projection));
    projector_init_ = false;
  }

  cv::Mat depth;
  int i = 0;
  for (auto &region : regions3d) {
    Mesh ellipsoid = region.ToMesh(100, 100, 1.0f, 1.0f, 1.0f, 1.0f);
    Eigen::Matrix4f view = image.GetCamera().GetGlView();
    depth_projector_->render(ellipsoid, view);
    std::vector<glow::vec4> data(static_cast<uint32_t>(
        image.GetCamera().GetWidth() * image.GetCamera().GetHeight()));
    depth_projector_->texture().download(data);
    std::shared_ptr<ProcessedImage> processed_img(
        new ProcessedImage(image, data));
    if (i > 0)
      processed_img->ComputePresenceImage(depth);
    else
      processed_img->ComputePresenceImage(image.GetRawImage());

    depth = processed_img->GetDepthImage();
    i++;
  }
  return depth;
}

cv::Mat ChangeDetector::ImageChange(Image &image, const std::vector<PointCovariance3d> &regions3d, Image &mask, int width) {
  // Scale the image to a fixed resolution
  if (width) {
    double scaling = static_cast<double>(width) /
                     static_cast<double>(image.GetCamera().GetWidth());
    image.Scale(scaling);
  }
  if (projector_init_) {
    uint32_t width = static_cast<uint32_t>(image.GetCamera().GetWidth());
    uint32_t height = static_cast<uint32_t>(image.GetCamera().GetHeight());
    Eigen::Matrix4f projection = image.GetCamera().GetGlProjection(0.1f, 50.0f);
    depth_projector_ = std::unique_ptr<DepthProjector>(
        new DepthProjector(width, height, projection));
    projector_init_ = false;
  }

  cv::Mat depth;
  int i = 0;
  for (auto &region : regions3d) {
    Mesh ellipsoid = region.ToMesh(100, 100, 1.0f, 1.0f, 1.0f, 1.0f);
    Eigen::Matrix4f view = image.GetCamera().GetGlView();
    depth_projector_->render(ellipsoid, view);
    std::vector<glow::vec4> data(static_cast<uint32_t>(
        image.GetCamera().GetWidth() * image.GetCamera().GetHeight()));
    depth_projector_->texture().download(data);
    std::shared_ptr<ProcessedImage> processed_img(
        new ProcessedImage(image, data));
    if (i > 0)
      processed_img->ComputePresenceImage(depth, mask.GetRawImage());
    else
      processed_img->ComputePresenceImage(image.GetRawImage(), mask.GetRawImage());

    depth = processed_img->GetDepthImage();
    i++;
  }
  return depth;
}

}  // namespace fastcd
#endif //NOLIVIER
