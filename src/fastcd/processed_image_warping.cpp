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
void ProcessedImage::Interpolate(cv::Mat &result, const int dilation_size) const {
  cv::Mat interpolation(result.rows, result.cols, result.type());
  cv::Mat dilation_kernel = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
      cv::Point(dilation_size, dilation_size));
  cv::dilate(result, interpolation, dilation_kernel);
  for (int i = 0; i < result.rows; i++)
    for (int j = 0; j < result.cols; j++)
      if (result.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0))
        result.at<cv::Vec3b>(i, j) = interpolation.at<cv::Vec3b>(i, j);
}

Image ProcessedImage::Warp(const ProcessedImage &target_image, bool interpolate, int factor) const {
  cv::Mat result(raw_image_.rows, raw_image_.cols, raw_image_.type(),
                 cv::Scalar(0, 0, 0));
  cv::Mat zbuffer(raw_image_.rows / factor, raw_image_.cols / factor, CV_64FC1,
                  cv::Scalar(1000));
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < raw_image_.rows; ++i) {
    for (int j = 0; j < raw_image_.cols; j++) {
      int idx = i * raw_image_.cols + j;
      if (depth_[idx].w > 0.5) {
        coord = target_image.GetCamera().Project(
            static_cast<double>(depth_[idx].x),
            static_cast<double>(depth_[idx].y),
            static_cast<double>(depth_[idx].z));
        // Check if the point is seen in both images
        // (taking into account occlusions)
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          double distance =
              sqrt(pow(static_cast<double>(depth_[idx].x) -
                           target_image.GetCamera().GetPosition()(0), 2) +
                   pow(static_cast<double>(depth_[idx].y) -
                           target_image.GetCamera().GetPosition()(1), 2) +
                   pow(static_cast<double>(depth_[idx].z) -
                           target_image.GetCamera().GetPosition()(2), 2));
          if (distance < zbuffer.at<double>(coord(1) / factor, coord(0) / factor)) {
            zbuffer.at<double>(coord(1) / factor, coord(0) / factor) = distance;
            int depth_idx =
                (raw_image_.rows - coord(1) - 1) * target_image.Width() +
                coord(0);
            double dist_target =
                sqrt(pow(target_image.GetCamera().GetPosition()(0) -
                             target_image.GetDepth()[depth_idx].x, 2) +
                     pow(target_image.GetCamera().GetPosition()(1) -
                             target_image.GetDepth()[depth_idx].y, 2) +
                     pow(target_image.GetCamera().GetPosition()(2) -
                             target_image.GetDepth()[depth_idx].z, 2));
            if (dist_target >= distance - STEP) {
              result.at<cv::Vec3b>(coord(1), coord(0)) =
                  raw_image_.at<cv::Vec3b>(raw_image_.rows - i - 1, j);
            }
          }
        }
      }
    }
  }

  if (interpolate)
    Interpolate(result);

  return Image(result, target_image.GetCamera());
}

Image ProcessedImage::Warp2(const ProcessedImage &target_image, int shadows) const {
  cv::Mat result(raw_image_.rows, raw_image_.cols, raw_image_.type(),
                 cv::Scalar(0, 0, 0));
  if (shadows == -2)
    target_image.GetRawImage().copyTo(result);
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        // Check if the point is seen in both images
        // (taking into account occlusions)
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          int idx2 = (raw_image_.rows - coord(1) - 1) * raw_image_.cols + coord(0);
          double distance =
              sqrt(pow(static_cast<double>(depth_[idx2].x) -
                           camera_.GetPosition()(0), 2) +
                   pow(static_cast<double>(depth_[idx2].y) -
                           camera_.GetPosition()(1), 2) +
                   pow(static_cast<double>(depth_[idx2].z) -
                           camera_.GetPosition()(2), 2)); // distance voxel-camera
          double dist_target =
              sqrt(pow(camera_.GetPosition()(0) -
                           target_image.GetDepth()[idx].x, 2) +
                   pow(camera_.GetPosition()(1) -
                           target_image.GetDepth()[idx].y, 2) +
                   pow(camera_.GetPosition()(2) -
                           target_image.GetDepth()[idx].z, 2));
          if (shadows >= 0 && (dist_target > distance + STEP || dist_target == distance)) { // Shadows
            result.at<cv::Vec3b>(raw_image_.rows - i - 1, j) =
                raw_image_.at<cv::Vec3b>(coord(1), coord(0));
          }
          if (shadows <= 0 && (dist_target <= distance + STEP)) {
            result.at<cv::Vec3b>(raw_image_.rows - i - 1, j) =
                raw_image_.at<cv::Vec3b>(coord(1), coord(0));
          }
        }
      }
    }
  }
  return Image(result, target_image.GetCamera());
}

cv::Mat ProcessedImage::Warp2(const ProcessedImage &target_image,
                              cv::Mat &visible,
                              cv::Mat &occluded) const {
  cv::Mat mask(raw_image_.rows, raw_image_.cols, CV_8UC1, cv::Scalar(0));
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < mask.rows; ++i) {
    for (int j = 0; j < mask.cols; j++) {
      int idx = i * mask.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          int idx2 = (raw_image_.rows - coord(1) - 1) * raw_image_.cols +
                     coord(0);
          double distance =
              sqrt(pow(static_cast<double>(depth_[idx2].x) -
                       camera_.GetPosition()(0), 2) +
                   pow(static_cast<double>(depth_[idx2].y) -
                       camera_.GetPosition()(1), 2) +
                   pow(static_cast<double>(depth_[idx2].z) -
                       camera_.GetPosition()(2), 2)); // distance voxel-camera
          double dist_target =
              sqrt(pow(camera_.GetPosition()(0) -
                       target_image.GetDepth()[idx].x, 2) +
                   pow(camera_.GetPosition()(1) -
                       target_image.GetDepth()[idx].y, 2) +
                   pow(camera_.GetPosition()(2) -
                       target_image.GetDepth()[idx].z, 2));
          mask.at<uchar>(raw_image_.rows - i - 1, j) = dist_target <= distance + STEP;
          if (mask.at<uchar>(raw_image_.rows - i - 1, j)) {
            visible.at<cv::Vec3b>(raw_image_.rows - i - 1, j) =
              raw_image_.at<cv::Vec3b>(coord(1), coord(0));
          } else {
            occluded.at<cv::Vec3b>(raw_image_.rows - i - 1, j) =
              raw_image_.at<cv::Vec3b>(coord(1), coord(0));
          }
        }
      }
    }
  }
  return mask;
}

Eigen::Vector2i ProcessedImage::WarpPixel(int i, int j, const ProcessedImage &target_image) const {
  int idx = i * raw_image_.cols + j;
  return camera_.Project(
      static_cast<double>(target_image.GetDepth()[idx].x),
      static_cast<double>(target_image.GetDepth()[idx].y),
      static_cast<double>(target_image.GetDepth()[idx].z));
}

Eigen::Vector2i ProcessedImage::WarpPixel(Eigen::Vector2i coord, const ProcessedImage &target_image) const {
  return WarpPixel(raw_image_.rows - coord(1) - 1, coord(0), target_image);
}

Image ProcessedImage::Warp2(const ProcessedImage &target_image, const cv::Mat &delta, int shadows) const {
  return ProcessedImage(Image(delta, camera_), depth_).Warp2(target_image, shadows);
}

Image ProcessedImage::Warp2f(const ProcessedImage &target_image, const cv::Mat &delta, int shadows) const {
  cv::Mat result(delta.rows, delta.cols, delta.type(), cv::Scalar(0, 0, 0));
  if (shadows == -2)
    delta.copyTo(result);
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        // Check if the point is seen in both images
        // (taking into account occlusions)
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          int idx2 = (raw_image_.rows - coord(1) - 1) * raw_image_.cols + coord(0);
          double distance =
              sqrt(pow(static_cast<double>(depth_[idx2].x) -
                           camera_.GetPosition()(0), 2) +
                   pow(static_cast<double>(depth_[idx2].y) -
                           camera_.GetPosition()(1), 2) +
                   pow(static_cast<double>(depth_[idx2].z) -
                           camera_.GetPosition()(2), 2)); // distance voxel-camera
          double dist_target =
              sqrt(pow(camera_.GetPosition()(0) -
                           target_image.GetDepth()[idx].x, 2) +
                   pow(camera_.GetPosition()(1) -
                           target_image.GetDepth()[idx].y, 2) +
                   pow(camera_.GetPosition()(2) -
                           target_image.GetDepth()[idx].z, 2));
          if (dist_target == distance || shadows == 0 ||
              (shadows > 0 && dist_target > distance + STEP) || // occluded
              (shadows < 0 && dist_target <= distance + STEP))  // visible
            result.at<double>(raw_image_.rows - i - 1, j) =
                delta.at<double>(coord(1), coord(0));
        }
      }
    }
  }
  return Image(result, target_image.GetCamera());
}

Image ProcessedImage::WarpFast(const ProcessedImage &target_image) const {
  cv::Mat result(raw_image_.rows, raw_image_.cols, raw_image_.type(),
                 cv::Scalar(0, 0, 0));
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          int idx2 = (raw_image_.rows - coord(1) - 1) * raw_image_.cols + coord(0);
          double distance =
              sqrt(pow(static_cast<double>(depth_[idx2].x) -
                           camera_.GetPosition()(0), 2) +
                   pow(static_cast<double>(depth_[idx2].y) -
                           camera_.GetPosition()(1), 2) +
                   pow(static_cast<double>(depth_[idx2].z) -
                           camera_.GetPosition()(2), 2));
          double dist_target =
              sqrt(pow(camera_.GetPosition()(0) -
                           static_cast<double>(target_image.GetDepth()[idx].x), 2) +
                   pow(camera_.GetPosition()(1) -
                           static_cast<double>(target_image.GetDepth()[idx].y), 2) +
                   pow(camera_.GetPosition()(2) -
                           static_cast<double>(target_image.GetDepth()[idx].z), 2));
          if (dist_target <= distance + STEP) {
            result.at<cv::Vec3b>(result.rows - i - 1, j) =
                raw_image_.at<cv::Vec3b>(coord(1), coord(0));
          }
        }
      }
    }
  }
  return Image(result, target_image.GetCamera());
}

Image ProcessedImage::WarpShadows(const ProcessedImage &target_image) const {
  cv::Mat result(raw_image_.rows, raw_image_.cols, raw_image_.type(),
                 cv::Scalar(0, 0, 0));
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          int idx2 = (raw_image_.rows - coord(1) - 1) * raw_image_.cols + coord(0);
          double distance =
              sqrt(pow(static_cast<double>(depth_[idx2].x) -
                           camera_.GetPosition()(0), 2) +
                   pow(static_cast<double>(depth_[idx2].y) -
                           camera_.GetPosition()(1), 2) +
                   pow(static_cast<double>(depth_[idx2].z) -
                           camera_.GetPosition()(2), 2));
          double dist_target =
              sqrt(pow(camera_.GetPosition()(0) -
                           target_image.GetDepth()[idx].x, 2) +
                   pow(camera_.GetPosition()(1) -
                           target_image.GetDepth()[idx].y, 2) +
                   pow(camera_.GetPosition()(2) -
                           target_image.GetDepth()[idx].z, 2));
          if (dist_target > distance + STEP || dist_target == distance) {
            result.at<cv::Vec3b>(result.rows - i - 1, j) =
                raw_image_.at<cv::Vec3b>(coord(1), coord(0));
          }
        }
      }
    }
  }
  return Image(result, target_image.GetCamera());
}

Image ProcessedImage::WarpHalf(const ProcessedImage &target_image, bool interpolate) const {
  cv::Mat result(raw_image_.rows, raw_image_.cols, raw_image_.type(),
                 cv::Scalar(0, 0, 0));
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; i+=2) {
    for (int j = 0; j < result.cols; j+=2) {
      int idx = i * result.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          int idx2 = (raw_image_.rows - coord(1) - 1) * raw_image_.cols + coord(0);
          double distance =
              sqrt(pow(static_cast<double>(depth_[idx2].x) -
                           camera_.GetPosition()(0), 2) +
                   pow(static_cast<double>(depth_[idx2].y) -
                           camera_.GetPosition()(1), 2) +
                   pow(static_cast<double>(depth_[idx2].z) -
                           camera_.GetPosition()(2), 2));
          double dist_target =
              sqrt(pow(camera_.GetPosition()(0) -
                           target_image.GetDepth()[idx].x, 2) +
                   pow(camera_.GetPosition()(1) -
                           target_image.GetDepth()[idx].y, 2) +
                   pow(camera_.GetPosition()(2) -
                           target_image.GetDepth()[idx].z, 2));
          if (dist_target <= distance + STEP) {
            result.at<cv::Vec3b>(result.rows - i - 1, j) =
                raw_image_.at<cv::Vec3b>(coord(1), coord(0));
          }
        }
      }
    }
  }

  if (interpolate)
    Interpolate(result);

  return Image(result, target_image.GetCamera());
}

Image ProcessedImage::WarpFastest(const ProcessedImage &target_image) const {
  cv::Mat result(raw_image_.rows, raw_image_.cols, raw_image_.type(),
                 cv::Scalar(0, 0, 0));
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          result.at<cv::Vec3b>(result.rows - i - 1, j) =
              raw_image_.at<cv::Vec3b>(coord(1), coord(0));
        }
      }
    }
  }
  return Image(result, target_image.GetCamera());
}

cv::Mat ProcessedImage::UnWarp(const ProcessedImage &target_image, const cv::Mat &delta) const {
  cv::Mat result(delta.rows, delta.cols, delta.type(), cv::Scalar(0));
  cv::Mat zbuffer(delta.rows, delta.cols, CV_64FC1, cv::Scalar(100000));
  Eigen::Vector2i coord, coord0;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < delta.rows; ++i) {
    for (int j = 0; j < delta.cols; j++) {
      int idx = i * delta.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5 && delta.at<float>(delta.rows - i - 1, j) != 0) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        // Check if the point is seen in both images
        // (taking into account occlusions)
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          int idx2 = (result.rows - coord(1) - 1) * result.cols + coord(0);
          coord0 = target_image.GetCamera().Project(
              static_cast<double>(depth_[idx2].x),
              static_cast<double>(depth_[idx2].y),
              static_cast<double>(depth_[idx2].z));
          if (coord0(1) > 0 && coord0(1) < result.rows &&
              coord0(0) > 0 && coord0(0) < result.cols) {
            // Remove edge artifacts
            if (coord0(0) < j-1 || coord0(0) > j+1 ||
                coord0(1) < delta.rows-i-2 || coord0(1) > delta.rows-i) {
              int idx0 = (result.rows - coord0(1) - 1) * result.cols + coord0(0);
              double distance =
                sqrt(pow(static_cast<double>(depth_[idx2].x) -
                             target_image.GetCamera().GetPosition()(0), 2) +
                     pow(static_cast<double>(depth_[idx2].y) -
                             target_image.GetCamera().GetPosition()(1), 2) +
                     pow(static_cast<double>(depth_[idx2].z) -
                             target_image.GetCamera().GetPosition()(2), 2)); // distance voxel-camera
              if (distance < zbuffer.at<double>(coord0(1), coord0(0))) {
                zbuffer.at<double>(coord0(1), coord0(0)) = distance;
                double dist_target =
                  sqrt(pow(target_image.GetCamera().GetPosition()(0) -
                                target_image.GetDepth()[idx0].x, 2) +
                        pow(target_image.GetCamera().GetPosition()(1) -
                                target_image.GetDepth()[idx0].y, 2) +
                        pow(target_image.GetCamera().GetPosition()(2) -
                                target_image.GetDepth()[idx0].z, 2));
                if (dist_target > distance - STEP) {
                  //if (result.at<float>(coord0(1), coord0(0)) > delta.at<float>(delta.rows - i - 1, j) || result.at<float>(coord0(1), coord0(0)) == 0)
                  result.at<float>(coord0(1), coord0(0)) =
                    delta.at<float>(delta.rows - i - 1, j);
                }
              }
            }
          }
        }
      }
    }
  }

  //Interpolate(result);
  return result;
}

cv::Mat ProcessedImage::UnWarp(const ProcessedImage &target_image, const cv::Mat &delta, int incertitudes) const {
  cv::Mat result(delta.rows, delta.cols, delta.type(), cv::Scalar(0, 0, 0));
  Eigen::Vector2i coord, coord0;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < delta.rows; ++i) {
    for (int j = 0; j < delta.cols; j++) {
      int idx = i * delta.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5 && cv::norm(delta.at<cv::Vec3b>(delta.rows - i - 1, j)) != 0) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          int idx2 = (result.rows - coord(1) - 1) * result.cols + coord(0);
          coord0 = target_image.GetCamera().Project(
              static_cast<double>(depth_[idx2].x),
              static_cast<double>(depth_[idx2].y),
              static_cast<double>(depth_[idx2].z));
          if (coord0(1) > 0 && coord0(1) < result.rows &&
              coord0(0) > 0 && coord0(0) < result.cols) {
            //if (coord0(0) < j-1 || coord0(0) > j+1 ||
            //    coord0(1) < delta.rows-i-2 || coord0(1) > delta.rows-i) {
              int idx0 = (result.rows - coord0(1) - 1) * result.cols + coord0(0);
              double distance =
                sqrt(pow(static_cast<double>(depth_[idx2].x) -
                             target_image.GetCamera().GetPosition()(0), 2) +
                     pow(static_cast<double>(depth_[idx2].y) -
                             target_image.GetCamera().GetPosition()(1), 2) +
                     pow(static_cast<double>(depth_[idx2].z) -
                             target_image.GetCamera().GetPosition()(2), 2)); // distance voxel-camera
              double dist_target =
                sqrt(pow(target_image.GetCamera().GetPosition()(0) -
                              target_image.GetDepth()[idx0].x, 2) +
                      pow(target_image.GetCamera().GetPosition()(1) -
                              target_image.GetDepth()[idx0].y, 2) +
                      pow(target_image.GetCamera().GetPosition()(2) -
                              target_image.GetDepth()[idx0].z, 2));
              if (incertitudes == 2 && dist_target <= distance - 0.0125) {
                result.at<cv::Vec3b>(delta.rows - i - 1, j) =
                  delta.at<cv::Vec3b>(delta.rows - i - 1, j);
              }
              if (incertitudes == -2 && dist_target > distance - 0.0125) {
                result.at<cv::Vec3b>(delta.rows - i - 1, j) =
                  delta.at<cv::Vec3b>(delta.rows - i - 1, j);
              }
              if (incertitudes == -1 || incertitudes == 1 && dist_target <= distance - STEP) {
                result.at<cv::Vec3b>(coord0(1), coord0(0)) =
                  delta.at<cv::Vec3b>(delta.rows - i - 1, j);
              }
              if (incertitudes == -1 || incertitudes == 0 && dist_target > distance - STEP) {
                result.at<cv::Vec3b>(coord0(1), coord0(0)) =
                  delta.at<cv::Vec3b>(delta.rows - i - 1, j);
              }
            //}
          }
        }
      }
    }
  }
  return result;
}

cv::Mat ProcessedImage::WarpEpiline(const ProcessedImage &target, const Eigen::Vector2i& coord) const {
  Eigen::Matrix3d F = camera_.ComputeF(target.GetCamera());
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_8UC1,
                  cv::Scalar(0));
    Eigen::Vector3d pt;
    pt << coord(0), coord(1), 1;
    Eigen::Vector3d line = F * pt;
    cv::line(
        result, cv::Point(0, -line(2) / line(1)),
        cv::Point(result.cols, -(line(2) + line(0) * result.cols) / line(1)),
        cv::Scalar(255));
  return result;
}

//cv::Mat ProcessedImage::WarpLine(int i, int j, const ProcessedImage &target_image) const {
cv::Mat ProcessedImage::WarpLine(const ProcessedImage &target_image, const Eigen::Vector2i& coord) const {
  //int idx = i * raw_image_.cols + j;
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_64FC1, cv::Scalar(NAN));
  //cv::Mat mask(raw_image_.rows, raw_image_.cols, CV_64FC1, cv::Scalar(1.0));
  //cv::Mat result(raw_image_.rows, raw_image_.cols, CV_64FC1,
  //               cv::Scalar(0));
  int idx = (result.rows - coord(1) - 1) * result.cols + coord(0);
  // voxel of interest
  Eigen::Vector2i voxcoord = target_image.GetCamera().Project(
      static_cast<double>(depth_[idx].x),
      static_cast<double>(depth_[idx].y),
      static_cast<double>(depth_[idx].z));
  Eigen::Vector3d voxtrans = target_image.GetCamera().Transform(
      static_cast<double>(depth_[idx].x),
      static_cast<double>(depth_[idx].y),
      static_cast<double>(depth_[idx].z));
  //camera
  Eigen::Vector2i camcoord = target_image.GetCamera().Project(
      static_cast<double>(camera_.GetPosition()(0)),
      static_cast<double>(camera_.GetPosition()(1)),
      static_cast<double>(camera_.GetPosition()(2)));
  Eigen::Vector3d camtrans = target_image.GetCamera().Transform(
    static_cast<double>(camera_.GetPosition()(0)),
    static_cast<double>(camera_.GetPosition()(1)),
    static_cast<double>(camera_.GetPosition()(2)));

  target_image.GetDepthImage().copyTo(result);

  //4 corners
  int left((voxcoord(0)*camcoord(1)-camcoord(0)*voxcoord(1))/(voxcoord(0)-camcoord(0)));
  int right(((voxcoord(1)-camcoord(1))*result.cols+voxcoord(0)*camcoord(1)-camcoord(0)*voxcoord(1))/(voxcoord(0)-camcoord(0)));
  int top((voxcoord(1)*camcoord(0)-camcoord(1)*voxcoord(0))/(voxcoord(1)-camcoord(1)));
  int bot(((voxcoord(0)-camcoord(0))*result.rows+voxcoord(1)*camcoord(0)-camcoord(1)*voxcoord(0))/(voxcoord(1)-camcoord(1)));

  bool lt(left >= 0 && left < result.rows);
  bool rt(right >= 0 && right < result.rows);
  bool tp(top >= 0 && top < result.cols);
  bool bt(bot >= 0 && bot < result.cols);

  double mind, maxd, distance;
  cv::minMaxLoc(result, &mind, &maxd);

  // If there is no voxel in the frame
  if (!lt && !rt && !tp && !bt)
    return result;

  if (target_image.GetCamera().GetPosition() == camera_.GetPosition())
    return result;

  Eigen::Vector2d f(target_image.GetCamera().GetK()(0, 0), target_image.GetCamera().GetK()(1, 1));
  Eigen::Vector2d ccoords(target_image.GetCamera().GetK()(0, 2), target_image.GetCamera().GetK()(1, 2));
  Eigen::Vector2i scoords(result.cols, result.rows);

  //Delta-uv
  Eigen::Vector2d vcoords(f(0)*(voxtrans(2)*camtrans(0)-voxtrans(0)*camtrans(2)),f(1)*(voxtrans(2)*camtrans(1)-voxtrans(1)*camtrans(2)));
  //Eigen::Vector2d vcoords(camcoord(0)-voxcoord(0), camcoord(1)-voxcoord(1));
  //Delta-max
  double vmax(std::max(std::max(vcoords(0),vcoords(1)),-std::max(-vcoords(0),-vcoords(1))));
  //delta-uv
  Eigen::Vector2d wcoords = vcoords/vmax;
  //axis
  int axis(wcoords(0) > wcoords(1) ? 0 : 1);
  //Delta-P
  Eigen::Vector3d dtrans = voxtrans-camtrans;
  //uv-inf
  Eigen::Vector2d icoords(f(0)*dtrans(0)/dtrans(2)+ccoords(0),f(1)*dtrans(1)/dtrans(2)+ccoords(1));

  int k0(vmax/(dtrans(2)*camtrans(2)));
  int k1(vmax/(dtrans(2)*voxtrans(2)));
  int kstep(vmax < 0 ? 1 : -1); // kstep == -1 -> kstart >= kend >= 1
  int kstart(kstep > 0 ? -icoords(axis) : scoords(axis)-1-icoords(axis)); // by default start from the edge of the frame
  int kend(0);
  // Is the end point outside the frame ?
  if (icoords(axis) < 0 || icoords(axis) > scoords(axis)-1)
    kend = kstep < 0 ? -icoords(axis) : scoords(axis)-1-icoords(axis);

  // Is the camera in the frame ?
  if ((kstep < 0 && k0 > 0 && k0 < kstart) || (kstep > 0 && k0 < 0 && k0 > kstart))
    kstart = k0; // start from here

  // Is the voxel in the frame ?
  if ((kstep < 0 && k1 > 0 && k1 < kstart) || (kstep > 0 && k1 < 0 && k1 > kstart))
    kstart = k1; // start from here

    if ((icoords(1-axis) < 0                 && kstart*wcoords(1-axis)+icoords(1-axis) < 0) ||
        (icoords(1-axis) > scoords(1-axis)-1 && kstart*wcoords(1-axis)+icoords(1-axis) > scoords(1-axis)-1)) {
        std::cout << icoords(1-axis) << '|' << scoords(1-axis) << '|' << kstart*wcoords(1-axis)+icoords(1-axis) << ' ' << std::flush;
        return result;
    }
    else
      std::cout << "trying" << std::endl;

  Eigen::Vector3d voxel(depth_[idx].x,depth_[idx].y,depth_[idx].z);
  Eigen::Vector3d v;

  //double a = depth_image_.at<double>(coord(1),coord(0)) *
  //           depth_image_.at<double>(coord(1),coord(0));
  double a = (camera_.GetPosition() - voxel).squaredNorm();
  double b = (camera_.GetPosition() - voxel).dot(
              camera_.GetPosition() - target_image.GetCamera().GetPosition());
    /*(camera_.GetPosition()(0) - target_image.GetCamera().GetPosition()(0)) *
    (camera_.GetPosition()(0) - voxel(0)) +
    (camera_.GetPosition()(1) - target_image.GetCamera().GetPosition()(1)) *
    (camera_.GetPosition()(1) - voxel(1)) +
    (camera_.GetPosition()(2) - target_image.GetCamera().GetPosition()(2)) *
    (camera_.GetPosition()(2) - voxel(2));*/
  double c = (camera_.GetPosition() -
              target_image.GetCamera().GetPosition()).squaredNorm();
    /*pow(camera_.GetPosition()(0) - target_image.GetCamera().GetPosition()(0), 2) +
    pow(camera_.GetPosition()(1) - target_image.GetCamera().GetPosition()(1), 2) +
    pow(camera_.GetPosition()(2) - target_image.GetCamera().GetPosition()(2), 2);*/

  double intersect(1); // -1 if not reversed
  for (int k = kend-kstep; k != kstart-2*kstep; k-=kstep) { // reversed for efficiency
    double t(vmax/(dtrans(2)*dtrans(2)*k)-camtrans(2)/dtrans(2));
    //v = t*voxel + (1.0-t)*camera_.GetPosition();
    v = t*voxel + (1.0-t)*camera_.GetPosition() - target_image.GetCamera().GetPosition();
    //result.at<double>(k*wcoords(1)+icoords(1),k*wcoords(0)+icoords(0)) = std::max(mind,std::min(maxd,sqrt(distance)));
    double distance = v.squaredNorm();//t*t*a-2*t*b+c;//
      /*pow(static_cast<double>(v(0)) -
       target_image.GetCamera().GetPosition()(0), 2) +
      pow(static_cast<double>(v(1)) -
       target_image.GetCamera().GetPosition()(1), 2) +
      pow(static_cast<double>(v(2)) -
       target_image.GetCamera().GetPosition()(2), 2);*/
    //result.at<double>(k*wcoords(1)+icoords(1),k*wcoords(0)+icoords(0)) = maxd;

    Eigen::Vector2i kcoords = (k*wcoords+icoords).cast<int>();
    // Intersections
    double dist_target = target_image.GetDepthImage().at<double>(kcoords(1),kcoords(0));//(k*wcoords(1)+icoords(1),k*wcoords(0)+icoords(0));
    dist_target *= dist_target;

    /*int idx = i * result.cols + j;
    coords(1)=result.rows - i - 1
    coords(0)=j*/
    //int idx0 = (raw_image_.rows - (k*wcoords(1)+icoords(1)) - 1) * raw_image_.cols + (k*wcoords(0)+icoords(0));
    int idx0 = (raw_image_.rows - kcoords(1) - 1) * raw_image_.cols + kcoords(0);
    double dist_target2 =
      pow(target_image.GetCamera().GetPosition()(0) -
          target_image.GetDepth()[idx0].x, 2) +
      pow(target_image.GetCamera().GetPosition()(1) -
          target_image.GetDepth()[idx0].y, 2) +
      pow(target_image.GetCamera().GetPosition()(2) -
          target_image.GetDepth()[idx0].z, 2);

    std::cout << kcoords(1) << ',' << kcoords(0) << '\t' << k*wcoords(1)+icoords(1) << ',' << k*wcoords(0)+icoords(0) << '\t' << idx0 << '\t' << dist_target << '\t' << dist_target2 << '\n' << std::flush;
    if (distance-dist_target <= 0 && intersect >= 0) { // >= and <= if not reversed
      //result.at<double>(k*wcoords(1)+icoords(1),k*wcoords(0)+icoords(0)) = maxd;
      result.at<double>(kcoords(1),kcoords(0)) = maxd;
      return result;
    }
    intersect = distance-dist_target;
  }
  return result;
}

void ProcessedImage::WarpDepthSparse(const ProcessedImage &target_image) {//image.depth(coords) = target.depth(i,j)
  //cv::Mat result(raw_image_.rows, raw_image_.cols, raw_image_.type(),
  //               cv::Scalar(0, 0, 0));
  //cv::Mat zbuffer(raw_image_.rows, raw_image_.cols, CV_64FC1,
  //                cv::Scalar(0));
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < raw_image_.rows; ++i) {
    for (int j = 0; j < raw_image_.cols; j++) { // for every voxel of the target
      int idx = i * raw_image_.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z)); // in image pov
        if (coord(1) > 0 && coord(1) < raw_image_.rows &&
            coord(0) > 0 && coord(0) < raw_image_.cols) {
          double distance =
              sqrt(pow(static_cast<double>(target_image.GetDepth()[idx].x) -
                           camera_.GetPosition()(0), 2) +
                   pow(static_cast<double>(target_image.GetDepth()[idx].y) -
                           camera_.GetPosition()(1), 2) +
                   pow(static_cast<double>(target_image.GetDepth()[idx].z) -
                           camera_.GetPosition()(2), 2));
          int depth_idx =
                (raw_image_.rows - coord(1) - 1) * raw_image_.cols +
                coord(0);
          if (background_[depth_idx].w <= 0.5) {//zbuffer.at<double>(coord(1), coord(0))) { // take farthest away
            //zbuffer.at<double>(coord(1), coord(0)) = distance;
            //int depth_idx =
            //    (raw_image_.rows - coord(1) - 1) * raw_image_.cols +
            //    coord(0);
            //double dist_target =
            //    sqrt(pow(camera_.GetPosition()(0) -
            //                 depth_[depth_idx].x, 2) +
            //         pow(camera_.GetPosition()(1) -
            //                 depth_[depth_idx].y, 2) +
            //         pow(camera_.GetPosition()(2) -
            //                 depth_[depth_idx].z, 2));
            //if (dist_target >= distance - STEP) {
              //result.at<cv::Vec3b>(coord(1), coord(0)) =
              //    raw_image_.at<cv::Vec3b>(raw_image_.rows - i - 1, j);
              background_[depth_idx] = target_image.GetDepth()[idx];
            //}
          } else {
            double dist_image =
                sqrt(pow(camera_.GetPosition()(0) -
                             background_[depth_idx].x, 2) +
                     pow(camera_.GetPosition()(1) -
                             background_[depth_idx].y, 2) +
                     pow(camera_.GetPosition()(2) -
                             background_[depth_idx].z, 2));
            if (dist_image < distance)
              background_[depth_idx] = target_image.GetDepth()[idx];
          }
        }
      }
    }
  }

  return;// Image(result, target_image.GetCamera());
}

void ProcessedImage::WarpDepth(const ProcessedImage &target_image) {//image.depth(i,j) = target.depth(coord)
  if (target_image.GetCamera().GetPosition() == camera_.GetPosition())
    return;

  Eigen::Vector3d camtrans = target_image.GetCamera().Transform(
    static_cast<double>(camera_.GetPosition()(0)),
    static_cast<double>(camera_.GetPosition()(1)),
    static_cast<double>(camera_.GetPosition()(2)));

  double c = (camera_.GetPosition() -
              target_image.GetCamera().GetPosition()).squaredNorm();

  Eigen::Vector2d f(target_image.GetCamera().GetK()(0, 0),
                    target_image.GetCamera().GetK()(1, 1));
  Eigen::Vector2d ccoords(target_image.GetCamera().GetK()(0, 2),
                          target_image.GetCamera().GetK()(1, 2));
  Eigen::Vector2i scoords(raw_image_.cols, raw_image_.rows);

  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < raw_image_.rows; ++i) {
    for (int j = 0; j < raw_image_.cols; j++) {
      int idx = i * raw_image_.cols + j;
      if (depth_[idx].w > 0.5) {
        Eigen::Vector3d voxtrans = target_image.GetCamera().Transform(
          static_cast<double>(depth_[idx].x),
          static_cast<double>(depth_[idx].y),
          static_cast<double>(depth_[idx].z));

        //Delta-uv
        Eigen::Vector2d vcoords = f.cwiseProduct(voxtrans(2)*camtrans.head<2>()-camtrans(2)*voxtrans.head<2>());
        //Delta-max
        int axis;
        double vmax = vcoords.cwiseAbs().maxCoeff(&axis);
        Eigen::Vector2d wcoords = vcoords/vmax;
        //Delta-P
        Eigen::Vector3d dtrans = voxtrans-camtrans;
        //uv-inf
        Eigen::Vector2d icoords = f.cwiseProduct(dtrans.head<2>()/dtrans(2))+ccoords;
        int k0(vmax/(dtrans(2)*camtrans(2)));
        int k1(vmax/(dtrans(2)*voxtrans(2)));
        int kstep(vmax < 0 ? 1 : -1); // kstep == -1 -> kstart >= kend >= 1
        int kstart(kstep > 0 ? -icoords(axis) : scoords(axis)-1-icoords(axis)); // by default start from the edge of the frame
        int kend(0); // by default end the infinity point

        // Is the line in the frame ?
        if ((icoords(1-axis) < 0                 && kstart*wcoords(1-axis)+icoords(1-axis) < 0) ||
            (icoords(1-axis) > scoords(1-axis)-1 && kstart*wcoords(1-axis)+icoords(1-axis) > scoords(1-axis)-1)) {
            continue;
        }
        // Is the end point outside the frame ?
        if (icoords(axis) < 0 && kstep < 0)
          kend = -icoords(axis);
        if (icoords(axis) < 0 && kstart*wcoords(axis)+icoords(axis) < 0)//&& kstep > 0)
          continue;
        if (icoords(axis) > scoords(axis)-1 && kstep > 0)
          kend = scoords(axis)-1-icoords(axis);
        if (icoords(axis) > scoords(axis)-1 && kstart*wcoords(axis)+icoords(axis) > scoords(axis)-1)//&& kstep < 0)
          continue;

        // Is the camera in the frame ?
        if ((kstep < 0 && k0 > 0 && k0 < kstart) || (kstep > 0 && k0 < 0 && k0 > kstart))
          kstart = k0; // start from here

        // Is the voxel in the frame ?
        if ((kstep < 0 && k1 > 0 && k1 < kstart) || (kstep > 0 && k1 < 0 && k1 > kstart))
          kstart = k1; // start from here

        Eigen::Vector3d voxel(depth_[idx].x,depth_[idx].y,depth_[idx].z);
        Eigen::Vector3d v;

        //double a = depth_image_.at<double>(coord(1),coord(0)) *
        //           depth_image_.at<double>(coord(1),coord(0));
        double a = (camera_.GetPosition() - voxel).squaredNorm();
        double b = (camera_.GetPosition() - voxel).dot(
                    camera_.GetPosition() - target_image.GetCamera().GetPosition());

        double intersect(1);
        for (int k = kend-kstep; k != kstart-2*kstep; k-=kstep) { // reversed for efficiency
          double t(vmax/(dtrans(2)*dtrans(2)*k)-camtrans(2)/dtrans(2));
          //v = t*voxel + (1.0-t)*camera_.GetPosition() - target_image.GetCamera().GetPosition();
          //double distance = v.squaredNorm();
          double distance =  t*t*a-2*t*b+c;

          Eigen::Vector2i kcoords = (k*wcoords+icoords).cast<int>();
          if (kcoords(1-axis) < 0 || kcoords(1-axis) > scoords(1-axis)-1) {
            //std::cout << "1-a " << std::flush;
            continue;
          }
          if (kcoords(axis) < 0 || kcoords(axis) > scoords(axis)-1) {
            //std::cout << "a " << std::flush;
            continue;
          }
          // Intersections
          //double dist_target = target_image.GetDepthImage().at<double>(kcoords(1),kcoords(0));//(k*wcoords(1)+icoords(1),k*wcoords(0)+icoords(0));
          //dist_target *= dist_target;
          int idx0 = (raw_image_.rows - kcoords(1) - 1) * raw_image_.cols + kcoords(0);
          if (target_image.GetDepth()[idx0].w > 0.5) {
            double dist_target =
              pow(target_image.GetCamera().GetPosition()(0) -
                  target_image.GetDepth()[idx0].x, 2) +
              pow(target_image.GetCamera().GetPosition()(1) -
                  target_image.GetDepth()[idx0].y, 2) +
              pow(target_image.GetCamera().GetPosition()(2) -
                  target_image.GetDepth()[idx0].z, 2);
            if (distance <= dist_target && intersect >= 0) {
              double d =
                pow(camera_.GetPosition()(0) -
                    target_image.GetDepth()[idx0].x, 2) +
                pow(camera_.GetPosition()(1) -
                    target_image.GetDepth()[idx0].y, 2) +
                pow(camera_.GetPosition()(2) -
                    target_image.GetDepth()[idx0].z, 2);
              double e =
                pow(camera_.GetPosition()(0) -
                    background_[idx].x, 2) +
                pow(camera_.GetPosition()(1) -
                    background_[idx].y, 2) +
                pow(camera_.GetPosition()(2) -
                    background_[idx].z, 2);
              if (d > e)
                background_[idx] = target_image.GetDepth()[idx0];
              break;  // we found the background voxel
            }
            intersect = distance-dist_target;
          }
        }
      }
    }
  }
  return;
}
