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
Image ProcessedImage::DepthMap(const ProcessedImage &target_image) const {
  cv::Mat result(raw_image_.rows, raw_image_.cols,  CV_64FC1,
                 cv::Scalar(NAN));
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (GetDepth()[idx].w > 0.5) {
        double distance =
            sqrt(pow(static_cast<double>(depth_[idx].x) -
                     target_image.GetCamera().GetPosition()(0), 2) +
                 pow(static_cast<double>(depth_[idx].y) -
                     target_image.GetCamera().GetPosition()(1), 2) +
                 pow(static_cast<double>(depth_[idx].z) -
                     target_image.GetCamera().GetPosition()(2), 2)); // distance voxel-camera
        result.at<double>(raw_image_.rows - i - 1, j) = distance;
      }
    }
  }
  cv::normalize(result, result, 0.0f, 1.0f, cv::NORM_MINMAX);
  return Image(result, GetCamera());
}

Image ProcessedImage::DepthDiff(const ProcessedImage &target_image) const {
  cv::Mat result(raw_image_.rows, raw_image_.cols,  CV_64FC1,
                 cv::Scalar(NAN));
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
        if (coord(1) > 0 && coord(1) < raw_image_.rows && coord(0) > 0 &&
            coord(0) < raw_image_.cols) {
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
          result.at<double>(result.rows - i - 1, j) = /*std::abs(dist_target -*/ distance/*)*/;
        }
      }
    }
  }
  cv::normalize(result, result, 0.0f, 1.0f, cv::NORM_MINMAX);
  return Image(result, target_image.GetCamera());
}

cv::Mat ProcessedImage::ComputeDepthImage() {
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_64FC1, cv::Scalar(NAN));
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (depth_[idx].w > 0.5) {
        result.at<double>(result.rows - i - 1, j) =
          sqrt(pow(static_cast<double>(depth_[idx].x) -
                   camera_.GetPosition()(0), 2) +
               pow(static_cast<double>(depth_[idx].y) -
                   camera_.GetPosition()(1), 2) +
               pow(static_cast<double>(depth_[idx].z) -
                   camera_.GetPosition()(2), 2));
      }
    }
  }
  depth_image_ = result;
  return depth_image_;
}

cv::Mat ProcessedImage::ComputeNormalImage() {
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_64FC1, cv::Scalar(NAN));
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (background_[idx].w > 0.5) {
        result.at<double>(result.rows - i - 1, j) = background_[idx].x;
      }
    }
  }
  background_image_ = result;
  return result;
}

cv::Mat ProcessedImage::ComputeCoordImage(const ProcessedImage &target_image) {
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_64FC3, cv::Scalar(0,0,0));
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = WarpPixel(i, j, target_image);
        result.at<cv::Vec3d>(result.rows - i - 1, j) =
          cv::Vec3d((raw_image_.rows - coord(1) - 1)/(double)raw_image_.rows,
                     coord(0)/(double)raw_image_.cols, 1);
      }
    }
  }
  //background_image_ = result;
  return result;
}

void ProcessedImage::DigDepthImage(const ProcessedImage &target_image) {
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < depth_image_.rows; ++i) {
    for (int j = 0; j < depth_image_.cols; j++) {
      int idx = i * depth_image_.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
            static_cast<double>(target_image.GetDepth()[idx].x),
            static_cast<double>(target_image.GetDepth()[idx].y),
            static_cast<double>(target_image.GetDepth()[idx].z));
        if (coord(1) > 0 && coord(1) < depth_image_.rows && coord(0) > 0 &&
            coord(0) < depth_image_.cols) {
          int idx2 = (depth_image_.rows - coord(1) - 1) * depth_image_.cols + coord(0);
          double distance =
              /*sqrt(pow(static_cast<double>(depth_[idx2].x) -
                           camera_.GetPosition()(0), 2) +
                   pow(static_cast<double>(depth_[idx2].y) -
                           camera_.GetPosition()(1), 2) +
                   pow(static_cast<double>(depth_[idx2].z) -
                           camera_.GetPosition()(2), 2));*/
              depth_image_.at<double>(coord(1), coord(0));
          double dist_target =
              sqrt(pow(camera_.GetPosition()(0) -
                           target_image.GetDepth()[idx].x, 2) +
                   pow(camera_.GetPosition()(1) -
                           target_image.GetDepth()[idx].y, 2) +
                   pow(camera_.GetPosition()(2) -
                           target_image.GetDepth()[idx].z, 2));
          if (dist_target > distance) {
            depth_image_.at<double>(coord(1), coord(0)) = dist_target;
            depth_[idx2] = target_image.GetDepth()[idx];
          }
        }
      }
    }
  }
}

void ProcessedImage::ComputePresenceImage() {
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_8UC1, cv::Scalar(0));
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (depth_[idx].w > 0.5)
        result.at<unsigned char>(result.rows - i - 1, j) = 255;
    }
  }
  depth_image_ = result;
}

void ProcessedImage::ComputePresenceImage(cv::Mat gt) {
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_8UC3, cv::Scalar(0,0,0));//BGR
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (depth_[idx].w > 0.5) {
        if (gt.at<cv::Vec3b>(result.rows - i - 1, j) == cv::Vec3b(255,255,255)) // White
          result.at<cv::Vec3b>(result.rows - i - 1, j)[1] = 255; // Green
        else if (gt.at<cv::Vec3b>(result.rows - i - 1, j)[1] == 255) // Green
          result.at<cv::Vec3b>(result.rows - i - 1, j) = cv::Vec3b(255,255,0); // Cyan
        else if (gt.at<cv::Vec3b>(result.rows - i - 1, j)[2] == 255) // Red
          result.at<cv::Vec3b>(result.rows - i - 1, j) = cv::Vec3b(255,0,255); // Magenta
        else // Black
          result.at<cv::Vec3b>(result.rows - i - 1, j)[2] = 255; // Red
      }
      else // Copy
        result.at<cv::Vec3b>(result.rows - i - 1, j) = gt.at<cv::Vec3b>(result.rows - i - 1, j);// = cv::Vec3b(255,255,255); // White
    }
  }
  depth_image_ = result;
}

void ProcessedImage::ComputePresenceImage(cv::Mat gt, cv::Mat mask) {
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_8UC3, cv::Scalar(0,0,0));//BGR
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (mask.at<cv::Vec3b>(result.rows - i - 1, j) != cv::Vec3b(255,255,255)) { // ignore this pixel ?
        if (depth_[idx].w > 0.5) {
          if (gt.at<cv::Vec3b>(result.rows - i - 1, j) == cv::Vec3b(255,255,255)) // White
            result.at<cv::Vec3b>(result.rows - i - 1, j)[1] = 255; // Green
          else if (gt.at<cv::Vec3b>(result.rows - i - 1, j)[1] == 255) // Green
            result.at<cv::Vec3b>(result.rows - i - 1, j) = cv::Vec3b(255,255,0); // Cyan
          else if (gt.at<cv::Vec3b>(result.rows - i - 1, j)[2] == 255) // Red
            result.at<cv::Vec3b>(result.rows - i - 1, j) = cv::Vec3b(255,0,255); // Magenta
          else // Black
            result.at<cv::Vec3b>(result.rows - i - 1, j)[2] = 255; // Red
        } else // Copy
          result.at<cv::Vec3b>(result.rows - i - 1, j) = gt.at<cv::Vec3b>(result.rows - i - 1, j);// = cv::Vec3b(255,255,255); // White
      } else
        result.at<cv::Vec3b>(result.rows - i - 1, j) = cv::Vec3b(0,0,0); // Black
    }
  }
  depth_image_ = result;
}

cv::Mat ProcessedImage::GetDepthImage() const { return depth_image_; }

void ProcessedImage::EraseBackground() {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < raw_image_.rows; ++i)
    for (int j = 0; j < raw_image_.cols; j++)
      background_[i * raw_image_.cols + j].w = 0; // < 0.5
}

void ProcessedImage::ComputeBackgroundImage() {
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_64FC1, cv::Scalar(NAN));
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (background_[idx].w > 0.5) {
        double distance =
          sqrt(pow(static_cast<double>(background_[idx].x) -
                   camera_.GetPosition()(0), 2) +
               pow(static_cast<double>(background_[idx].y) -
                   camera_.GetPosition()(1), 2) +
               pow(static_cast<double>(background_[idx].z) -
                   camera_.GetPosition()(2), 2)); // distance voxel-camera
        result.at<double>(result.rows - i - 1, j) = distance;
      }
    }
  }
  background_image_ = result;
}

cv::Mat ProcessedImage::GetBackgroundImage() const { return background_image_; }

cv::Mat ProcessedImage::Occlusion(const ProcessedImage &target_image) const {
  cv::Mat result(raw_image_.rows, raw_image_.cols, raw_image_.type(),
                 cv::Scalar(0, 0, 0));
  Eigen::Vector2i coord;
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (depth_[idx].w > 0.5) {
        coord = target_image.GetCamera().Project(
          static_cast<double>(depth_[idx].x),
          static_cast<double>(depth_[idx].y),
          static_cast<double>(depth_[idx].z));
        if (coord(1) > 0 && coord(1) < target_image.GetRawImage().rows &&
            coord(0) > 0 && coord(0) < target_image.GetRawImage().cols) {
          int idx2 = (target_image.GetRawImage().rows - coord(1) - 1) *
            target_image.GetRawImage().cols + coord(0);
          double distance =
            sqrt(pow(static_cast<double>(depth_[idx].x) -
                     camera_.GetPosition()(0), 2) +
                 pow(static_cast<double>(depth_[idx].y) -
                     camera_.GetPosition()(1), 2) +
                 pow(static_cast<double>(depth_[idx].z) -
                     camera_.GetPosition()(2), 2));
          double dist_target =
            sqrt(pow(camera_.GetPosition()(0) -
                     target_image.GetDepth()[idx2].x, 2) +
                 pow(camera_.GetPosition()(1) -
                     target_image.GetDepth()[idx2].y, 2) +
                 pow(camera_.GetPosition()(2) -
                     target_image.GetDepth()[idx2].z, 2));
          result.at<cv::Vec3b>(result.rows - i - 1, j) = cv::Vec3b(0, 0, 255); //R
          if (dist_target <= distance + STEP) { // target in front - background
            result.at<cv::Vec3b>(result.rows - i - 1, j) += cv::Vec3b(255, 0, 0); //B
          }
          if (dist_target >= distance - STEP) { // target behind - foreground
            result.at<cv::Vec3b>(result.rows - i - 1, j) += cv::Vec3b(0, 255, 0); //G
          }
        }
      }
    }
  }
  for (int i = 0; i < result.rows; ++i) {
    for (int j = 0; j < result.cols; j++) {
      int idx = i * result.cols + j;
      if (target_image.GetDepth()[idx].w > 0.5) {
        coord = camera_.Project(
          static_cast<double>(target_image.GetDepth()[idx].x),
          static_cast<double>(target_image.GetDepth()[idx].y),
          static_cast<double>(target_image.GetDepth()[idx].z));
        if (coord(1) > 0 && coord(1) < result.rows &&
            coord(0) > 0 && coord(0) < result.cols) {
          int idx2 = (result.rows - coord(1) - 1) * result.cols + coord(0);
          double distance =
            sqrt(pow(static_cast<double>(depth_[idx2].x) -
                     target_image.GetCamera().GetPosition()(0), 2) +
                 pow(static_cast<double>(depth_[idx2].y) -
                     target_image.GetCamera().GetPosition()(1), 2) +
                 pow(static_cast<double>(depth_[idx2].z) -
                     target_image.GetCamera().GetPosition()(2), 2));
          double dist_target =
            sqrt(pow(target_image.GetCamera().GetPosition()(0) -
                     target_image.GetDepth()[idx].x, 2) +
                 pow(target_image.GetCamera().GetPosition()(1) -
                     target_image.GetDepth()[idx].y, 2) +
                 pow(target_image.GetCamera().GetPosition()(2) -
                     target_image.GetDepth()[idx].z, 2));
          if (dist_target >= distance + STEP) {  // target behind - foreground
            result.at<cv::Vec3b>(coord(1), coord(0))[2] = 0; //R
          }
          if (dist_target >= distance - STEP) {
            //result.at<cv::Vec3b>(coord(1), coord(0)) += cv::Vec3b(0, 255, 0); //G
          }
        }
      }
    }
  }
  return result;
}

void ProcessedImage::UpdatePlanes(const cv::Mat& plane) {
  if (num_planes_compared_ > 0)
    planes_ = MinimumMC(planes_, plane);
  else
    planes_ = plane;

  num_planes_compared_++;
}

cv::Mat ProcessedImage::GetPlanes() { return planes_; }

cv::Mat ProcessedImage::GetMask() {
  cv::Mat result(raw_image_.rows, raw_image_.cols, CV_8UC1,
                 cv::Scalar(0));
  int mask_size(0);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < raw_image_.rows; ++i) {
    for (int j = 0; j < raw_image_.cols; j++) {
      int idx = i * raw_image_.cols + j;
      double distance =
          sqrt(pow(static_cast<double>(depth_[idx].x) -
                       camera_.GetPosition()(0), 2) +
               pow(static_cast<double>(depth_[idx].y) -
                       camera_.GetPosition()(1), 2) +
               pow(static_cast<double>(depth_[idx].z) -
                       camera_.GetPosition()(2), 2));
      double dist_target =
          sqrt(pow(static_cast<double>(background_[idx].x) -
                       camera_.GetPosition()(0), 2) +
               pow(static_cast<double>(background_[idx].y) -
                       camera_.GetPosition()(1), 2) +
               pow(static_cast<double>(background_[idx].z) -
                       camera_.GetPosition()(2), 2));
      if (dist_target >= distance + STEP) {
        result.at<uchar>(result.rows - i - 1, j) = 255;
        mask_size++;
      }
      /*if (depth_[idx].x != background_[idx].x || depth_[idx].y != background_[idx].y || depth_[idx].z != background_[idx].z) {
            result.at<uchar>(result.rows - i - 1, j) = 255;
            mask_size++;
      }*/
    }
  }
  if (mask_size_ < 0)
    mask_size_ = mask_size;
  return result;
}

int ProcessedImage::GetMaskSize() {
  cv::Mat result(raw_image_.rows, raw_image_.cols, raw_image_.type(),
                 cv::Scalar(0, 0, 0));
  if (mask_size_ < 0) {
    int mask_size(0);
    Eigen::Vector2i coord;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < raw_image_.rows; ++i) {
      for (int j = 0; j < raw_image_.cols; j++) {
        int idx = i * raw_image_.cols + j;
        if (depth_[idx].x != background_[idx].x || depth_[idx].y != background_[idx].y || depth_[idx].z != background_[idx].z) {
              mask_size++;
        }
      }
    }
    mask_size_ = mask_size;
  }
  return mask_size_;
}
