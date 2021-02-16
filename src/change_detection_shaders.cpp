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
#include <sstream>
#include <cstring>
#include <vector>
#include "fastcd/change_detector.h"
#include "utils/obj_reader.h"

#include <chrono>
#include <fstream>
#include "example.h"

#include <SDL.h>
#include <SDL_opengl.h>

struct Dataset {
  bool loaded = false;
  Mesh model;
  std::vector<fastcd::Image> images;
  std::vector<fastcd::Image> insertions;
  std::vector<fastcd::Image> removals;
};

Dataset LoadDataset(char *path, int n_images) {
  Dataset data;
  std::ostringstream filename;
  if (path[strlen(path) - 1] != '/') {
    filename << path << "/";
  } else {
    filename << path;
  }
  std::string basepath = filename.str();

  std::cout << "Loading model... " << std::flush;
  filename << "model.obj";
  try {
    data.model = ObjReader::FromFileNoMaterial(filename.str());
  } catch(std::runtime_error test) {
    std::cout << "Error reading Obj file:\n" << test.what() << std::endl;
    return Dataset();
  }
  std::cout << "Done." << std::endl;

  std::cout << "Loading images... " << std::flush;
  for (int i = 0; i < n_images; i++) {
    fastcd::Camera cam;
    filename.str("");
    filename.clear();
    filename << basepath << "cameras.xml";
    try {
      cam.ReadCalibration(filename.str(), i);
    } catch (...) {
      std::cout << "Error reading calibration!" << std::endl;
      return Dataset();
    }

    fastcd::Image img;
    filename.str("");
    filename.clear();
    filename << basepath << "images/Image" << i << ".JPG";
    if (!img.LoadImage(filename.str(), cam)) {
      std::cout << "Error reading images!" << std::endl;
      return Dataset();
    }
    data.images.push_back(img);

    if (i == 0 || !data.insertions.empty()) { // do not load if it already failed
      filename.str("");
      filename.clear();
      filename << basepath << "ground_truth/Image" << i << "gt.png";
      if (!img.LoadImage(filename.str(), cam)) {
        //std::cout << "Insertion ground truth incomplete or absent." << std::endl;
        data.insertions.clear(); // Remove all other images
      } else
        data.insertions.push_back(img);
    }

    if (i == 0 || !data.removals.empty()) { // do not load if it already failed
      filename.str("");
      filename.clear();
      filename << basepath << "ground_truth/Image" << i << "gt2.png";
      if (!img.LoadImage(filename.str(), cam)) {
        //std::cout << "Removal ground truth incomplete or absent." << std::endl;
        data.removals.clear(); // Remove all other images
      } else
        data.removals.push_back(img);
    }
  }
  std::cout << "Done." << std::endl;
  data.loaded = true;
  return data;
}

void SaveResults(char *path, const fastcd::ChangeDetector::ChangeDetectorOptions &opts,
                 double insertIoU, double insertCov,
                 double removeIoU, double removeCov,
                 std::chrono::duration<double> time) {
  std::ostringstream filename;
  filename << path;
  if (path[strlen(path) - 1] != '/')
    filename << "/";
  std::string basepath = filename.str();
  filename << "out" << SUFFIX << ".txt";

  std::ofstream output;
  output.open(filename.str(), std::ofstream::out | std::ofstream::trunc);
  // Parameters
  output << basepath <<
            "\ncache_size\t"            << opts.cache_size <<
            "\tmax_comparisons\t"       << opts.max_comparisons <<
            "\tthreshold_change_area\t" << opts.threshold_change_area <<
            "\tthreshold_change_value\t"<< opts.threshold_change_value <<
            "\trescale_width\t"         << opts.rescale_width <<
            "\tchi_square2d\t"          << opts.chi_square2d <<
            "\tchi_square3d\t"          << opts.chi_square3d << std::endl;
  // Results
  output << "insertion_IoU\t" << insertIoU << "\ninsertion_coverage\t" << insertCov <<
            "\nremoval_IoU\t" << removeIoU << "\nremoval_coverage\t" << removeCov << std::endl;

  output << "time\t" << std::chrono::duration_cast<std::chrono::microseconds>(time).count()*1e-6 << "\ts" << std::endl;

  output.close();
  return;
}

void SaveROC(char *path, const fastcd::ChangeDetector::ChangeDetectorOptions &opts,
             double insertTP, double insertFP,
             double removeTP, double removeFP, bool area = false) {
  std::ostringstream filename;
  filename << path;
  if (path[strlen(path) - 1] != '/')
    filename << "/";
  std::string basepath = filename.str();
  filename << (area ? "roc" : "roc2") << SUFFIX << ".txt";

  std::ofstream output;
  int threshA(area ?
    opts.threshold_change_value : opts.threshold_change_area);
  int threshB(area ?
    opts.threshold_change_area : opts.threshold_change_value);
  output.open(filename.str(), std::ofstream::out | ((threshB == 255) ? std::ofstream::trunc : std::ofstream::app));
  // Parameters
  if (threshB == 255) {
    output << basepath <<
              "\ncache_size\t"            << opts.cache_size <<
              "\tmax_comparisons\t"       << opts.max_comparisons <<
              "\tthreshold_change_" << (area ? "value\t" : "area\t") << threshA <<
              "\trescale_width\t"         << opts.rescale_width <<
              "\tchi_square2d\t"          << opts.chi_square2d <<
              "\tchi_square3d\t"          << opts.chi_square3d << std::endl;
    output << "threshold_change_" << (area ? "area" : "value") << "\tinsertion_TP\tinsertion_FP\tremoval_TP\tremoval_FP" << std::endl;
  }
  // Results
  output << threshB << '\t' << insertTP << '\t' << insertFP << '\t' << removeTP << '\t' << removeFP << std::endl;
  output.close();
  return;
}

void metrics(cv::Mat input, double& fPR, double& IoU, double& tPR) {
  int in(0), un(0), ar(0);
  std::vector<cv::Mat> channels(3);
  cv::split(input, channels);
  un = cv::sum(channels[2] & cv::Scalar(1) |   channels[1] & cv::Scalar(1) )[0];
  in = cv::sum(channels[1] & cv::Scalar(1) & ~(channels[2] & cv::Scalar(1)))[0];
  ar = cv::sum(channels[1] & cv::Scalar(1))[0];
  fPR += static_cast<double>(un - ar)/static_cast<double>(input.cols*input.rows - ar); //false positive rate
  IoU += static_cast<double>(in)/static_cast<double>(un);
  tPR += static_cast<double>(in)/static_cast<double>(ar); //true positive rate
}

bool initContext() {
  glewExperimental = GL_TRUE;
  glow::inititializeGLEW();
  return true;
}

glow::GlTexture TextureFromMat(cv::Mat mat) {
  cv::Mat texture_data(mat.clone());
  cv::flip(texture_data, texture_data, 0);
  glow::GlTexture tex(mat.cols, mat.rows);
  tex.assign<uchar>(
    glow::PixelFormat::BRG,        // glow::PixelFormat::RGBA,
    glow::PixelType::UNSIGNED_BYTE,// glow::PixelType::FLOAT,
    texture_data.data);
  return tex;
}

int main(int argc, char **argv) {
  SDL_Init(SDL_INIT_VIDEO);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_Window *window =
      SDL_CreateWindow("OpenGL", 100, 100, 800, 600, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
  SDL_GLContext context = SDL_GL_CreateContext(window);

  if (argc < 2) {
    std::cout << "Usage: fastcd_example DATASET_PATH [kernel_size] [max_comparisons] [rescale_width] [threshold_change_area] [threshold_change_value]" << std::endl;
    return -1;
  }

  char basepath[MAX_PATH];
  if (argv[1][strlen(argv[1]) - 1] != '/')
    sprintf(basepath, "%s/", argv[1]);
  else
    sprintf(basepath, "%s", argv[1]);
  int kernel_size            = argc > 2 ? std::atoi(argv[2]) : 3;   // default given in example.cpp
  int max_comparisons        = argc > 3 ? std::atoi(argv[3]) : 2;   // minimum useful (3 images) // == num_img - 1
  int rescale_width          = argc > 4 ? std::atoi(argv[4]) : 500; // default given in example.cpp
  int threshold_change_area  = argc > 5 ? std::atoi(argv[5]) : 50;  // default given in example.cpp
  int threshold_change_value = argc > 6 ? std::atoi(argv[6]) : -1;  // triangle threshold

  // Initialize the OpenGL context
  initContext();

  // Load data
  Dataset data = LoadDataset(argv[1], max_comparisons+1);
  if (!data.loaded) return -1;

  // Initialize Change Detector
  fastcd::ChangeDetector::ChangeDetectorOptions cd_opts;
  cd_opts.cache_size = 10;
  cd_opts.max_comparisons = max_comparisons;
  cd_opts.threshold_change_area = threshold_change_area;
  cd_opts.threshold_change_value = threshold_change_value;
  cd_opts.rescale_width = rescale_width >= 1 ? rescale_width :
    data.images[0].GetCamera().GetWidth();
  cd_opts.chi_square2d = 3.219;
  cd_opts.chi_square3d = 4.642;
  fastcd::ChangeDetector change_detector(data.model, cd_opts);

  auto start = std::chrono::high_resolution_clock::now();

  // Add the images to the sequence. Each new image is compared with the others
  // and the inconsistencies are updated.
  for (auto &img : data.images) {
    change_detector.AddImage(img, kernel_size, true); // Insertion detection
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end-start;
  std::cout << "Adding images time:\t" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << "\tus" << std::endl;

  std::ofstream output;
  output.open("AddTime.txt", std::ofstream::out | std::ofstream::app);
  output << rescale_width << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << std::endl;
  output.close();

  char name[256];
  char path[256];
  sprintf(path, "%sout%s_", basepath, SUFFIX);

  //if (true) {
    int i;
    double scaling = static_cast<double>(cd_opts.rescale_width) /
                     static_cast<double>(data.images[0].GetCamera().GetWidth());
    for (auto &image : data.images)
      image.Scale(scaling);

    fastcd::Image image0 = data.images[0];
    Eigen::Matrix4f view0(image0.GetCamera().GetGlView());
    Eigen::Matrix4f projection0(image0.GetCamera().GetGlProjection(0.1f, 1000.0f));
    uint32_t width = static_cast<uint32_t>(image0.GetCamera().GetWidth());
    uint32_t height = static_cast<uint32_t>(image0.GetCamera().GetHeight());
    glow::GlTexture texturec0 = TextureFromMat(image0.GetRawImage());

    fastcd::DepthProjector* warp_projector;
    warp_projector = new fastcd::DepthProjector(width, height, projection0, 5);//2//3//5//
    unsigned int shadow0 = warp_projector->shadows(data.model, view0);

    auto interA = std::chrono::high_resolution_clock::now();
    for (int loop = 0; loop < 1; loop++) {
      i = -1;
      for (auto &image : data.images) {
        if (++i == 0)
          continue;
        Eigen::Matrix4f view = image.GetCamera().GetGlView();
        /*glow::GlTexture texturec(width, height);
        cv::Mat texture_data(image.GetRawImage().clone());
        cv::flip(texture_data, texture_data, 0);
        texturec.assign<uchar>(
          glow::PixelFormat::BRG,
          glow::PixelType::UNSIGNED_BYTE,
          texture_data.data);*/

        glow::GlTexture texturec(TextureFromMat(image.GetRawImage()));

        //warp_projector->render(data.model, view0, view, texturec);
        unsigned int shadow = warp_projector->shadows(data.model, view);
        glow::GlTexture inTexture(warp_projector->texture().clone());
        // mode 3
        //warp_projector->render(data.model, kernel_size,
        //  view, texturec, shadow,
        //  view0, texturec0, shadow0);
        // mode 5
        warp_projector->render(data.model, kernel_size, inTexture,
          view, texturec, shadow,//view0, //
          view0, texturec0, shadow0);//view, texturec);//
        glDeleteTextures(1, &shadow);
        //PRINT
        /*glow::GlTexture texture(warp_projector->texture());
        sprintf(name, "%swarp_%d.ppm", path, i);
        texture.save(name);*/
      }
    }
    auto interB = std::chrono::high_resolution_clock::now();
    //MEDIAN
    glow::GlTexture inTexture = warp_projector->texture().clone();
    glow::GlTexture med(warp_projector->median(inTexture));
    /*sprintf(name, "%swarp_out.ppm", path);
    med.save(name);*/
    glDeleteTextures(1, &shadow0);
    delete warp_projector;
    elapsed = interB-interA;
    std::cout << "Warping with shader time:\t" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << "\tus" << std::endl;

    fastcd::DepthProjector* warp2_projector;
    warp2_projector = new fastcd::DepthProjector(width, height, projection0);

    //fastcd::ProcessedImage image_i((*change_detector.GetImageSequence())[0]);
    warp2_projector->render(data.model, view0);
    std::vector<glow::vec4> data0(static_cast<uint32_t>(width * height));
    warp2_projector->texture().download(data0);
    fastcd::ProcessedImage image_i(image0, data0);

    interA = std::chrono::high_resolution_clock::now();
    for (int loop = 0; loop < 1; loop++) {
      i = -1;
      for (auto &image : data.images) {
        if (++i == 0)
          continue;
        Eigen::Matrix4f view = image.GetCamera().GetGlView();

        warp2_projector->render(data.model, view);
        std::vector<glow::vec4> data(static_cast<uint32_t>(width * height));
        warp2_projector->texture().download(data);
        fastcd::ProcessedImage processed_img(image, data);

        fastcd::Image warp2(processed_img.WarpFast(image_i));
        // Inconsistencies
        //cv::Mat delta;
        //image_i.CheckInconsistencies(warp2, kernel_size).convertTo(delta, CV_8UC1, 255);
        image_i.UpdateInconsistencies(warp2, kernel_size);
        //PRINT
        /*sprintf(name, "%swarp_%d.png", path, i);
        cv::imwrite(name, warp2.GetRawImage());//cv::imwrite(name, delta);//cv::imwrite(name, 255*image_i.GetInconsistencies());//*/
      }
    }
    interB = std::chrono::high_resolution_clock::now();
    /*sprintf(name, "%swarp_out.png", path);
    cv::imwrite(name, 255*image_i.GetInconsistencies());*/
    elapsed = interB-interA;
    std::cout << "Warping with cpu time:   \t" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << "\tus" << std::endl;
    /*i = 0;
    for (auto &image : data.images) {
      sprintf(name, "%s%d.jpg", path, i++);
      cv::imwrite(name, image.GetRawImage());
    }*/
  //}

  // Compute and get the changes in the environment in the form of mean position
  // and covariance of the points of the regions
#if defined(PALAZZOLO)
  std::vector<fastcd::PointCovariance3d> changes = change_detector.GetChanges();
#else
  std::vector<fastcd::PointCovariance3d> changes[2];
  change_detector.GetChanges2(changes);
#endif //PALAZZOLO

  end = std::chrono::high_resolution_clock::now();
  elapsed = end-start;
  std::cout << "Detection time:\t" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << "\tus" << std::endl;

  output.open("DecTime.txt", std::ofstream::out | std::ofstream::app);
  output << rescale_width << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << std::endl;
  output.close();

  i = 0;
  double IoU(0), tPR(0), fPR(0);
  for (auto &img : data.insertions) {
#if defined(PALAZZOLO)
    fastcd::Image img_mask(data.removals[i]);
    cv::Mat insertion(change_detector.ImageChange(img, changes, img_mask));
#else
    cv::Mat insertion(change_detector.ImageChange(img, changes[0]));
#endif //PALAZZOLO
    sprintf(name, "%sinsertion_%d.png", path, i++);
    cv::imwrite(name, insertion);
    metrics(insertion, fPR, IoU, tPR);
  }
  IoU /= static_cast<double>(i);
  tPR /= static_cast<double>(i);
  fPR /= static_cast<double>(i);
  std::cout << "Insertion IoU:\t" << IoU << "\nInsertion Coverage:\t" << tPR << std::endl;
  double IoU1(IoU), tPR1(tPR), fPR1(fPR);

  i = 0;
  IoU = tPR = fPR = 0;
  for (auto &img : data.removals) {
#if defined(PALAZZOLO)
    fastcd::Image img_mask(data.insertions[i]);
    cv::Mat removal(change_detector.ImageChange(img, changes, img_mask));
#else
    cv::Mat removal(change_detector.ImageChange(img, changes[1]));
#endif //PALAZZOLO
    sprintf(name, "%sremoval_%d.png", path, i++);
    cv::imwrite(name, removal);
    metrics(removal, fPR, IoU, tPR);
  }
  IoU /= static_cast<double>(i);
  tPR /= static_cast<double>(i);
  fPR /= static_cast<double>(i);
  std::cout << "Removals IoU:\t" << IoU << "\nRemovals Coverage:\t" << tPR << std::endl;

  SaveResults(argv[1], cd_opts, IoU1, tPR1, IoU, tPR, elapsed);
  SaveROC(argv[1], cd_opts, tPR1, fPR1, tPR, fPR);

  char ell('A');
#if defined(PALAZZOLO)
  for (auto change : changes) {
    if (ell == '[')
      ell = 'a';
    Mesh ellipsoid = change.ToMesh(100, 100, 0.0f, 0.0f, 1.0f, 0.5f);
    sprintf(name, "%sinsertion_removal_%c.obj", path, ell++);
    change.ToObj(name, 100, 100);
  }
#else
  for (auto change : changes[0]) {
    if (ell == '[')
      ell = 'a';
    Mesh ellipsoid = change.ToMesh(100, 100, 0.0f, 0.0f, 1.0f, 0.5f);
    sprintf(name, "%sinsertion_%c.obj", path, ell++);
    change.ToObj(name, 100, 100);
  }
  ell = 'A';
  for (auto change : changes[1]) {
    if (ell == '[')
      ell = 'a';
    Mesh ellipsoid = change.ToMesh(100, 100, 1.0f, 0.0f, 0.0f, 0.5f);
    sprintf(name, "%sremoval_%c.obj", path, ell++);
    change.ToObj(name, 100, 100);
  }
#endif //PALAZZOLO
  return 0;
}
