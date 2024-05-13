// -*-c++-*----------------------------------------------------------------------------------------
// Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "frequency_cam/frequency_cam.h"

#include <math.h>

#include <fstream>
#include <iomanip>
#include <iostream>

template <typename T, typename A> int arg_max(std::vector<T, A> const &vec) {
  return static_cast<int>(
      std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

template <typename T, typename A> int arg_min(std::vector<T, A> const &vec) {
  return static_cast<int>(
      std::distance(vec.begin(), min_element(vec.begin(), vec.end())));
}

namespace frequency_cam
{

FrequencyCam::~FrequencyCam() {
  mean_position_csv_file_.close();
  hough_circle_position_csv_file_.close();
  blob_detection_position_csv_file_.close();
  delete[] state_;

  std::cout << "Number of external triggers: " << nrExtTriggers_ << std::endl;
  std::cout << "Number of runs: " << nrRuns_ << std::endl;
  std::cout << "Number of time synchronization matches: " << nrSyncMatches_ << std::endl;
  std::cout << "Number of detected wands by the mean approach: " << nrMeanDetectedWands_ << std::endl;
  std::cout << "Number of detected wands by the hough approach: " << nrHoughDetectedWands_ << std::endl;
  std::cout << "Number of detected wands by the blog approach: " << nrBlobDetectedWands_ << std::endl;
}

static void compute_alpha_beta(const double T_cut, double * alpha, double * beta)
{
  // compute the filter coefficients alpha and beta (see paper)
  const double omega_cut = 2 * M_PI / T_cut;
  const double phi = 2 - std::cos(omega_cut);
  *alpha = (1.0 - std::sin(omega_cut)) / std::cos(omega_cut);
  *beta = phi - std::sqrt(phi * phi - 1.0);  // see paper
}

bool FrequencyCam::initialize(
  double minFreq, double maxFreq, double cutoffPeriod, int timeoutCycles, uint16_t debugX,
  uint16_t debugY, int visualization_choice, bool debug_frames)
{
#ifdef DEBUG  // the debug flag must be set in the header file
  debug_.open("freq.txt", std::ofstream::out);
#endif

  freq_[0] = std::max(minFreq, 0.1);
  freq_[1] = maxFreq;
  dtMax_ = 1.0 / freq_[0];
  dtMaxHalf_ = 0.5 * dtMax_;
  dtMin_ = 1.0 / (freq_[1] >= freq_[0] ? freq_[1] : 1.0);
  dtMinHalf_ = 0.5 * dtMin_;
  timeoutCycles_ = timeoutCycles;
  const double T_prefilter = cutoffPeriod;
  double alpha_prefilter, beta_prefilter;
  compute_alpha_beta(T_prefilter, &alpha_prefilter, &beta_prefilter);

  // compute IIR filter coefficient from alpha and beta (see paper)
  c_[0] = alpha_prefilter + beta_prefilter;
  c_[1] = -alpha_prefilter * beta_prefilter;
  c_p_ = 0.5 * (1 + beta_prefilter);

  debugX_ = debugX;
  debugY_ = debugY;

  visualizationChoice_ = visualization_choice;

  debugFrames_ = debug_frames;

  return (true);
}

void FrequencyCam::initializeState(uint32_t width, uint32_t height, uint64_t t_full, uint64_t t_off)
{
  const uint32_t t = shorten_time(t_full) - 1;
#ifdef DEBUG
  timeOffset_ = (t_off / 1000) - shorten_time(t_off);  // safe high bits lost by shortening
#else
  (void)t_off;
#endif

  for (std::size_t i = 0; i < width; ++i) {
    x_updates_.emplace_back(false);
  }
  std::cerr << "x_updates_.size(): " << x_updates_.size() << std::endl;
  for (std::size_t i = 0; i < height; ++i) {
    y_updates_.emplace_back(false);
  }
  std::cerr << "y_updates_.size(): " << y_updates_.size() << std::endl;

  width_ = width;
  height_ = height;
  state_ = new State[width * height];
  for (size_t i = 0; i < width * height; i++) {
    State & s = state_[i];
    s.t_flip_up_down = t;
    s.t_flip_down_up = t;
    s.L_km1 = 0;
    s.L_km2 = 0;
    s.period = -1;
    s.set_time_and_polarity(t, 0);
  }
}

cv::Mat FrequencyCam::makeFrequencyAndEventImage(
  cv::Mat * evImg, bool overlayEvents, bool useLogFrequency, float dt, uint64_t trigger_timestamp)
{
  if (overlayEvents) {
    *evImg = cv::Mat::zeros(height_, width_, CV_8UC1);
  }
  if (useLogFrequency) {
    return (
      overlayEvents ? makeTransformedFrequencyImage<LogTF, EventFrameUpdater>(evImg, dt, trigger_timestamp)
                    : makeTransformedFrequencyImage<LogTF, NoEventFrameUpdater>(evImg, dt, trigger_timestamp));
  }
  return (
    overlayEvents ? makeTransformedFrequencyImage<NoTF, EventFrameUpdater>(evImg, dt, trigger_timestamp)
                  : makeTransformedFrequencyImage<NoTF, NoEventFrameUpdater>(evImg, dt, trigger_timestamp));
}

void FrequencyCam::detectHouchCircles(cv::Mat& raw_img, const cv::Mat& gray, const uint64_t trigger_timestamp) {
  auto debug_position_color = cv::Scalar(100, 100, 100);

  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1/*dp*/, 20/*minDist*/, 10/*param1*/, 8/*param2*/, 0, 10);

  if (3 == circles.size()) {
    std::vector<Point> circles_points;
    for (const auto& circle: circles) {
      circles_points.emplace_back(circle[0], circle[1]);
    }
    int idx_min;
    int idx_max;
    double dist_0_1;
    double dist_1_2;
    double dist_0_2;
    sort3Kp(circles_points, idx_min, idx_max, dist_0_1, dist_1_2, dist_0_2);

    hough_circle_position_csv_file_ << trigger_timestamp;
    for (const auto& circle : circles_points) {
      hough_circle_position_csv_file_ << ";" << circle.x << ";" << circle.y;

      if (1 == visualizationChoice_) {
        cv::Point center(cvRound(circle.x), cvRound(circle.y));
        // int radius = cvRound(circle[2]);
        // draw the circle center
        // cv::circle(raw_img, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0 );
        cv::circle(raw_img, center, 3, debug_position_color, 1, 8, 0);
        // draw the circle outline
        // cv::circle(raw_img, center, radius, cv::Scalar(800, 800, 800), 1, 8, 0);
      }
    }
    if (1 == visualizationChoice_) {
      cv::putText(raw_img, "Nr. of markers: " + std::to_string(circles.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
    }
    hough_circle_position_csv_file_ << "\n";
    nrHoughDetectedWands_++;
  } else {
    // std::cout << "trigger_timestamp: " << trigger_timestamp << std::endl;
    hough_circle_position_csv_file_ << trigger_timestamp;
    hough_circle_position_csv_file_ << ";" << -1 << ";" << -1  << ";" << -1 << ";" << -1 << ";" << -1 << ";" << -1 << "\n";

    if (1 == visualizationChoice_) {
      cv::putText(raw_img, "Nr. of markers: " + std::to_string(circles.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
    }
  }
}

void FrequencyCam::detectBlobs(cv::Mat& raw_img, const cv::Mat& gray, const uint64_t trigger_timestamp) {
  auto debug_position_color = cv::Scalar(100, 100, 100);

  std::vector<cv::KeyPoint> keypoints;
  blob_detector_->detect(gray, keypoints);

  if (3 == keypoints.size()) {
    std::vector<Point> circles_points;
    for (const auto& keypoint: keypoints) {
      circles_points.emplace_back(keypoint.pt.x, keypoint.pt.y);
    }
    int idx_min;
    int idx_max;
    double dist_0_1;
    double dist_1_2;
    double dist_0_2;
    sort3Kp(circles_points, idx_min, idx_max, dist_0_1, dist_1_2, dist_0_2);

    blob_detection_position_csv_file_ << trigger_timestamp;
    // cv::drawKeypoints(raw_img, keypoints, raw_img, cv::Scalar(800, 800, 800), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    for (const auto& circle : circles_points) {
      blob_detection_position_csv_file_ << ";" << circle.x << ";" << circle.y;
      if (2 == visualizationChoice_) {
        cv::circle(raw_img, {cvRound(circle.x), cvRound(circle.y)}, 3, debug_position_color, 1);
      }
    }

    if (2 == visualizationChoice_) {
      cv::putText(raw_img, "Nr. of markers: " + std::to_string(keypoints.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
    }
    blob_detection_position_csv_file_ << "\n";
    nrBlobDetectedWands_++;
  } else {
    // std::cout << "trigger_timestamp: " << trigger_timestamp << std::endl;
    blob_detection_position_csv_file_ << trigger_timestamp;
    blob_detection_position_csv_file_ << ";" << -1 << ";" << -1  << ";" << -1 << ";" << -1 << ";" << -1 << ";" << -1 << "\n";

    if (2 == visualizationChoice_) {
      cv::putText(raw_img, "Nr. of markers: " + std::to_string(keypoints.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
    }
  }
}

void FrequencyCam::getMeanPosition(cv::Mat& raw_img, const std::vector<Point>& frequency_points, const uint64_t trigger_timestamp) {
    std::vector<Point> filtered_frequency_points;
    std::vector<std::size_t> number_of_points;
    std::vector<std::size_t> assigned_indices;
    // Going through all the points which are in the frequency range
    // for the reference point
    for (std::size_t i = 0; i < frequency_points.size(); ++i) {
      // Skip if the point was already assigned to a cluster
      if (std::count(assigned_indices.begin(), assigned_indices.end(), i)) {
        continue;
      }

      std::vector<std::size_t> candidate_indices;
      std::vector<double> x_values;
      std::vector<double> y_values;
      auto x = frequency_points.at(i).x;
      auto y = frequency_points.at(i).y;

      std::size_t counts = 0;
      // Going through all the remaining points which are in the frequency range
      // for the candidate points
      for (std::size_t j = i + 1; j < frequency_points.size(); ++j) {
        // Skip if the point was already assigned to a cluster
        if (std::count(assigned_indices.begin(), assigned_indices.end(), j)) {
          continue;
        }

        auto x_candidate = frequency_points.at(j).x;
        auto y_candidate = frequency_points.at(j).y;

        // Make sure that points in the same cluster are close
        // double distance = 10;
        double distance = 8;
        if ((std::fabs(x - x_candidate) < distance) && (std::fabs(y - y_candidate) < distance)) {
          candidate_indices.emplace_back(j);
          x_values.emplace_back(x_candidate);
          y_values.emplace_back(y_candidate);
          counts++;
        }
      }

      // We want a certain amount of points for a cluster
      if (10 < counts) {
        candidate_indices.emplace_back(i);
        x_values.emplace_back(x);
        y_values.emplace_back(y);

        // Calculate mean position of the cluster
        auto mean_x = std::reduce(x_values.begin(), x_values.end());
        mean_x /= x_values.size();
        auto mean_y = std::reduce(y_values.begin(), y_values.end());
        mean_y /= y_values.size();

        // Centroid via Moments
        // double m00 = 0.0;
        // double m10 = 0.0;
        // double m01 = 0.0;
        // for (std::size_t x_i = 0; x_i < x_values.size(); ++x_i) {
        //   for (std::size_t y_i = 0; y_i < y_values.size(); ++y_i) {
        //      m00 += 1.0; 
        //      m10 += x_values.at(x_i); 
        //      m01 += y_values.at(y_i); 
        //   }
        // }
        // std::cout << "Mean x: " << mean_x << ", y: " << mean_y << std::endl;
        // mean_x = m10 / m00;
        // mean_y = m01 / m00;
        // std::cout << "Centroid x: " << mean_x << ", y: " << mean_y << std::endl;

        bool insert = true;
        for (const auto & point : filtered_frequency_points) {
          x = point.x;
          y = point.y;

          // Do not add cluster if its mean position is close to an already added
          // cluster
          // double cluster_distance = 20;
          double cluster_distance = 15;
          if ((std::fabs(x - mean_x) < cluster_distance) && (std::fabs(y - mean_y) < cluster_distance)) {
            insert = false;
            break;
          }
        }
        if (insert) {
          // filtered_frequency_points.emplace_back(mean_x, mean_y, frequency_point.first);
          filtered_frequency_points.emplace_back(mean_x, mean_y);
          number_of_points.emplace_back(x_values.size());
        }

        assigned_indices.insert(
          assigned_indices.end(), candidate_indices.begin(), candidate_indices.end());
      }
    }

    // Only proceed if we detected three clusters (three markers)
    // if (!filtered_frequency_points.empty()) {
    if (filtered_frequency_points.size() == 3) {
      int idx_min;
      int idx_max;
      double dist_0_1;
      double dist_1_2;
      double dist_0_2;
      sort3Kp(filtered_frequency_points, idx_min, idx_max, dist_0_1, dist_1_2, dist_0_2);

      // Check if points are on a line (are collinear)
      // double residual = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
      // double term_1 = std::get<0>(filtered_frequency_points.at(0)) * (std::get<1>(filtered_frequency_points.at(1)) - std::get<1>(filtered_frequency_points.at(2)));
      // double term_2 = std::get<0>(filtered_frequency_points.at(1)) * (std::get<1>(filtered_frequency_points.at(2)) - std::get<1>(filtered_frequency_points.at(0)));
      // double term_3 = std::get<0>(filtered_frequency_points.at(2)) * (std::get<1>(filtered_frequency_points.at(0)) - std::get<1>(filtered_frequency_points.at(1)));
      // double residual = std::get<0>(filtered_frequency_points.at(0)) * (std::get<1>(filtered_frequency_points.at(1)) - std::get<1>(filtered_frequency_points.at(2)))
      //                   + std::get<0>(filtered_frequency_points.at(1)) * (std::get<1>(filtered_frequency_points.at(2)) - std::get<1>(filtered_frequency_points.at(0)))
      //                   + std::get<0>(filtered_frequency_points.at(2)) * (std::get<1>(filtered_frequency_points.at(0)) - std::get<1>(filtered_frequency_points.at(1)));
      // std::cout << "x1: " << std::get<0>(filtered_frequency_points.at(0)) << std::endl;
      // std::cout << "y1 : " << std::get<1>(filtered_frequency_points.at(0)) << std::endl;
      // std::cout << "x2: " << std::get<0>(filtered_frequency_points.at(1)) << std::endl;
      // std::cout << "y2 : " << std::get<1>(filtered_frequency_points.at(1)) << std::endl;
      // std::cout << "x3: " << std::get<0>(filtered_frequency_points.at(2)) << std::endl;
      // std::cout << "y3 : " << std::get<1>(filtered_frequency_points.at(2)) << std::endl;
      // std::cout << "term 1: " << term_1 << ", term 2: " << term_2 << ", term_3: " << term_3 << std::endl;
      // std::cout << "residual: " << std::fabs(residual) << std::endl;


      // Line constraint from https://stackoverflow.com/questions/28619791/how-do-i-check-to-see-if-three-points-form-a-straight-line
      // Eigen::Matrix3f matrix;
      // matrix << std::get<0>(filtered_frequency_points.at(0)), std::get<1>(filtered_frequency_points.at(0)), 1.0,
      //           std::get<0>(filtered_frequency_points.at(1)), std::get<1>(filtered_frequency_points.at(1)), 1.0,
      //           std::get<0>(filtered_frequency_points.at(2)), std::get<1>(filtered_frequency_points.at(2)), 1.0;
      // Eigen::FullPivLU<Eigen::Matrix3f> lu_decomp(matrix);
      // auto rank = lu_decomp.rank();
      // std::cout << "Rank: " << rank << std::endl;
      // // result = rank([x2-x1, y2-y1; x3-x1, y3-y1]) < 2;

      // std::cout << "Filtered points:" << std::endl;
      // std::cout << "time stamp: " << lastEventTime_ << std::endl;
      //for (const auto & filtered_point : filtered_points) {

      // Write to csv file
      mean_position_csv_file_ << trigger_timestamp;
      for (std::size_t i = 0; i < filtered_frequency_points.size(); ++i) {
        // std::cout << "x: " << std::get<0>(filtered_frequency_points.at(i))
        //           << ", y: " << std::get<1>(filtered_frequency_points.at(i))
        //           << ", frequency: " << std::get<2>(filtered_frequency_points.at(i))
        //           << ", number of points: " << number_of_points.at(i) << std::endl;
        // auto frequency = std::get<0>(filtered_frequency_points.at(i));
        mean_position_csv_file_ << ";" << filtered_frequency_points.at(i).x << ";"
                  << filtered_frequency_points.at(i).y;

        // Visualize detected markers with circles
        if (3 == visualizationChoice_) {
          double color_level = 0;
          if (i == 0) {
            color_level = 1000;
          } else if (i == 1) {
            color_level = 800;
          } else if (i == 2) {
            color_level = 600;
          }

          cv::circle(
            raw_img,
            {static_cast<int>(filtered_frequency_points.at(i).x),
             static_cast<int>(filtered_frequency_points.at(i).y)},
            // 2, CV_RGB(550, 550, 550), 4);
            12, CV_RGB(color_level, color_level, color_level), 6);
             //    cv::putText(raw_img, std::to_string(i),
             //                {static_cast<int>(std::get<0>(filtered_frequency_points.at(i))),
             //                 static_cast<int>(std::get<1>(filtered_frequency_points.at(i)))},
             //                cv::FONT_HERSHEY_SIMPLEX,
             //                2,
             //                550,
             //                4);
        }
      }
      mean_position_csv_file_ << "\n";

      if (3 == visualizationChoice_) {
        cv::putText(raw_img, "Nr. of markers: " + std::to_string(filtered_frequency_points.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      }

      // Add debug information to frame
      /*
      cv::putText(raw_img, "idx_min: " + std::to_string(idx_min), {50, 420}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(raw_img, "idx_max: " + std::to_string(idx_max), {50, 460}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(raw_img, "dist_0_1: " + std::to_string(dist_0_1), {50, 300}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(raw_img, "dist_1_2: " + std::to_string(dist_1_2), {50, 340}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(raw_img, "dist_0_2: " + std::to_string(dist_0_2), {50, 380}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);

      auto p1x = filtered_frequency_points.at(0).x;
      auto p1y = filtered_frequency_points.at(0).y;
      cv::putText(raw_img, "p0: x: " + std::to_string(p1x), {1000, 300}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(raw_img, "p0: y: " + std::to_string(p1y), {1000, 340}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      auto p2x = filtered_frequency_points.at(1).x;
      auto p2y = filtered_frequency_points.at(1).y;
      cv::putText(raw_img, "p1: x: " + std::to_string(p2x), {1000, 380}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(raw_img, "p1: y: " + std::to_string(p2y), {1000, 420}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      auto p3x = filtered_frequency_points.at(2).x;
      auto p3y = filtered_frequency_points.at(2).y;
      cv::putText(raw_img, "p2: x: " + std::to_string(p3x), {1000, 460}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(raw_img, "p2: y: " + std::to_string(p3y), {1000, 500}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      */

      nrMeanDetectedWands_++;
    } else {
      // std::cout << "trigger_timestamp: " << trigger_timestamp << std::endl;
      mean_position_csv_file_ << trigger_timestamp;
      mean_position_csv_file_ << ";" << -1 << ";" << -1  << ";" << -1 << ";" << -1 << ";" << -1 << ";" << -1 << "\n";

      if (3 == visualizationChoice_) {
        cv::putText(raw_img, "Nr. of markers: " + std::to_string(filtered_frequency_points.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      }
    }
  }

void FrequencyCam::getStatistics(size_t * numEvents) const { *numEvents = eventCount_; }

void FrequencyCam::resetStatistics() { eventCount_ = 0; }

// void FrequencyCam::sort3Kp(vector<cv::KeyPoint> &kp) {
void FrequencyCam::sort3Kp(std::vector<Point>& kp, int& idx_min, int& idx_max, double& dist_0_1, double& dist_1_2, double& dist_0_2) {
  // Sorts from closest together to most seperated
  // vector<cv::KeyPoint> cp_kp;
  std::vector<Point> cp_kp;
  cp_kp = kp;
  std::vector<double> d;
  for (std::size_t i = 0; i < 2; i++) {
    for (std::size_t j = i + 1; j < 3; j++) {
      // cv::Point2f diff = kp.at(i).pt - kp.at(j).pt;
      cv::Point2d diff {static_cast<double>(kp.at(i).x - kp.at(j).x),
                        static_cast<double>(kp.at(i).y - kp.at(j).y)};
      double dist = sqrt(diff.x * diff.x + diff.y * diff.y);
      d.push_back(dist);
    }
  }

  // int idx_min = arg_min(d);
  // int idx_max = arg_max(d);
  idx_min = arg_min(d);
  idx_max = arg_max(d);
  switch (idx_max) {
  case 0:
    kp.at(1) = cp_kp.at(2);
    if (idx_min == 1) {
      kp.at(0) = cp_kp.at(0);
      kp.at(2) = cp_kp.at(1);
    } else {
      kp.at(0) = cp_kp.at(1);
      kp.at(2) = cp_kp.at(0);
    }
    break;
  case 1:
    kp.at(1) = cp_kp.at(1);
    if (idx_min == 0) {
      kp.at(0) = cp_kp.at(0);
      kp.at(2) = cp_kp.at(2);
    } else {
      kp.at(0) = cp_kp.at(2);
      kp.at(2) = cp_kp.at(0);
    }
    break;
  case 2:
    kp.at(1) = cp_kp.at(0);
    if (idx_min == 0) {
      kp.at(0) = cp_kp.at(1);
      kp.at(2) = cp_kp.at(2);
    } else {
      kp.at(0) = cp_kp.at(2);
      kp.at(2) = cp_kp.at(1);
    }
    break;
  }

  cv::Point2d diff_0_1 {static_cast<double>(kp.at(0).x - kp.at(1).x),
                        static_cast<double>(kp.at(0).y - kp.at(1).y)};
  dist_0_1 = sqrt(diff_0_1.x * diff_0_1.x + diff_0_1.y * diff_0_1.y);
  cv::Point2d diff_1_2 {static_cast<double>(kp.at(1).x - kp.at(2).x),
                        static_cast<double>(kp.at(1).y - kp.at(2).y)};
  dist_1_2 = sqrt(diff_1_2.x * diff_1_2.x + diff_1_2.y * diff_1_2.y);
  cv::Point2d diff_0_2 {static_cast<double>(kp.at(0).x - kp.at(2).x),
                        static_cast<double>(kp.at(0).y - kp.at(2).y)};
  dist_0_2 = sqrt(diff_0_2.x * diff_0_2.x + diff_0_2.y * diff_0_2.y);

  if (dist_0_1 >= dist_1_2) {
    std::cout << "diff_0_1 >= diff_1_2" << std::endl;
  }
  if (dist_1_2 >= dist_0_2) {
    std::cout << "diff_1_2 >= diff_0_2" << std::endl;
  }
}

std::ostream & operator<<(std::ostream & os, const FrequencyCam::Event & e)
{
  os << std::fixed << std::setw(10) << std::setprecision(6) << e.t * 1e-6 << " "
     << static_cast<int>(e.polarity) << " " << e.x << " " << e.y;
  return (os);
}

}  // namespace frequency_cam
