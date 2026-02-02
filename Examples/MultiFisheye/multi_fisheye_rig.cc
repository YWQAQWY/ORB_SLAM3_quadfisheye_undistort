#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <sophus/se3.hpp>

#include <System.h>
#include <Tracking.h>
#include <MapPoint.h>

using namespace std;

struct FrameEntry {
    double timestamp;
    array<string, 4> image_paths;
};

struct CameraIntrinsics {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
};

struct ORBParams {
    int nFeatures = 4000;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int iniThFAST = 12;
    int minThFAST = 7;
};

struct PoseEstimate {
    bool valid = false;
    int inliers = 0;
    Sophus::SE3f T_w_c;
};

static bool LoadAssociation(const string &association_path, vector<FrameEntry> &entries)
{
    ifstream f(association_path.c_str());
    if (!f.is_open()) {
        return false;
    }

    string line;
    while (getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        stringstream ss(line);
        FrameEntry entry;
        ss >> entry.timestamp;
        for (int i = 0; i < 4; ++i) {
            if (!(ss >> entry.image_paths[i])) {
                entry.image_paths[i].clear();
            }
        }
        if (!entry.image_paths[0].empty() && !entry.image_paths[1].empty() &&
            !entry.image_paths[2].empty() && !entry.image_paths[3].empty()) {
            entries.push_back(entry);
        }
    }
    return !entries.empty();
}

static bool ReadRigExtrinsics(const string &config_path, array<Sophus::SE3f, 4> &tbc_cams)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    array<Sophus::SE3f, 4> twc_cams;
    array<bool, 4> has_twc = {false, false, false, false};
    for (int i = 0; i < 4; ++i) {
        string key = "Camera" + to_string(i) + ".Tcw";
        cv::Mat mat;
        fs[key] >> mat;
        if (mat.empty()) {
            key = "Camera" + to_string(i) + ".Twc";
            fs[key] >> mat;
        }
        if (mat.empty() || mat.rows != 4 || mat.cols != 4) {
            return false;
        }
        cv::Mat mat32;
        mat.convertTo(mat32, CV_32F);
        Eigen::Matrix4f eigen_mat;
        cv::cv2eigen(mat32, eigen_mat);
        Eigen::Matrix3f R = eigen_mat.block<3, 3>(0, 0);
        Eigen::Vector3f t = eigen_mat.block<3, 1>(0, 3);
        twc_cams[i] = Sophus::SE3f(R, t);
        has_twc[i] = true;
    }
    for (int i = 0; i < 4; ++i) {
        tbc_cams[i] = twc_cams[i].inverse();
    }
    return true;
}

static bool ReadFloat(const cv::FileStorage &fs, const string &key, float &value)
{
    cv::FileNode node = fs[key];
    if (node.empty()) {
        return false;
    }
    value = static_cast<float>(node);
    return true;
}

static bool ReadInt(const cv::FileStorage &fs, const string &key, int &value)
{
    cv::FileNode node = fs[key];
    if (node.empty()) {
        return false;
    }
    value = static_cast<int>(node);
    return true;
}

static bool ReadIntrinsics(const string &config_path, int cam_index, CameraIntrinsics &intrinsics, float scale)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    string prefix = "Camera" + to_string(cam_index) + ".";
    bool ok = ReadFloat(fs, prefix + "fx", intrinsics.fx);
    ok = ok && ReadFloat(fs, prefix + "fy", intrinsics.fy);
    ok = ok && ReadFloat(fs, prefix + "cx", intrinsics.cx);
    ok = ok && ReadFloat(fs, prefix + "cy", intrinsics.cy);

    if (!ok) {
        return false;
    }

    intrinsics.fx *= scale;
    intrinsics.fy *= scale;
    intrinsics.cx *= scale;
    intrinsics.cy *= scale;
    return true;
}

static ORBParams ReadORBParams(const string &config_path)
{
    ORBParams params;
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return params;
    }

    ReadInt(fs, "ORBextractor.nFeatures", params.nFeatures);
    ReadFloat(fs, "ORBextractor.scaleFactor", params.scaleFactor);
    ReadInt(fs, "ORBextractor.nLevels", params.nLevels);
    ReadInt(fs, "ORBextractor.iniThFAST", params.iniThFAST);
    ReadInt(fs, "ORBextractor.minThFAST", params.minThFAST);
    return params;
}

static cv::Mat To8U(const cv::Mat &image)
{
    if (image.empty()) {
        return image;
    }
    if (image.depth() == CV_8U) {
        return image;
    }
    cv::Mat converted;
    if (image.depth() == CV_16U) {
        image.convertTo(converted, CV_8U, 1.0 / 256.0);
    } else {
        image.convertTo(converted, CV_8U, 255.0);
    }
    return converted;
}

static cv::Mat EnsureGray(const cv::Mat &image)
{
    if (image.empty()) {
        return image;
    }
    if (image.channels() == 1) {
        return To8U(image);
    }
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return To8U(gray);
}

static void WritePose(ofstream &ofs, double timestamp, const Sophus::SE3f &T_w_c)
{
    Eigen::Quaternionf q(T_w_c.rotationMatrix());
    q.normalize();
    Eigen::Vector3f t = T_w_c.translation();

    ofs << fixed << setprecision(6)
        << timestamp << " "
        << setprecision(9)
        << t.x() << " " << t.y() << " " << t.z() << " "
        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
}

static cv::Matx33f ToK(const CameraIntrinsics &intr)
{
    return cv::Matx33f(intr.fx, 0.0f, intr.cx,
                       0.0f, intr.fy, intr.cy,
                       0.0f, 0.0f, 1.0f);
}

static bool BuildMapPointDescriptors(const vector<ORB_SLAM3::MapPoint*> &map_points,
                                     vector<ORB_SLAM3::MapPoint*> &filtered_points,
                                     cv::Mat &descriptors)
{
    filtered_points.clear();
    descriptors.release();
    vector<cv::Mat> desc_rows;
    desc_rows.reserve(map_points.size());
    for (ORB_SLAM3::MapPoint *mp : map_points) {
        if (!mp || mp->isBad()) {
            continue;
        }
        cv::Mat desc = mp->GetDescriptor();
        if (desc.empty()) {
            continue;
        }
        desc_rows.push_back(desc);
        filtered_points.push_back(mp);
    }

    if (desc_rows.empty()) {
        return false;
    }

    cv::vconcat(desc_rows, descriptors);
    return true;
}

static PoseEstimate EstimatePosePnP(const vector<ORB_SLAM3::MapPoint*> &map_points,
                                    const cv::Mat &map_descriptors,
                                    const vector<cv::KeyPoint> &keypoints,
                                    const cv::Mat &descriptors,
                                    const cv::Matx33f &K)
{
    PoseEstimate estimate;
    if (map_descriptors.empty() || descriptors.empty()) {
        return estimate;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(map_descriptors, descriptors, knn_matches, 2);

    vector<cv::Point3f> object_points;
    vector<cv::Point2f> image_points;
    object_points.reserve(knn_matches.size());
    image_points.reserve(knn_matches.size());

    for (const auto &m : knn_matches) {
        if (m.size() < 2) {
            continue;
        }
        if (m[0].distance > 50.0f) {
            continue;
        }
        if (m[0].distance < 0.75f * m[1].distance) {
            int idx_mp = m[0].queryIdx;
            int idx_kp = m[0].trainIdx;
            const Eigen::Vector3f &pos = map_points[idx_mp]->GetWorldPos();
            object_points.emplace_back(pos.x(), pos.y(), pos.z());
            image_points.push_back(keypoints[idx_kp].pt);
        }
    }

    if (object_points.size() < 15) {
        return estimate;
    }

    cv::Mat rvec, tvec, inliers;
    bool ok = cv::solvePnPRansac(object_points, image_points, K, cv::noArray(),
                                 rvec, tvec, false, 100, 3.0, 0.99, inliers);
    if (!ok || inliers.empty()) {
        return estimate;
    }

    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    cv::cv2eigen(Rcv, R);
    cv::cv2eigen(tvec, t);

    Sophus::SE3f T_c_w(R, t);
    estimate.T_w_c = T_c_w.inverse();
    estimate.inliers = inliers.rows;
    estimate.valid = true;
    return estimate;
}

static Sophus::SE3f AverageRigPose(const vector<PoseEstimate> &estimates,
                                   const array<Sophus::SE3f, 4> &T_c_r,
                                   const Sophus::SE3f &fallback_T_w_r)
{
    Eigen::Vector3f t_sum = Eigen::Vector3f::Zero();
    Eigen::Vector4f q_sum = Eigen::Vector4f::Zero();
    int weight_sum = 0;
    Eigen::Quaternionf q_ref(fallback_T_w_r.rotationMatrix());

    for (size_t i = 0; i < estimates.size(); ++i) {
        if (!estimates[i].valid || estimates[i].inliers <= 0) {
            continue;
        }
        Sophus::SE3f T_w_r = estimates[i].T_w_c * T_c_r[i];
        Eigen::Quaternionf q(T_w_r.rotationMatrix());
        if (q_ref.dot(q) < 0.0f) {
            q.coeffs() *= -1.0f;
        }
        float w = static_cast<float>(estimates[i].inliers);
        t_sum += w * T_w_r.translation();
        q_sum += w * q.coeffs();
        weight_sum += estimates[i].inliers;
    }

    if (weight_sum == 0) {
        return fallback_T_w_r;
    }

    Eigen::Vector3f t = t_sum / static_cast<float>(weight_sum);
    Eigen::Quaternionf q(q_sum);
    q.normalize();
    return Sophus::SE3f(q.toRotationMatrix(), t);
}

static bool TriangulateMatches(const vector<cv::KeyPoint> &kp0,
                               const vector<cv::KeyPoint> &kpi,
                               const cv::Mat &desc0,
                               const cv::Mat &desci,
                               const cv::Matx33f &K0,
                               const cv::Matx33f &Ki,
                               const Sophus::SE3f &T_ci_c0,
                               vector<Eigen::Vector3f> &points_c0)
{
    if (desc0.empty() || desci.empty()) {
        return false;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc0, desci, knn_matches, 2);

    vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size());
    for (const auto &m : knn_matches) {
        if (m.size() < 2) {
            continue;
        }
        if (m[0].distance < 0.85f * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    if (good_matches.size() < 8) {
        return false;
    }

    sort(good_matches.begin(), good_matches.end(),
         [](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });
    if (good_matches.size() > 1500) {
        good_matches.resize(1500);
    }

    vector<cv::Point2f> pts0;
    vector<cv::Point2f> ptsi;
    pts0.reserve(good_matches.size());
    ptsi.reserve(good_matches.size());
    for (const auto &m : good_matches) {
        pts0.push_back(kp0[m.queryIdx].pt);
        ptsi.push_back(kpi[m.trainIdx].pt);
    }

    Eigen::Matrix3f R_ci_c0 = T_ci_c0.rotationMatrix();
    Eigen::Vector3f t_ci_c0 = T_ci_c0.translation();

    cv::Matx34f P0(
        K0(0, 0), K0(0, 1), K0(0, 2), 0.0f,
        K0(1, 0), K0(1, 1), K0(1, 2), 0.0f,
        K0(2, 0), K0(2, 1), K0(2, 2), 0.0f);

    cv::Matx34f Pi(
        Ki(0, 0) * R_ci_c0(0, 0) + Ki(0, 1) * R_ci_c0(1, 0) + Ki(0, 2) * R_ci_c0(2, 0),
        Ki(0, 0) * R_ci_c0(0, 1) + Ki(0, 1) * R_ci_c0(1, 1) + Ki(0, 2) * R_ci_c0(2, 1),
        Ki(0, 0) * R_ci_c0(0, 2) + Ki(0, 1) * R_ci_c0(1, 2) + Ki(0, 2) * R_ci_c0(2, 2),
        Ki(0, 0) * t_ci_c0(0) + Ki(0, 1) * t_ci_c0(1) + Ki(0, 2) * t_ci_c0(2),
        Ki(1, 0) * R_ci_c0(0, 0) + Ki(1, 1) * R_ci_c0(1, 0) + Ki(1, 2) * R_ci_c0(2, 0),
        Ki(1, 0) * R_ci_c0(0, 1) + Ki(1, 1) * R_ci_c0(1, 1) + Ki(1, 2) * R_ci_c0(2, 1),
        Ki(1, 0) * R_ci_c0(0, 2) + Ki(1, 1) * R_ci_c0(1, 2) + Ki(1, 2) * R_ci_c0(2, 2),
        Ki(1, 0) * t_ci_c0(0) + Ki(1, 1) * t_ci_c0(1) + Ki(1, 2) * t_ci_c0(2),
        Ki(2, 0) * R_ci_c0(0, 0) + Ki(2, 1) * R_ci_c0(1, 0) + Ki(2, 2) * R_ci_c0(2, 0),
        Ki(2, 0) * R_ci_c0(0, 1) + Ki(2, 1) * R_ci_c0(1, 1) + Ki(2, 2) * R_ci_c0(2, 1),
        Ki(2, 0) * R_ci_c0(0, 2) + Ki(2, 1) * R_ci_c0(1, 2) + Ki(2, 2) * R_ci_c0(2, 2),
        Ki(2, 0) * t_ci_c0(0) + Ki(2, 1) * t_ci_c0(1) + Ki(2, 2) * t_ci_c0(2));

    cv::Mat points4d;
    cv::triangulatePoints(P0, Pi, pts0, ptsi, points4d);

    points_c0.clear();
    points_c0.reserve(points4d.cols);
    for (int i = 0; i < points4d.cols; ++i) {
        float w = points4d.at<float>(3, i);
        if (w == 0.0f) {
            continue;
        }
        Eigen::Vector3f X_c0(points4d.at<float>(0, i) / w,
                             points4d.at<float>(1, i) / w,
                             points4d.at<float>(2, i) / w);
        if (X_c0.z() <= 0.0f) {
            continue;
        }
        Eigen::Vector3f X_ci = R_ci_c0 * X_c0 + t_ci_c0;
        if (X_ci.z() <= 0.0f) {
            continue;
        }

        Eigen::Vector2f reproj0(
            K0(0, 0) * X_c0.x() / X_c0.z() + K0(0, 2),
            K0(1, 1) * X_c0.y() / X_c0.z() + K0(1, 2));
        Eigen::Vector2f reproji(
            Ki(0, 0) * X_ci.x() / X_ci.z() + Ki(0, 2),
            Ki(1, 1) * X_ci.y() / X_ci.z() + Ki(1, 2));

        Eigen::Vector2f obs0(pts0[i].x, pts0[i].y);
        Eigen::Vector2f obsi(ptsi[i].x, ptsi[i].y);
        if ((reproj0 - obs0).norm() > 5.0f || (reproji - obsi).norm() > 5.0f) {
            continue;
        }

        points_c0.push_back(X_c0);
    }

    return !points_c0.empty();
}

static void SavePoints(const string &path, const vector<Eigen::Vector3f> &points)
{
    ofstream ofs(path.c_str());
    for (const auto &p : points) {
        ofs << fixed << setprecision(6)
            << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        cerr << endl
             << "Usage: ./multi_fisheye_rig path_to_vocabulary "
             << "path_to_settings path_to_extrinsics "
             << "path_to_association main_cam_index output_dir" << endl;
        return 1;
    }

    string vocab_path = argv[1];
    string settings_path = argv[2];
    string extrinsics_path = argv[3];
    string association_path = argv[4];
    int main_cam_index = stoi(argv[5]);
    string output_dir = argv[6];

    if (main_cam_index < 0 || main_cam_index > 3) {
        cerr << "main_cam_index must be 0..3" << endl;
        return 1;
    }

    vector<FrameEntry> entries;
    if (!LoadAssociation(association_path, entries)) {
        cerr << "Failed to load association file: " << association_path << endl;
        return 1;
    }

    array<Sophus::SE3f, 4> T_b_c;
    if (!ReadRigExtrinsics(extrinsics_path,T_b_c)) {
        cerr << "Failed to load extrinsics file: " << extrinsics_path << endl;
        return 1;
    }

    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, false);
    //ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, ture);
    float imageScale = SLAM.GetImageScale();

    array<CameraIntrinsics, 4> intrinsics;
    array<cv::Matx33f, 4> K;
    array<cv::Ptr<cv::ORB>, 4> orb_extractors;
    for (int cam = 0; cam < 4; ++cam) {
        if (!ReadIntrinsics(settings_path, cam, intrinsics[cam], imageScale)) {
            cerr << "Failed to read intrinsics from: " << settings_path << endl;
            return 1;
        }
        K[cam] = ToK(intrinsics[cam]);
        ORBParams orb = ReadORBParams(settings_path);
        orb_extractors[cam] = cv::ORB::create(
            orb.nFeatures, orb.scaleFactor, orb.nLevels, 31, 0, 2,
            cv::ORB::HARRIS_SCORE, 31, orb.iniThFAST);
    }

    Sophus::SE3f T_b_c0 = T_b_c[main_cam_index];
    Sophus::SE3f T_c0_b = T_b_c0.inverse();

    array<Sophus::SE3f, 4> T_ci_c0;
    array<Sophus::SE3f, 4> T_c_b;
    for (int cam = 0; cam < 4; ++cam) {
        T_ci_c0[cam] = T_b_c[cam].inverse() * T_b_c0;
        T_c_b[cam] = T_b_c[cam].inverse();
    }

    array<vector<Eigen::Vector3f>, 4> per_cam_points;
    vector<Eigen::Vector3f> merged_points;

    string traj_dir = output_dir;
    array<ofstream, 4> traj_files;
    for (int cam = 0; cam < 4; ++cam) {
        string traj_path = traj_dir + "/trajectory_cam" + to_string(cam) + ".txt";
        traj_files[cam].open(traj_path.c_str());
        if (!traj_files[cam].is_open()) {
            cerr << "Failed to open output file: " << traj_path << endl;
            return 1;
        }
    }

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Frames in the sequence: " << entries.size() << endl << endl;

    for (size_t ni = 0; ni < entries.size(); ++ni) {
        array<cv::Mat, 4> images;
        array<cv::Mat, 4> images_8u;
        for (int cam = 0; cam < 4; ++cam) {
            images[cam] = cv::imread(entries[ni].image_paths[cam], cv::IMREAD_UNCHANGED);
            if (images[cam].empty()) {
                cerr << "Failed to load image at: " << entries[ni].image_paths[cam] << endl;
                return 1;
            }
            if (imageScale != 1.f) {
                int width = static_cast<int>(images[cam].cols * imageScale);
                int height = static_cast<int>(images[cam].rows * imageScale);
                cv::resize(images[cam], images[cam], cv::Size(width, height));
            }
            images_8u[cam] = To8U(images[cam]);
            if (images_8u[cam].empty() || images_8u[cam].depth() != CV_8U) {
                cerr << "Failed to convert image to CV_8U at: " << entries[ni].image_paths[cam] << endl;
                return 1;
            }
        }

        vector<cv::Mat> gray_images(4);
        for (int cam = 0; cam < 4; ++cam) {
            gray_images[cam] = EnsureGray(images_8u[cam]).clone();
        }

        Sophus::SE3f Tcw = SLAM.TrackMulti(gray_images, entries[ni].timestamp);
        if (SLAM.GetTrackingState() != ORB_SLAM3::Tracking::OK) {
            continue;
        }

        Sophus::SE3f T_w_b_raw = Tcw.inverse();
        Sophus::SE3f T_w_c0_raw = T_w_b_raw * T_c0_b;

        vector<ORB_SLAM3::MapPoint*> map_points_all = SLAM.GetTrackedMapPoints();
        vector<ORB_SLAM3::MapPoint*> map_points;
        cv::Mat map_descriptors;
        if (!BuildMapPointDescriptors(map_points_all, map_points, map_descriptors)) {
            continue;
        }

        array<vector<cv::KeyPoint>, 4> keypoints;
        array<cv::Mat, 4> descriptors;
        for (int cam = 0; cam < 4; ++cam) {
            cv::Mat cam_gray = gray_images[cam];
            orb_extractors[cam]->detectAndCompute(cam_gray, cv::noArray(), keypoints[cam], descriptors[cam]);
        }

        vector<PoseEstimate> estimates(4);
        estimates[main_cam_index].valid = true;
        estimates[main_cam_index].inliers = 50;
        estimates[main_cam_index].T_w_c = T_w_c0_raw;

        for (int cam = 0; cam < 4; ++cam) {
            if (cam == main_cam_index) {
                continue;
            }
            estimates[cam] = EstimatePosePnP(map_points, map_descriptors, keypoints[cam], descriptors[cam], K[cam]);
        }

        Sophus::SE3f T_w_b = AverageRigPose(estimates, T_c_b, T_w_b_raw);
        Sophus::SE3f T_w_c0 = T_w_b * T_b_c[main_cam_index];

        for (int cam = 0; cam < 4; ++cam) {
            Sophus::SE3f T_w_ci = T_w_b * T_b_c[cam];
            WritePose(traj_files[cam], entries[ni].timestamp, T_w_ci);
        }

        for (int cam = 0; cam < 4; ++cam) {
            if (cam == main_cam_index) {
                continue;
            }
            vector<Eigen::Vector3f> points_c0;
            if (!TriangulateMatches(keypoints[main_cam_index], keypoints[cam],
                                    descriptors[main_cam_index], descriptors[cam],
                                    K[main_cam_index], K[cam], T_ci_c0[cam], points_c0)) {
                continue;
            }

            per_cam_points[cam].reserve(per_cam_points[cam].size() + points_c0.size());
            merged_points.reserve(merged_points.size() + points_c0.size());
            for (const auto &p_c0 : points_c0) {
                Eigen::Vector3f p_w = T_w_c0 * p_c0;
                per_cam_points[cam].push_back(p_w);
                merged_points.push_back(p_w);
            }
        }
    }

    SLAM.Shutdown();

    for (int cam = 0; cam < 4; ++cam) {
        string map_path = output_dir + "/map_rig_cam" + to_string(cam) + ".xyz";
        SavePoints(map_path, per_cam_points[cam]);
        cout << "Saved " << map_path << " with " << per_cam_points[cam].size() << " points" << endl;
    }

    string merged_path = output_dir + "/map_rig_fused.xyz";
    SavePoints(merged_path, merged_points);
    cout << "Saved " << merged_path << " with " << merged_points.size() << " points" << endl;

    return 0;
}
