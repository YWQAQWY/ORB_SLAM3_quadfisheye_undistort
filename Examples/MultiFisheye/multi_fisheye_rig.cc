#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
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
        Sophus::SE3f Twc(R, t);
        if (key.find("Tcw") != string::npos) {
            Twc = Twc.inverse();
        }
        twc_cams[i] = Twc;
    }
    const Sophus::SE3f &Twc_front = twc_cams[0];
    for (int i = 0; i < 4; ++i) {
        tbc_cams[i] = twc_cams[i].inverse() * Twc_front;
    }
    return true;
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

    cout << "Main camera index for initialization: " << main_cam_index << endl;

    const int rig_cam_index = 0;

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

    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, true);
    SLAM.SetMainCamIndex(main_cam_index);
    //ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    Sophus::SE3f T_b_c0 = T_b_c[rig_cam_index];
    Sophus::SE3f T_c0_b = T_b_c0.inverse();

    array<Sophus::SE3f, 4> T_r_c;
    for (int cam = 0; cam < 4; ++cam) {
        T_r_c[cam] = T_c0_b * T_b_c[cam];
    }

    array<vector<Eigen::Vector3f>, 4> per_cam_points;
    vector<Eigen::Vector3f> merged_points;
    array<unordered_set<ORB_SLAM3::MapPoint*>, 4> per_cam_point_ids;
    unordered_set<ORB_SLAM3::MapPoint*> merged_point_ids;

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

    string rig_traj_path = traj_dir + "/trajectory_rig.txt";
    ofstream rig_traj(rig_traj_path.c_str());
    if (!rig_traj.is_open()) {
        cerr << "Failed to open output file: " << rig_traj_path << endl;
        return 1;
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

        Sophus::SE3f T_w_r = Tcw.inverse();
        WritePose(rig_traj, entries[ni].timestamp, T_w_r);

        for (int cam = 0; cam < 4; ++cam) {
            Sophus::SE3f T_w_ci = T_w_r * T_r_c[cam];
            WritePose(traj_files[cam], entries[ni].timestamp, T_w_ci);
        }

    }

    SLAM.Shutdown();

    const vector<ORB_SLAM3::MapPoint*> all_points = SLAM.GetAllMapPoints();
    for (int cam = 0; cam < 4; ++cam) {
        per_cam_points[cam].clear();
        per_cam_point_ids[cam].clear();
    }
    merged_points.clear();
    merged_point_ids.clear();

    for (ORB_SLAM3::MapPoint *mp : all_points) {
        if (!mp || mp->isBad()) {
            continue;
        }
        const auto observations = mp->GetObservations();
        Eigen::Vector3f pos = mp->GetWorldPos();
        bool added = false;
        for (const auto &kv : observations) {
            const vector<int> &indexes = kv.second;
            for (size_t cam_id = 0; cam_id < indexes.size() && cam_id < 4; ++cam_id) {
                if (indexes[cam_id] == -1) {
                    continue;
                }
                if (per_cam_point_ids[cam_id].insert(mp).second) {
                    per_cam_points[cam_id].push_back(pos);
                }
                added = true;
            }
        }
        if (added && merged_point_ids.insert(mp).second) {
            merged_points.push_back(pos);
        }
    }

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
