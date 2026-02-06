#include <algorithm>
#include <array>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
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

static string ToLower(string value)
{
    transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(tolower(c));
    });
    return value;
}

static int NormalizeCamIndex(int camIndex, int nCam)
{
    if(camIndex < 0)
        return 0;
    if(nCam > 0 && camIndex >= nCam)
        return 0;
    return camIndex;
}

static int ParseRigCamIndex(const cv::FileStorage &fs, const string &key, int nCam, int defaultIndex, bool *wasSet)
{
    if(wasSet)
        *wasSet = false;
    cv::FileNode node = fs[key];
    if(node.empty())
        return defaultIndex;
    if(wasSet)
        *wasSet = true;
    if(node.isInt())
        return NormalizeCamIndex(node.operator int(), nCam);
    if(node.isString())
    {
        string value = ToLower(node.string());
        if(value == "front")
            return NormalizeCamIndex(1, nCam);
        if(value == "left")
            return NormalizeCamIndex(0, nCam);
        if(value == "right")
            return NormalizeCamIndex(2, nCam);
        if(value == "rear")
            return NormalizeCamIndex(3, nCam);
        try
        {
            return NormalizeCamIndex(stoi(value), nCam);
        }
        catch(const std::exception &)
        {
            return defaultIndex;
        }
    }
    return defaultIndex;
}

static bool ReadRigExtrinsics(const string &config_path, int main_cam_index, array<Sophus::SE3f, 4> &tbc_cams)
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
    main_cam_index = NormalizeCamIndex(main_cam_index, 4);
    const Sophus::SE3f &Twc_main = twc_cams[main_cam_index];
    for (int i = 0; i < 4; ++i) {
        tbc_cams[i] = twc_cams[i].inverse() * Twc_main;
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
    if (argc != 6 && argc != 7 && argc != 8) {
        cerr << endl
             << "Usage: ./multi_fisheye_rig path_to_vocabulary "
             << "path_to_settings path_to_extrinsics "
             << "path_to_association [main_cam_index] [init_cam_index] output_dir" << endl;
        return 1;
    }

    string vocab_path = argv[1];
    string settings_path = argv[2];
    string extrinsics_path = argv[3];
    string association_path = argv[4];
    int main_cam_index = -1;
    int init_cam_index = -1;
    string output_dir;
    if (argc == 6) {
        output_dir = argv[5];
    } else if (argc == 7) {
        main_cam_index = stoi(argv[5]);
        output_dir = argv[6];
    } else {
        main_cam_index = stoi(argv[5]);
        init_cam_index = stoi(argv[6]);
        output_dir = argv[7];
    }

    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open settings: " << settings_path << endl;
    }
    bool main_cam_set = false;
    bool init_cam_set = false;
    int config_main_cam = 0;
    int config_init_cam = 0;
    if (fs.isOpened()) {
        config_main_cam = ParseRigCamIndex(fs, "Rig.main_cam", 4, 0, &main_cam_set);
        if (!main_cam_set)
            config_main_cam = ParseRigCamIndex(fs, "Rig.cam_main", 4, config_main_cam, &main_cam_set);
        config_init_cam = ParseRigCamIndex(fs, "Rig.init_cam", 4, config_main_cam, &init_cam_set);
        if (!init_cam_set)
            config_init_cam = ParseRigCamIndex(fs, "Rig.cam_init", 4, config_init_cam, &init_cam_set);
    }

    if (main_cam_index < 0)
        main_cam_index = config_main_cam;
    if (init_cam_index < 0)
        init_cam_index = init_cam_set ? config_init_cam : main_cam_index;
    main_cam_index = NormalizeCamIndex(main_cam_index, 4);
    init_cam_index = NormalizeCamIndex(init_cam_index, 4);

    cout << "Rig main camera index: " << main_cam_index << endl;
    cout << "Rig init camera index: " << init_cam_index << endl;

    vector<FrameEntry> entries;
    if (!LoadAssociation(association_path, entries)) {
        cerr << "Failed to load association file: " << association_path << endl;
        return 1;
    }

    array<Sophus::SE3f, 4> T_b_c;
    if (!ReadRigExtrinsics(extrinsics_path, main_cam_index, T_b_c)) {
        cerr << "Failed to load extrinsics file: " << extrinsics_path << endl;
        return 1;
    }

    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, false);
    SLAM.SetMainCamIndex(main_cam_index);
    SLAM.SetInitCamIndex(init_cam_index);
    //ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    Sophus::SE3f T_b_cmain = T_b_c[main_cam_index];
    Sophus::SE3f T_cmain_b = T_b_cmain.inverse();

    array<Sophus::SE3f, 4> T_r_c;
    for (int cam = 0; cam < 4; ++cam) {
        T_r_c[cam] = T_cmain_b * T_b_c[cam];
    }

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
        const int track_state = SLAM.GetTrackingState();
        if (ni % 200 == 0) {
            const auto map_points = SLAM.GetAllMapPoints();
            cout << "[Map] frame=" << ni << " state=" << track_state << " MP=" << map_points.size() << endl;
        }
        if (track_state != ORB_SLAM3::Tracking::OK) {
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

    vector<Eigen::Vector3f> map_points;
    const vector<ORB_SLAM3::MapPoint*> all_points = SLAM.GetAllMapPoints();
    map_points.reserve(all_points.size());
    for(ORB_SLAM3::MapPoint* mp : all_points)
    {
        if(!mp || mp->isBad())
            continue;
        map_points.push_back(mp->GetWorldPos());
    }

    string merged_path = output_dir + "/map_rig_fused.xyz";
    SavePoints(merged_path, map_points);
    cout << "Saved " << merged_path << " with " << map_points.size() << " points" << endl;

    return 0;
}
