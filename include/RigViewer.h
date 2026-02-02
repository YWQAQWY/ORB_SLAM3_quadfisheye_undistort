/**
 * RigViewer: lightweight Pangolin viewer for cam0 trajectory.
 */

#ifndef RIGVIEWER_H
#define RIGVIEWER_H

#include <Eigen/Core>

#include <mutex>
#include <vector>

namespace ORB_SLAM3
{

class Tracking;
class Atlas;

class RigViewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    RigViewer(Tracking* pTracking, Atlas* pAtlas);

    void Run();
    void RequestFinish();
    bool isFinished();

private:
    bool CheckFinish();
    void SetFinish();

    Tracking* mpTracker;
    Atlas* mpAtlas;

    std::vector<Eigen::Vector3f> mTrajectory;
    std::mutex mMutexTrajectory;

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;
};

}

#endif // RIGVIEWER_H
